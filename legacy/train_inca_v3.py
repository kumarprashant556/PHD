import torch
from transformers import AutoTokenizer
from torch.optim import AdamW

from inca_dataloader import INCALoader
from inca_eval import RealTimeQAEvaluator
from inca_plateau import PlateauDetector
from inca_replay import ExperienceReplay
from inca_model_v2 import INCA_GPT2
from inca_qa_loss import QALoss, CombinedLoss
import os
import json

# Config
CONFIG = {
    "data_root": "/Users/nishantkumar/Desktop/phd/code/My project/WorkingDir/realtimeqa",
    "model_name": "gpt2",  # Upgraded from distilgpt2 (355M vs 66M params)
    "output_dir": "results/inca_v3",
    "batch_size": 4,
    "lr": 1e-4,
    "epochs_per_week": 30, 
    "max_grad_norm": 1.0,
    "replay_ratio": 0.25,
    "buffer_size": 500,
    "plateau_threshold": 0.65,
    "plateau_patience": 7,
    "selector_type": "cross_attn", 
    "device": "mps" #if torch.cuda.is_available() else "cpu"
}

def main():
    print(f"--- INCA-2.0 v3: {CONFIG['selector_type']} Selector ---")
    
    # Suppress tokenizer warning
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    model = INCA_GPT2(CONFIG['model_name'], selector_type=CONFIG['selector_type'])
    model.to(CONFIG['device'])
    
    # Aggregate 12 weeks (~3 months) of probes to get 240+ probes per eval period
    # This reduces accuracy granularity from 5% (20 probes) to 0.42% (240 probes)
    loader = INCALoader(CONFIG['data_root'], tokenizer, batch_size=CONFIG['batch_size'], aggregate_weeks=12)
    evaluator = RealTimeQAEvaluator(model, tokenizer, CONFIG['device'])
    qa_loss = QALoss(tokenizer, model, CONFIG['device'])
    combined_loss_fn = CombinedLoss(alpha=0.8, beta=0.2)
    detector = PlateauDetector(CONFIG)
    replay = ExperienceReplay(capacity=CONFIG['buffer_size'])
    
    def get_optimizer(model):
        params = list(model.layer_manager.current_block.parameters()) + \
                 list(model.layer_manager.selector_head.parameters())
        return AdamW(params, lr=CONFIG['lr'])

    optimizer = get_optimizer(model)
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    results_log = []

    all_probes = [] # For stability evaluation
    
    print("Starting Stream...")
    for step_idx, (week_id, train_loader, probes) in enumerate(loader):
        if step_idx > 6:
            break;
        print(f"\n=== Week {week_id} ===")
        #print(f"  Training Samples: {train_loader.dataset.examples} | Probe Samples: {probes}")
        model.train()
        
        # 1. Update Replay Buffer
        if len(train_loader) > 0:
            raw_texts = train_loader.dataset.examples
            replay.add(raw_texts)
        
        current_acc = 0.0 # Track accuracy for logging

        if len(train_loader) > 0:
            for epoch in range(CONFIG['epochs_per_week']):
                epoch_loss = 0
                epoch_qa_loss = 0
                for batch in train_loader:
                    # Train on current batch
                    batch = {k: v.to(CONFIG['device']) for k, v in batch.items() if k != 'labels'}
                    optimizer.zero_grad()
                    out = model(**batch, labels=batch['input_ids'])
                    lm_loss = out.loss
                    
                    # Calculate QA loss every 3 epochs for efficiency
                    if epoch % 3 == 0 and len(probes) > 0:
                        qa_loss_val = qa_loss(probes)
                        combined = combined_loss_fn(lm_loss, qa_loss_val)
                        combined.backward()
                        epoch_qa_loss += qa_loss_val.item()
                    else:
                        lm_loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                    optimizer.step()
                    epoch_loss += lm_loss.item()
                    
                    # Replay buffer training (prevent forgetting)
                    if len(replay) > 0 and epoch % 2 == 0:  # Every other epoch
                        replay_size = max(1, int(CONFIG['batch_size'] * CONFIG['replay_ratio']))
                        replay_texts = replay.sample(replay_size)
                        
                        # Tokenize replay samples
                        replay_enc = tokenizer(
                            replay_texts,
                            truncation=True,
                            max_length=512,
                            padding="max_length",
                            return_tensors="pt"
                        )
                        replay_batch = {
                            'input_ids': replay_enc['input_ids'].to(CONFIG['device']),
                            'attention_mask': replay_enc['attention_mask'].to(CONFIG['device'])
                        }
                        
                        optimizer.zero_grad()
                        replay_out = model(**replay_batch, labels=replay_batch['input_ids'])
                        replay_out.loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                        optimizer.step()
                
                avg_loss = epoch_loss / len(train_loader)

                # --- NEW: Evaluate mid-training for Detector signal ---
                model.eval()
                # Combine current probes with all past probes for stability
                all_probes = all_probes + probes
                current_acc = evaluator.evaluate_week(probes)
                model.train()

                # Check Plateau with REAL Accuracy
                metrics = {'loss': avg_loss, 'accuracy': current_acc} 
                score, triggered = detector.update(metrics, model.layer_manager.current_block)
                
                print(f"   Ep {epoch+1}: Loss {avg_loss:.4f} | Acc {current_acc:.2%} | Saturation: {score:.4f}")
                
                if triggered:
                    print(f"   >>> [PLATEAU] Score {score:.4f} -> FREEZING & GROWING!")
                    model.trigger_growth()
                    detector.reset()
                    optimizer = get_optimizer(model)

        # Final Post-Week Eval (Redundant if last epoch eval is used, but good for logging consistency)
        print(f"   Final Probe Acc: {current_acc:.2%}")
        torch.save(model,f"{CONFIG['output_dir']}/inca_week{week_id}.pt")
    
        results_log.append({
            "week": week_id, "acc": current_acc,
            "frozen_blocks": len(model.layer_manager.frozen_blocks)
        })
        with open(f"{CONFIG['output_dir']}/log.json", "w") as f:
            json.dump(results_log, f, indent=2)

    # Generate visualizations after training completes
    print("\n" + "="*60)
    print("TRAINING COMPLETE - GENERATING VISUALIZATIONS")
    print("="*60)
    
    from visualize_model import INCAVisualizer
    visualizer = INCAVisualizer(model, CONFIG)
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()