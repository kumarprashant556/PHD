"""Phase 0 baseline implementations (B1 – B7).

Each module exposes a ``main()`` entry point driven by the shared
:class:`Phase0.common.runner.BaselineRunner`.  Run individual baselines::

    python -m Phase0.baselines.b1_finetune   --config Phase0/configs/base.yaml
    python -m Phase0.baselines.b2_replay     --config Phase0/configs/base.yaml
    python -m Phase0.baselines.b3_ewc        --config Phase0/configs/base.yaml
    python -m Phase0.baselines.b4_l2p        --config Phase0/configs/base.yaml
    python -m Phase0.baselines.b5_lora_moe   --config Phase0/configs/base.yaml
    python -m Phase0.baselines.b6_llama_pro  --config Phase0/configs/base.yaml
    python -m Phase0.baselines.b7_pnn        --config Phase0/configs/base.yaml

Or use the unified runner::

    python Phase0/scripts/run_baseline.py --method all --config Phase0/configs/base.yaml
    python Phase0/scripts/run_baseline.py --method naive --dataset temporalwiki
"""

BASELINES = [
    "b1_finetune",
    "b2_replay",
    "b3_ewc",
    "b4_l2p",
    "b5_lora_moe",
    "b6_llama_pro",
    "b7_pnn",
]
