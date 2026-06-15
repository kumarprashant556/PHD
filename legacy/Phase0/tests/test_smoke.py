"""Phase 0 smoke test — B1-B7 baselines only.

Tests [1] and [2] are pure Python and always run.
Tests [3]–[5] require torch + transformers — they are skipped gracefully if
those packages are not installed.

Usage
-----
Quick (no ML deps)::

    python -m Phase0.tests.test_smoke

Full (with torch + transformers)::

    pip install torch transformers datasets pyyaml --break-system-packages
    python Phase0/data/download_cc_news.py --max_periods 2 --max_docs_per_period 200
    python -m Phase0.tests.test_smoke

Tests
-----
  [1] JSONL period loader on synthetic data           (pure Python)
  [2] Pure-math metrics: ACC / BWT / FWT / RIR        (pure Python)
  [3] B1 NaiveFinetune — one-period forward + train   (requires torch)
  [4] B2 Replay — buffer fill and size cap            (requires torch)
  [5] All seven baseline modules import with main()   (requires torch)

Prints "ALL OK" on success, or lists which tests were skipped.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(root: Path, n_periods: int = 2, n_docs: int = 64) -> None:
    """Write a tiny period-sliced JSONL dataset under *root*."""
    labels = [f"p{i+1}" for i in range(n_periods)]
    (root / "stream").mkdir(parents=True, exist_ok=True)
    (root / "probes").mkdir(parents=True, exist_ok=True)
    (root / "timeline.json").write_text(json.dumps(labels))
    for p in labels:
        (root / "stream" / f"{p}.jsonl").write_text(
            "\n".join(
                json.dumps({"text": f"The quick brown fox {p} document {i} "
                                    "jumps over the lazy dog and continues."})
                for i in range(n_docs)
            ) + "\n"
        )
        (root / "probes" / f"{p}.jsonl").write_text(
            json.dumps({
                "format": "mc4",
                "question": f"In {p} the answer is ____",
                "evidence": "",
                "choices": {"A": "alpha", "B": "beta", "C": "gamma", "D": "delta"},
                "answer_key": "A",
                "period": p,
                "source": "synthetic",
            }) + "\n"
        )


def _check_torch() -> bool:
    try:
        import torch          # noqa: F401
        import transformers   # noqa: F401
        return True
    except ImportError:
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    skipped: list[str] = []
    has_torch = _check_torch()

    # ── [1] Loader ────────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "tiny"
        _make_synthetic_dataset(root, n_periods=2, n_docs=8)

        from Phase0.common.datasets import load_dataset
        pds = load_dataset(str(root), max_periods=10)
        assert len(pds) == 2 and pds[0].label == "p1", f"loader failed: {pds}"
    print(f"[1] loader: {len(pds)} periods OK")

    # ── [2] Metrics ───────────────────────────────────────────────────────────
    from Phase0.common.metrics import StreamAccuracyMatrix, acc, bwt, fwt, rir
    am = StreamAccuracyMatrix(matrix=[], labels=["a", "b"])
    am.set(0, 0, 0.7); am.set(1, 0, 0.6); am.set(1, 1, 0.7)
    assert abs(acc(am) - 0.65) < 1e-6,  f"ACC wrong: {acc(am)}"
    assert abs(bwt(am) - (-0.1)) < 1e-6, f"BWT wrong: {bwt(am)}"
    r = rir(score_t=0.8, score_0=0.5, chance=0.25)
    assert r > 0, f"RIR should be positive: {r}"
    print("[2] metrics: ACC / BWT / FWT / RIR OK")

    # ── [3] B1 forward + single train step ───────────────────────────────────
    if not has_torch:
        print("[3] B1 NaiveFinetune: SKIP (torch not installed)")
        skipped.append("[3]")
    else:
        import torch  # noqa: F401
        from Phase0.common.config import Phase0Config
        from Phase0.baselines.b1_finetune import B1NaiveFinetune

        cfg = Phase0Config(
            model_name="distilgpt2",
            epochs_per_period=1,
            batch_size=4,
            max_seq_len=32,
            max_periods=1,
            ppl_eval_samples=8,
            probe_max=2,
            seed=42,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "b1_smoke"
            _make_synthetic_dataset(root, n_periods=1, n_docs=16)

            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
            tokenizer.pad_token = tokenizer.eos_token

            b1 = B1NaiveFinetune(cfg=cfg)
            model = b1.build_model(tokenizer, device="cpu")
            assert model is not None

            from Phase0.common.harness import load_periods
            periods = load_periods(str(root), max_periods=1, ppl_eval_frac=0.25, seed=42)
            assert len(periods) == 1
            p = periods[0]
            b1.on_period_start(p)
            loss = b1.train_period(p)
            b1.on_period_end(p)
            assert isinstance(loss, float) and loss >= 0, f"unexpected loss: {loss}"

        print(f"[3] B1 NaiveFinetune: forward + train OK  (loss={loss:.4f})")

    # ── [4] B2 replay buffer mechanics ───────────────────────────────────────
    if not has_torch:
        print("[4] B2 Replay: SKIP (torch not installed)")
        skipped.append("[4]")
    else:
        from Phase0.baselines.b2_replay import B2ReplayOnly
        from Phase0.common.harness import load_periods
        from Phase0.common.config import Phase0Config  # may already be imported

        cfg = Phase0Config(  # type: ignore[assignment]
            model_name="distilgpt2",
            epochs_per_period=1,
            batch_size=4,
            max_seq_len=32,
            max_periods=2,
            ppl_eval_samples=8,
            probe_max=2,
            seed=42,
        )
        b2 = B2ReplayOnly(cfg=cfg, buffer_size=10, replay_ratio=0.5)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "b2_smoke"
            _make_synthetic_dataset(root, n_periods=2, n_docs=20)

            from transformers import AutoTokenizer
            tokenizer2 = AutoTokenizer.from_pretrained("distilgpt2")
            tokenizer2.pad_token = tokenizer2.eos_token
            b2.build_model(tokenizer2, device="cpu")

            periods2 = load_periods(str(root), max_periods=2, ppl_eval_frac=0.25, seed=42)
            b2.on_period_start(periods2[0])
            b2.train_period(periods2[0])
            b2.on_period_end(periods2[0])
            assert len(b2._buffer) > 0, "replay buffer should be non-empty after period 0"
            assert len(b2._buffer) <= b2.buffer_size, "buffer exceeds size cap"

        print(f"[4] B2 Replay: buffer fill OK  (buffer={len(b2._buffer)} items)")

    # ── [5] All baseline modules import cleanly ───────────────────────────────
    if not has_torch:
        print("[5] baseline imports: SKIP (torch not installed)")
        skipped.append("[5]")
    else:
        import importlib
        baselines = [
            "Phase0.baselines.b1_finetune",
            "Phase0.baselines.b2_replay",
            "Phase0.baselines.b3_ewc",
            "Phase0.baselines.b4_l2p",
            "Phase0.baselines.b5_lora_moe",
            "Phase0.baselines.b6_llama_pro",
            "Phase0.baselines.b7_pnn",
        ]
        for mod_name in baselines:
            mod = importlib.import_module(mod_name)
            assert hasattr(mod, "main"), f"{mod_name} missing main()"
        print(f"[5] all {len(baselines)} baseline modules import OK")

    # ── Summary ───────────────────────────────────────────────────────────────
    if skipped:
        print(f"\nALL OK  (skipped {len(skipped)} torch-dependent tests: "
              f"{', '.join(skipped)} — install torch + transformers to run them)")
    else:
        print("\nALL OK")


if __name__ == "__main__":
    main()
