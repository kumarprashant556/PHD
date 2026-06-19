"""Self-zip: bundles paper1/ into paper1_cuda_run.zip for GPU deployment.

Run from inside paper1/:
    python zip_me.py

Creates paper1_cuda_run.zip in the parent directory (WorkingDir/).
"""

import zipfile
from pathlib import Path

HERE    = Path(__file__).resolve().parent   # paper1/
ZIP_OUT = HERE.parent / "paper1_cuda_run.zip"

FILES = [
    # ── models ──────────────────────────────────────────────────────────────
    "models/__init__.py",
    "models/inca/__init__.py",
    "models/inca/config.py",
    "models/inca/layer_manager.py",
    "models/inca/plateau.py",
    "models/inca/cka.py",
    "models/inca/replay.py",
    "models/inca/selectors.py",
    "models/inca/uclbr.py",
    "models/inca/lateral.py",

    # ── data ────────────────────────────────────────────────────────────────
    "data/__init__.py",
    "data/_base.py",
    "data/domain_sequential.py",
    "data/tokenizer.py",

    # ── training ────────────────────────────────────────────────────────────
    "training/__init__.py",
    "training/inca_trainer.py",
    "training/memory_tracker.py",

    # ── baselines ───────────────────────────────────────────────────────────
    "baselines/__init__.py",
    "baselines/b6_llama_pro.py",
    "baselines/_runtime/__init__.py",
    "baselines/_runtime/data.py",
    "baselines/_runtime/eval.py",
    "baselines/_runtime/logging_setup.py",
    "baselines/_runtime/precision.py",
    "baselines/_runtime/runner.py",
    "baselines/_runtime/trainer_factory.py",

    # ── evaluation ──────────────────────────────────────────────────────────
    "evaluation/__init__.py",
    "evaluation/metrics.py",

    # ── entry points ────────────────────────────────────────────────────────
    "run.py",
    "infer_inca.py",

    # ── configs ─────────────────────────────────────────────────────────────
    "configs/base.yaml",
    "configs/inca.yaml",
    "configs/main.yaml",
    "configs/main_cuda.yaml",
    "configs/baselines/b6_main.yaml",
    "configs/baselines/b6_main_cuda.yaml",
    "configs/ablations/e_cls3.yaml",
    "configs/ablations/e_prim.yaml",
    "configs/ablations/e_route.yaml",
    "configs/ablations/e_sat.yaml",
    "configs/ablations/e_scope.yaml",
    "configs/ablations/e_timing.yaml",

    # ── deps ────────────────────────────────────────────────────────────────
    "requirements.txt",
    "requirements_cuda.txt",
]

missing = [f for f in FILES if not (HERE / f).exists()]
if missing:
    print("ERROR — missing files:")
    for m in missing:
        print(f"  {m}")
    raise SystemExit(1)

with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for rel in FILES:
        src = HERE / rel
        arcname = f"paper1/{rel}"
        zf.write(src, arcname)
        print(f"  + {arcname}")

print(f"\nCreated: {ZIP_OUT}  ({ZIP_OUT.stat().st_size / 1024:.0f} KB)")
print(f"Files  : {len(FILES)}")
print()
print("Deploy on GPU server:")
print("  unzip paper1_cuda_run.zip")
print("  cd paper1")
print("  pip install torch --index-url https://download.pytorch.org/whl/cu121")
print("  pip install -r requirements_cuda.txt")
print("  python run.py --group main")
