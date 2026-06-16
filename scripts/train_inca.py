"""Thin CLI wrapper around training.inca_trainer.main().

Lets the documented quick-start command work as advertised:

    python scripts/train_inca.py --config configs/inca.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

# Put repo root on sys.path so `import training.inca_trainer` resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from training.inca_trainer import main  # noqa: E402


if __name__ == "__main__":
    main()
