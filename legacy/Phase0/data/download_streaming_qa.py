"""Download source data for StreamingQA (CC-News).

StreamingQA (Liska et al., 2022) is built on CC-News articles.
We derive temporal open-QA probes from the same source using
make_open_qa_probe, giving us equivalent temporal structure
(monthly periods) without requiring the original dataset's annotations.

The raw source is CC-News, already downloaded by download_cc_news.py.
This script is a convenience wrapper that:
  1. Checks whether raw/cc_news/raw.jsonl exists.
  2. If not, runs download_cc_news.py automatically.
  3. Prints next-step instructions.

Preprocessing is handled entirely by preprocess_streaming_qa.py.

Run::

    python Phase0/data/download_streaming_qa.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

RAW_CC_NEWS = Path(__file__).resolve().parent / "raw" / "cc_news" / "raw.jsonl"
DOWNLOAD_CC  = Path(__file__).resolve().parent / "download_cc_news.py"


def main() -> None:
    if RAW_CC_NEWS.exists() and RAW_CC_NEWS.stat().st_size > 0:
        mb = RAW_CC_NEWS.stat().st_size / 1024 / 1024
        print(f"[download:streaming_qa] CC-News raw data already present "
              f"({mb:.0f} MB) — nothing to download.")
    else:
        print("[download:streaming_qa] CC-News raw.jsonl not found; "
              "running download_cc_news.py …")
        rc = subprocess.call([sys.executable, str(DOWNLOAD_CC)])
        if rc != 0:
            print(f"[download:streaming_qa] download_cc_news.py failed "
                  f"(exit {rc}).")
            sys.exit(rc)

    print("[download:streaming_qa] next step: "
          "python Phase0/data/preprocess_streaming_qa.py")


if __name__ == "__main__":
    main()
