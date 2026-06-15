from __future__ import annotations

import json
from pathlib import Path

from data.temporal_processing_v2 import process_cc_news_raw, process_tic_lm_raw


def _long_text(name: str, year: int = 2018) -> str:
    sentence = (
        f"{name} Corporation appointed Alice Johnson as chief scientist in {year}. "
        f"The announcement said Alice Johnson would lead research teams in London "
        f"and support product planning for the next release. "
    )
    return sentence * 8


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_process_cc_news_raw_v2(tmp_path: Path) -> None:
    raw = tmp_path / "cc_raw.jsonl"
    rows = [
        {
            "date": "2018-02",
            "title": "Alpha update",
            "text": _long_text("Alpha", 2018),
            "url": "https://example.test/a",
        },
        {
            "date": "2018-09",
            "title": "Beta update",
            "text": _long_text("Beta", 2018),
            "url": "https://example.test/b",
        },
    ]
    raw.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "processed"
    meta = process_cc_news_raw(
        raw_path=raw,
        output_root=out,
        probes_per_period=4,
        min_words=20,
        min_chars=80,
        min_sentences=2,
        force=True,
    )

    assert meta["timeline"] == ["2018_H1", "2018_H2"]
    assert (out / "timeline_v2.json").exists()
    assert (out / "metadata_v2.json").exists()

    h1_docs = _read_jsonl(out / "stream_v2" / "2018_H1.jsonl")
    h1_probes = _read_jsonl(out / "probes_v2" / "2018_H1.jsonl")
    assert h1_docs[0]["period"] == "2018_H1"
    assert h1_docs[0]["source"] == "cc_news"
    assert h1_docs[0]["text"]
    assert {probe["origin_period"] for probe in h1_probes} == {"2018_H1"}
    assert any(probe["probe_type"] == "completion" for probe in h1_probes)
    assert any(probe["probe_type"] == "entity_cloze" for probe in h1_probes)


def test_process_tic_lm_raw_v2(tmp_path: Path) -> None:
    raw_dir = tmp_path / "tic_raw"
    raw_dir.mkdir()
    path = raw_dir / "2019-04-18.jsonl"
    path.write_text(
        json.dumps(
            {
                "period": "period_2019-04-18",
                "date": "2019-04-18",
                "text": _long_text("Gamma", 2019),
                "doc_id": "raw-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "processed"
    meta = process_tic_lm_raw(
        raw_dir=raw_dir,
        output_root=out,
        probes_per_period=4,
        min_words=20,
        min_chars=80,
        min_sentences=2,
        force=True,
    )

    assert meta["timeline"] == ["period_2019-04-18"]
    docs = _read_jsonl(out / "stream_v2" / "period_2019-04-18.jsonl")
    probes = _read_jsonl(out / "probes_v2" / "period_2019-04-18.jsonl")
    assert docs[0]["source"] == "tic_lm"
    assert docs[0]["period"] == "period_2019-04-18"
    assert docs[0]["raw_id"] == "raw-1"
    assert probes
    assert {probe["origin_period"] for probe in probes} == {"period_2019-04-18"}
