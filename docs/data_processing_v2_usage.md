# CAPSEL Temporal Data Processing v2

This is the current preprocessing path for CC-News and TiC-LM.

## What It Produces

For each dataset, the v2 processor writes:

```text
datasets/<dataset>/processed/
  stream_v2/<period>.jsonl
  probes_v2/<period>.jsonl
  timeline_v2.json
  metadata_v2.json
```

The stream files are for training. Each row contains:

```json
{
  "text": "...",
  "doc_id": "cc_news_2018_H1_000001",
  "period": "2018_H1",
  "source": "cc_news",
  "date": "2018-02",
  "char_len": 1234,
  "word_count": 210,
  "title": "...",
  "url": "..."
}
```

The probe files are for evaluation/BWT matrices. Probe types currently include:

- `completion`
- `entity_cloze`
- `date_cloze`
- `salient_span_denoising`

## CC-News

Default raw file:

```text
/Users/nishantkumar/Desktop/phd/code/My project/WorkingDir/datasets/cc_news/raw/raw.jsonl
```

Smoke run:

```bash
python scripts/process_temporal_data_v2.py cc_news \
  --force \
  --max-docs-per-period 500 \
  --max-periods 2 \
  --probes-per-period 100
```

Full Phase 1 run with half-year periods:

```bash
python scripts/process_temporal_data_v2.py cc_news \
  --force \
  --cc-period-granularity half_year \
  --max-docs-per-period 20000 \
  --probes-per-period 300
```

## TiC-LM

Default raw directory:

```text
/Users/nishantkumar/Desktop/phd/code/My project/WorkingDir/datasets/tic_lm/raw/
```

Smoke run:

```bash
python scripts/process_temporal_data_v2.py tic_lm \
  --force \
  --max-docs-per-period 500 \
  --max-periods 2 \
  --probes-per-period 100
```

Full local TiC-LM run with daily periods:

```bash
python scripts/process_temporal_data_v2.py tic_lm \
  --force \
  --tic-period-granularity day \
  --probes-per-period 300
```

## Loading v2 Streams

The existing loaders support v2 as an opt-in:

```python
from data import load_periods

cc_news = load_periods("cc_news", processed_version="v2", n_per_period=20000)
tic_lm = load_periods("tic_lm", processed_version="v2", n_per_period=3000)
```

Without `processed_version="v2"`, loaders keep their previous behavior.

## Legacy Compatibility

To overwrite the old loader-facing folders as well:

```bash
python scripts/process_temporal_data_v2.py cc_news --force --write-legacy-copy
```

This refreshes:

```text
processed/stream/
processed/probes/
processed/timeline.json
processed/metadata.json
```

Use this only when you intentionally want the old default loaders to consume the
new v2 files.
