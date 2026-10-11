# nEMO speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on nEMO alone and scores it on nEMO's own test split, classifying each
utterance into one of six emotions.

[nEMO](https://huggingface.co/datasets/amu-cai/nEMO) is Polish read speech
from nine actors (five male, four female): 4481 utterances in six emotions,
`neutral` (809), `sad` (769), `angry` (749), `happy` (749), `fear` (736) and
`surprise` (669). The actors contribute unequally, from 256 to 540 utterances
each.

The release is two parquet shards on Hugging Face; each row holds the
recording as WAV bytes along with its `emotion` and `speaker_id`. The label
comes from the `emotion` column, renamed (`anger` to `angry`, `happiness` to
`happy`, `sadness` to `sad`, `surprised` to `surprise`). The recordings are
24 kHz mono, so `create_dataset` pipes each one through ffmpeg to a 16 kHz WAV
file.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `NEMO` — a directory holding the two shards (`data/train-*.parquet`), or
  where to download them to
- `NEMO_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the two parquet shards (531 MB in total) from
[Hugging Face](https://huggingface.co/datasets/amu-cai/nEMO), pinned to one
revision in `dataset/config.yaml`. The dataset is not gated. The corpus is
CC BY-NC-SA 4.0.

## Splits

The release is a single `train` split with no partition. Utterances are split
70/10/20 at random within each label (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`). Speakers are not held
out, so all three splits contain all nine speakers.

| split | utterances |
|---|---|
| train | 3135 |
| valid | 449 |
| test | 897 |

## Quick start

Run the three commands separately. With several GPUs, `train` starts one
process per GPU by re-running its own command, so any other stage listed in the
same command would run once per GPU as well.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    collect_stats --training_config conf/training.yaml
python run.py --stages train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
