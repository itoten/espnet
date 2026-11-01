# nEMO speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on nEMO alone and scores it on nEMO's own test split, classifying each
utterance into one of six emotions.

[nEMO](https://huggingface.co/datasets/amu-cai/nEMO) is Polish read speech
from nine speakers (five male, four female; three are qualified voice actors and
six are not professionals, which the paper says made no great difference in listening or
in acoustic analysis): 4481 utterances in six emotions,
`neutral` (809), `sad` (769), `angry` (749), `happy` (749), `fear` (736) and
`surprise` (669). The speakers contribute unequally, from 256 to 540 utterances
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

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `amu-cai/nEMO` (two parquet shards; repository revision `140015d56f6adaad7e453dc471ba7c2856c02c8e`, last modified 2024-05-28; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 4,481 | 9 | 3.0 (approx.) |

How these were counted: every row of the two parquet shards (24 kHz mono 16-bit, 0.87 to 5.66 s); the hours are the figure from the corpus description (about 3), not re-measured. Speakers are the 9 distinct `speaker_id` values (`EB0` 539, `IC0` 540, `KD0` 540, `KS0` 540, `MK0` 450, `MP0` 256, `SM0` 538, `TP0` 539, `WR0` 539 rows; five male and four female, aged 23 to 28; three are qualified voice actors and six are not professionals, per the paper).

### Files per label

| label (the `emotion` column) | samples |
|---|---|
| neutral | 809 |
| sadness | 769 |
| anger | 749 |
| happiness | 749 |
| fear | 736 |
| surprised | 669 |

How these were counted: the `emotion` column of all rows.

### Issues in the release

- The speakers contribute unequally: `MP0` has 256 rows with an uneven emotion distribution and `MK0` has no `surprised` recording.
- According to the paper, recordings whose emotion was doubted by human evaluation were removed and only those that conveyed the target emotion were kept; the number of evaluators and the method are not given.


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

## Listening test

Doubtful recordings were removed after a human evaluation, so that only recordings expressing the intended emotion remain (paper). The number of evaluators and the method are not stated and the ratings are not distributed.

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
