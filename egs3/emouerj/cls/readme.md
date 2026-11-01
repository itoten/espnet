# emoUERJ speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on emoUERJ alone and scores it on emoUERJ's own test split, classifying each
utterance into one of four emotions.

[emoUERJ](https://zenodo.org/records/5427549) (Germano et al., 2021) is a Brazilian
Portuguese acted corpus: 8 actors (4 women, 4 men) read sentences chosen from a
set of ten in four emotions, `angry`, `happy`, `neutral` and `sad`: 377
utterances (94, 91, 92 and 100), 0.32 hours in total, each 1.2 to 7.1 seconds
long. The corpus is CC BY 4.0.

The label comes from the letter after the actor code in the file name,
`<m|w><actor><emotion><take>.wav` (e.g. `w02h16.wav` is happy, from woman 2);
`dataset/config.yaml` maps `h`, `a`, `s` and `n` to labels. The recordings are
44.1 kHz stereo, so `create_dataset` converts them to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOUERJ` — the directory holding the extracted `emoUERJ/` wav files, or where to download them to
- `EMOUERJ_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 5427549 (emoUERJ, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 377 | 8 | 0.32 |

How these were counted: every `.wav` in the archive (44.1 kHz stereo, 1.2 to 7.1 s); hours from the file lengths. Speakers are the 8 distinct actor codes in the file names (`m01` to `m04`, `w01` to `w04`: four men, four women).

### Files per label

| emotion letter in the file name | read as | samples |
|---|---|---|
| a | angry | 94 |
| h | happy | 91 |
| n | neutral | 92 |
| s | sad | 100 |

How these were counted: the letter after the actor code in `<m|w><actor><emotion><take>.wav`.

### Issues in the release

- The record lists the ten sentences that were read, but the release does not say which file contains which sentence.
- The left and right channels differ in 115 files (correlation 0.86 to 1.0); in 262 files they are identical.


## Download

`prepare_source` downloads `emoUERJ.zip` (105 MB, md5-checked) from
[Zenodo](https://zenodo.org/records/5427549) and extracts the 377 wav files. Measured on one run: about 4.5 minutes to download;
`create_dataset`, `remove_long_short` and `prepare_labels` together took about 48 seconds.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`), so all three splits contain all 8 actors.

| split | utterances |
|---|---|
| train | 265 |
| valid | 37 |
| test | 75 |

No clip is shorter than the 0.2 s floor of `remove_long_short`.

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
