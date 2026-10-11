# YSED speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on YSED alone and scores it on YSED's own test split, classifying each
recording into one of five emotions.

[YSED](https://zenodo.org/records/15227219) (Yemeni Speech Emotion Dataset;
Derhem, AL-Mekhlafi, AL-Majmar and AL-Makhlafi, *Data in Brief* 63, 2025,
[doi:10.1016/j.dib.2025.112233](https://doi.org/10.1016/j.dib.2025.112233)) is
Yemeni Arabic speech. The Zenodo record describes 1,432 recordings from 71 male
and female participants aged 15 to 40. The emotions are `angry`, `fear`
(`fearful` in the release), `happy`, `neutral` and `sad`; there is no `disgust` or
`surprise`.

## What is used

All 1,432 recordings (1.01 hours). The label is the name of the folder the file
is in; the file name's first letter (`a`, `f`, `h`, `n`, `s`) is checked against it.
The release is 48 kHz stereo; `create_dataset` converts it to 16 kHz mono 16-bit
WAV. Durations range from 1.0 to 7.5 seconds (median 2.3), so
`remove_long_short` drops nothing. The release does not say who spoke which
recording, so the speaker is not known.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `YSED` — the directory holding the `YSED/<emotion>/` folders, or where to download it to
- `YSED_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `YSED-v1.0.0.zip` (506 MB) from Zenodo, checks it against
the MD5 Zenodo lists, drops the archive's top-level folder while extracting, and
counts the 1,432 files. The zip is deleted afterwards. The release is CC BY 4.0.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Because speakers are unknown they cannot be held out, so
the splits probably share speakers.

| split | recordings | after `remove_long_short` |
|---|---|---|
| train | 1003 | 1003 |
| valid | 142 | 142 |
| test | 287 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network, and another large download ran on the same connection
for part of this one, so treat it as an upper bound.

| step | time |
|---|---|
| `prepare_source` (download of the 506 MB zip, MD5 check, extraction) | 10 min 10 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 88.6 s |
| `remove_long_short` | 18.2 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 55 s (includes starting `pixi` and Python) |

The extracted audio takes 667 MB and the converted data 114 MB.

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
