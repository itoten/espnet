# VIVAE speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on VIVAE alone and scores it on VIVAE's own test split, classifying each
recording into one of six emotions.

[VIVAE](https://zenodo.org/records/4066235) (The Variably Intense Vocalizations of
Affect and Emotion Corpus; Holz, Larrouy-Maestri and Poeppel) is **non-speech**:
screams, cries, laughs and other vocalizations from eleven speakers, each
expressing three positive states (`achievement`, `pleasure`, `surprise`) and three
negative ones (`angry`, `fear`, `pain`) at four intensities (low, moderate, strong,
peak). Only `angry`, `fear` and `surprise` are among the seven labels shared with the
other recipes.

## What is used

The `full_set` folder of the archive: 1,085 recordings, 0.27 hours, from 11
speakers (`S01`–`S11`). The `core_set` folder is a subset of the same files and is
not used. Each recording is 0.40 to 2.00 seconds long (median 0.81) and is 44.1 kHz,
16-bit mono; `create_dataset` converts it to 16 kHz mono.

The speaker, emotion and intensity come from the file name,
`<speaker>_<emotion>_<intensity>_<number>.wav` (for example `S04_surprise_peak_10.wav`).
The emotion word is the label, except that `anger` is written `angry` as in the other
recipes.

Because the recordings are so short, no minimum length is applied in
`remove_long_short`: a 0.5 s floor would drop 120 of 1,085.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `VIVAE` — the directory holding the extracted audio, or where to download it to
- `VIVAE_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `VIVAE.zip` (93 MB) from Zenodo, checks it against the
MD5 Zenodo lists, and extracts only the `full_set` WAV files. The corpus is
CC BY-NC 4.0 (non-commercial use only). Please cite Holz, N., Larrouy-Maestri, P.,
& Poeppel, D. (2022), *Emotion* 22(1), 213–225.

## Splits

The corpus publishes no partition. Recordings are split 70/10/20 at random within
each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all 11
speakers, and the same speaker's recordings of the same emotion at different
intensities appear across splits.

| split | recordings | after `remove_long_short` |
|---|---|---|
| train | 761 | 761 |
| valid | 108 | 108 |
| test | 216 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of the 93 MB `VIVAE.zip` from Zenodo, MD5 check, extraction) | 2 min 24 s (a 30 GB download was running at the same time) |
| `create_dataset remove_long_short prepare_labels` (conversion to 16 kHz, one file at a time) | 1 min 15 s (includes starting `pixi` and Python) |

The extracted audio takes 85 MB and the converted data 33 MB. `prepare_source`
deletes the zip after a successful extraction.

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
