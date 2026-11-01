# CaFE speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on CaFE alone and scores it on CaFE's own test split, classifying each
utterance into one of seven emotions.

[CaFE](https://zenodo.org/records/1478765) (Canadian French Emotional speech
dataset) is Canadian French read speech from 12 actors (odd-numbered male,
even-numbered female), each reading the same six sentences: 936 utterances in
seven emotions, `angry`, `disgust`, `happy`, `fear`, `surprise`, `sad`
(144 each) and `neutral` (72). Every emotion except `neutral` is recorded at
two intensities, low and high; both are used, and `neutral` has half as many
utterances because it was recorded at one intensity only.

The label comes from the file name, `<actor>-<emotion>-<intensity>-<sentence>.wav`
(e.g. `07-P-2-3.wav`); no annotation file is read. The release nests the files
under French emotion and intensity directories, which are not parsed. The
recordings are 48 kHz mono, so `create_dataset` resamples them to 16 kHz with
ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CAFE` — an extracted `CaFE_48k.zip`, or where to download it to
- `CAFE_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 1478765 (CaFE, Canadian French Emotional speech dataset, version 1.1 of 2018-11-06, `CaFE_48k.zip`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 936 | 12 | 1.15 |

How these were counted: every `.wav` in the archive (48 kHz mono 16-bit, 2.56 to 7.74 s); hours from the file lengths. Speakers are the 12 actor numbers in the file names `<actor>-<emotion>-<intensity>-<sentence>.wav` (odd numbers male, even numbers female; six sentences).

### Files per label

| label (French emotion directory) | code in the file name | samples |
|---|---|---|
| Colère | C | 144 |
| Dégoût | D | 144 |
| Joie | J | 144 |
| Peur | P | 144 |
| Surprise | S | 144 |
| Tristesse | T | 144 |
| Neutre | N | 72 |

How these were counted: the emotion directory of each file (the letter in the file name agrees). Every emotion except neutral was recorded at two intensities, low and high; neutral at one intensity only, hence half as many files.

### Issues in the release

- Version 1.1 corrected the swapped actors 11 and 12 and updated `01-D-2-6.wav`. A 192 kHz version (two zip files) holds the same recordings and is not counted here.
- Every combination of emotion, intensity, actor and sentence is present and no duplicate was found.


## Download

`prepare_source` downloads `CaFE_48k.zip` (187 MB, the 48 kHz / 16-bit
release) from [Zenodo](https://zenodo.org/records/1478765). The 192 kHz /
24-bit release holds the same recordings and is not used. The corpus is
CC BY-NC-SA 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Actors are not held out, so all three splits contain
all 12 actors.

| split | utterances |
|---|---|
| train | 657 |
| valid | 91 |
| test | 188 |

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
