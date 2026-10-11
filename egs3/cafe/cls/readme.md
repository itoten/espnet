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
