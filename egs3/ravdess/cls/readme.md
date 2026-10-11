# RAVDESS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on RAVDESS alone and scores it on RAVDESS's own test split, classifying
each utterance into one of eight emotions.

[RAVDESS](https://zenodo.org/records/1188976) is English read speech from 24
actors (odd-numbered male, even-numbered female). This recipe
uses the speech, audio-only release: 1440 utterances in eight emotions,
`neutral`, `calm`, `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`.
`neutral` has half as many utterances as the others because it was recorded
at one intensity only.

The label comes from the file name,
`<modality>-<vocal channel>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav`,
all two-digit codes; no annotation file is read. The recordings are 48 kHz and
five of them are stereo, so `create_dataset` converts all of them to 16 kHz
mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `RAVDESS` — an existing `Actor_NN/` tree, or where to download it to
- `RAVDESS_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads `Audio_Speech_Actors_01-24.zip` (208 MB) from
[Zenodo](https://zenodo.org/records/1188976). The corpus is CC BY-NC-SA 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Actors are not held out, so all three splits contain
all 24 actors.

| split | utterances |
|---|---|
| train | 1012 |
| valid | 143 |
| test | 285 |

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
