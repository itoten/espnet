# MCAE-SPPS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MCAE-SPPS alone and scores it on MCAE-SPPS's own test split, classifying each
utterance into one of seven emotions.

[MCAE-SPPS](https://osf.io/9jyzc/) (Mandarin Chinese Auditory Emotion database
of Subject-Personal-Pronoun Sentences, version 2025.11.30) is acted Mandarin
Chinese speech: 6 actors read sentences in `neutral` and the six basic emotions
`angry`, `disgust`, `fear`, `happy`, `sad` and `surprise`. The release holds 6,675
validated recordings, 2.30 hours in total, each 0.38 to 3.98 seconds long. The
labels are the seven shared by the other recipes.

The label comes from the file name, `<speaker><emotion><sentence>.wav` (e.g.
`33189.wav` is speaker 3, emotion 3 = anger, sentence 189). Each file also sits
in a folder named after its emotion, and `create_dataset` fails if the two
disagree. The recordings are 48 kHz mono, so `create_dataset` resamples them to
16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MCAESPPS` — the directory holding the extracted emotion folders, or where to download them to
- `MCAESPPS_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the seven per-emotion zip files (640 MB) from
[OSF](https://osf.io/9jyzc/), checks each against the MD5 OSF lists, and
extracts the WAV files. The corpus is CC BY-NC 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all three splits contain
all 6 speakers.

| split | utterances |
|---|---|
| train | 4672 (4667 after `remove_long_short`) |
| valid | 668 |
| test | 1335 |

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
