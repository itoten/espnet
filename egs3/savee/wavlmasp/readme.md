# SAVEE speech emotion recognition (single-corpus baseline)

Trains `wavlmasp` (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SAVEE alone and scores it on SAVEE's own test split, classifying
each utterance into one of seven emotions.

[SAVEE](https://cvssp.org/savee/) (Surrey Audio-Visual Expressed Emotion) is
English read speech from four male speakers (`DC`, `JE`, `JK`, `KL`): 480
utterances labelled `neutral` (120), `angry`, `disgust`, `fear`, `happy`, `sad`
and `surprise` (60 each).

The label comes from the file name, `<speaker>_<emotion code><index>.wav`
(e.g. `DC_a01.wav`); no annotation file is read. The recordings are 44.1 kHz,
so `create_dataset` resamples them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SAVEE` — a directory holding the WAV files, or the downloaded zip
- `SAVEE_OUTPUT` — where the resampled audio and manifests are written

## Download

The corpus needs a free registration at
[cvssp.org/savee/Download.html](https://cvssp.org/savee/Download.html), so it
cannot be fetched from a script. Put either the extracted WAV files or a single
downloaded zip in `$SAVEE`; `prepare_source` extracts the zip.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all three splits contain
all four speakers.

| split | utterances |
|---|---|
| train | 336 |
| valid | 48 |
| test | 96 |

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
