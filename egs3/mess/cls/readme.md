# MESS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MESS alone and scores it on MESS's own test split, classifying each
utterance into one of four emotions.

[MESS](https://zenodo.org/records/3813437) (Morgan Emotional Speech Set;
Morgan, 2019) is English speech from 6 talkers (3 female, 3 male) in four
emotional speaking styles, `angry`, `calm`, `happy` and `sad`: 1800 utterances,
450 per style, 1.33 hours in total, each 1.7 to 3.6 seconds long. Each
utterance is one of the sentence frames "Michael / Theo / Victor discussed the X
and the Y today" with different words filled in. `calm` is not one of the seven
labels shared by the other recipes (`angry`, `disgust`, `fear`, `happy`,
`neutral`, `sad`, `surprise`) and is kept as its own label.

The label comes from the file name,
`<emotion><gender><speaker><frame><item>_SCR.wav` (e.g. `AF1M01_SCR.wav` is
angry, female talker 1, frame M, item 01). The recordings are 24.4 kHz mono,
so `create_dataset` resamples them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MESS` — the extracted WAV files, or where to download them to
- `MESS_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads `MESS compressed.zip` (192 MB) from
[Zenodo](https://zenodo.org/records/3813437). The corpus is CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Talkers are not held out, so all three splits contain
all 6 talkers.

| split | utterances |
|---|---|
| train | 1260 |
| valid | 180 |
| test | 360 |

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
