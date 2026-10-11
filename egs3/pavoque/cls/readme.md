# PAVOQUE speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on this corpus alone and scores it on its own test split, classifying each
utterance into one of five speaking styles.

[PAVOQUE](https://github.com/marytts/pavoque-data) (Pretty Audio Voice Of QUality
Emotion) is German speech by one male speaker, recorded in five styles: `neutral`
(3,126 utterances, 5.4 hours), `angry` (601), `happy` (584), `sad` (556) and
`poker` (575), 5,442 utterances and 8.6 hours in all. Each is 1.2 to 20.7 seconds
long. `poker` is not one of the seven common emotions; it is kept as its own
label. There is no `disgust`, `fear` or `surprise`, and `neutral` is 57% of the
utterances.

The recordings come as one long FLAC per style, with a YAML file listing each
utterance's `prompt`, `text`, `style` and its `start`/`end` in seconds.
`create_dataset` cuts the utterances out by those times and labels each with the
YAML `style`, which must match the file it comes from. The audio is 44.1 kHz
mono; `create_dataset` resamples it to 16 kHz. The release's `pavoque-outtakes`
files are not used.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `PAVOQUE` — the directory with the FLAC and YAML files, or where to download them to
- `PAVOQUE_OUTPUT` — where the cut audio and manifests are written

## Download

`prepare_source` downloads the five FLAC and five YAML files (about 760 MB) from
release v0.2 of the GitHub repository. No checksums are published, so the
download is checked only by the presence of all ten files.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). There is only one speaker, so the splits cannot be
speaker-independent.

| split | utterances |
|---|---|
| train | 3809 |
| valid | 545 |
| test | 1088 |

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
