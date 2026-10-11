# TESS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on TESS alone and scores it on TESS's own test split, classifying
each utterance into one of seven emotions.

[TESS](https://doi.org/10.5683/SP2/E8H2MF) (Toronto emotional speech set) has
two female speakers, aged 26 (`YAF`) and 64 (`OAF`), each saying 200 target
words in the carrier phrase "Say the word ___" in seven emotions: 2800
utterances labelled `angry`, `disgust`, `fear`, `happy`, `sad`, `neutral` and
`surprise` (`ps`, "pleasant surprise", in the file names).

The label comes from the file name, `<speaker>_<word>_<emotion>.wav`; no
annotation file is read. The recordings are 24.4 kHz (one is 96 kHz), so
`create_dataset` resamples all of them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `TESS` — an existing directory of the WAV files, or where to download it to
- `TESS_OUTPUT` — where the resampled audio and manifests are written

## Download

The dataset on Borealis Dataverse has no archive file. `prepare_source` asks
Dataverse to zip all 2800 files (about 270 MB) and extracts the result. The
corpus is CC BY-NC 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Neither speakers nor words are held out, so all three
splits contain both speakers, and a word in test also appears in training.

| split | utterances |
|---|---|
| train | 1960 |
| valid | 280 |
| test | 560 |

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
