# EARS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the emotional recordings of EARS alone and scores it on the corpus's own
held-out speakers, classifying each recording into one of 23 labels.

[EARS](https://github.com/facebookresearch/ears_dataset) (Expressive Anechoic
Recordings of Speech) is an anechoic, fullband (48 kHz) English speech corpus from
107 speakers, made for speech enhancement and dereverberation benchmarks. Besides
read speech in several styles, every speaker recorded the same material in a set of
emotional styles. This recipe uses only those emotional recordings and none of the
other parts of the release.

For each style a speaker has two recordings:

- `emo_<style>_sentences.wav`: three sentences read in that style
- `emo_<style>_freeform.wav`: an image described freely in that style

The label is the style in the file name. The release spells two styles wrongly
(`embarassment`, `extasy`); the recipe corrects them to `embarrassment` and
`ecstasy` (`label_fix` in `dataset/config.yaml`). Nothing else is renamed or
merged, so the recipe keeps all 23 labels (22 emotional styles and neutral).
`dataset/config.yaml` also holds a proposal, not read by any code, of how some
labels could be read as the seven basic emotion classes used by multi-corpus
recipes.

`create_dataset` converts the recordings (48 kHz, 32-bit float) to 16 kHz mono
16-bit PCM with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EARS` — the directory holding the speaker folders `pNNN/` or the release's
  per-speaker archives `pNNN.zip` (directly under it or under `ARCHIVES/`), or
  where to download them to
- `EARS_OUTPUT` — where the converted audio and manifests are written

Archives are read in place; nothing is extracted.

## Download

`prepare_source` downloads the per-speaker archives `p001.zip` to `p107.zip` from
the `dataset` release of the
[GitHub repository](https://github.com/facebookresearch/ears_dataset/releases),
keeps only the `emo_*` recordings of each and deletes the archive. The whole set of
archives is large (tens of GB); if they are already present, nothing is downloaded.

## The data

TODO: filled after the first build (counts of the release).

## Splits

The speaker split of the corpus paper (Richter et al., Interspeech 2024, Sec. 3,
written for its speech-enhancement benchmarks) is used: p001 to p099 train, p100
and p101 valid, p102 to p107 test (`split_speakers` in `dataset/config.yaml`). No
speaker is in two splits.

## License and citation

CC BY-NC 4.0 (non-commercial). Please cite:

```
@inproceedings{richter2024ears,
  title     = {{EARS}: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation},
  author    = {Richter, Julius and Wu, Yi-Chiao and Krenn, Steven and Welker, Simon and Lay, Bunlong and Watanabe, Shinji and Richard, Alexander and Gerkmann, Timo},
  booktitle = {Interspeech},
  year      = {2024},
}
```

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
