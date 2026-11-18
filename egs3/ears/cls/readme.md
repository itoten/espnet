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
merged, so the recipe keeps the corpus's own 23 labels (22 emotional styles and
neutral).

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

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-18 against the per-speaker archives `p001.zip` to `p107.zip` of the `dataset` release of the GitHub repository (copies obtained before this check; their file dates are June 2024). Only the emotional recordings (`emo_*`) were counted; the other parts of the release were not examined. All counts cover every such recording in the release, before any filtering by this recipe.

### Access

GitHub release, direct download, no sign-in.

### Size of the release

| recordings | speakers | total hours |
|---|---|---|
| 4,922 | 107 | 17.95 |

How these were counted: every `pNNN/emo_<style>_{sentences,freeform}.wav` member of the 107 archives, read in place; hours from the WAV headers (1.92 to 63.99 s). All are 48 kHz mono. Every speaker has both recordings of every style: 2,461 `sentences` recordings (7.83 hours) and 2,461 `freeform` recordings (10.12 hours).

### Files per label

| style in the file name | recordings | hours |
|---|---|---|
| adoration | 214 | 0.806 |
| amazement | 214 | 0.687 |
| amusement | 214 | 0.791 |
| anger | 214 | 0.794 |
| confusion | 214 | 0.714 |
| contentment | 214 | 0.729 |
| cuteness | 214 | 0.692 |
| desire | 214 | 0.765 |
| disappointment | 214 | 0.839 |
| disgust | 214 | 0.782 |
| distress | 214 | 0.779 |
| embarassment | 214 | 0.816 |
| extasy | 214 | 0.717 |
| fear | 214 | 0.799 |
| guilt | 214 | 0.724 |
| interest | 214 | 0.754 |
| neutral | 214 | 0.761 |
| pain | 214 | 0.741 |
| pride | 214 | 0.829 |
| realization | 214 | 1.027 |
| relief | 214 | 0.764 |
| sadness | 214 | 0.835 |
| serenity | 214 | 0.802 |

How these were counted: the style field of the file name. Hours per label are measured in the same way as the total hours above and summed over the recordings of that label.

### Issues in the release

- Two styles are misspelled in the file names: `embarassment` and `extasy`. The recipe reads them as `embarrassment` and `ecstasy`.
- No two emotional recordings have byte-identical content.

### Not stated in the documentation

- Nine recordings are longer than 30 s, the limit that `remove_long_short` applies to train and valid: eight `freeform` recordings (six of them `realization`; the longest 64.0 s) and `p101/emo_realization_sentences` (31.6 s).


## Splits

The speaker split of the corpus paper (Richter et al., Interspeech 2024, Sec. 3,
written for its speech-enhancement benchmarks) is used: p001 to p099 train, p100
and p101 valid, p102 to p107 test (`split_speakers` in `dataset/config.yaml`). No
speaker is in two splits.

| split | speakers | recordings | per label |
|---|---|---|---|
| train | 99 | 4,554 | 198 |
| valid | 2 | 92 | 4 |
| test | 6 | 276 | 12 |

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
