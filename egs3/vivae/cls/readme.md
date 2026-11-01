# VIVAE speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on VIVAE alone and scores it on VIVAE's own test split, classifying each
recording into one of six emotions.

[VIVAE](https://zenodo.org/records/4066235) (The Variably Intense Vocalizations of
Affect and Emotion Corpus; Holz, Larrouy-Maestri and Poeppel) is **non-speech**:
screams, cries, laughs and other vocalizations from eleven speakers, each
expressing three positive states (`achievement`, `pleasure`, `surprise`) and three
negative ones (`angry`, `fear`, `pain`) at four intensities (low, moderate, strong,
peak). Only `angry`, `fear` and `surprise` are among the seven labels shared with the
other recipes.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 4066235 (VIVAE, Variably Intense Vocalizations of Affect and Emotion Corpus; the archive holds `full_set` and `core_set`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,085 | 11 | 0.27 |

How these were counted: every `.wav` in `full_set` (44.1 kHz mono 16-bit, 0.40 to 2.00 s, median 0.81 s); hours from the file lengths. Speakers are `S01` to `S11` (`S11` has 54 files, the others 87 to 111; every speaker has all six states). The recordings are non-speech vocalizations (screams, cries, laughs). `core_set` (480 files) holds files that are all also in `full_set` and was not counted separately.

### Files per label

| label (emotion word in the file name) | samples |
|---|---|
| pleasure | 202 |
| surprise | 187 |
| pain | 185 |
| fear | 176 |
| anger | 174 |
| achievement | 161 |

How these were counted: the emotion word of each file name `<speaker>_<emotion>_<intensity>_<number>.wav`; four intensities (low 262, moderate 269, strong 282, peak 272). `pleasure` denotes sexual pleasure in the description.

### Issues in the release

- `full_set` contains 605 recordings that are not in `core_set`. The core set (6 emotions x 4 intensities x 10 speakers x 2 items = 480 files; speakers `S01` to `S10` with 48 files each, none from `S11`) was selected by the authors "based on judged authenticity"; the number of raters and the cut-off of this selection are not stated in the sources read.
- No duplicate recordings were found.

### Listening test ratings in the release

The 480 core-set recordings were rated by 90 participants (30 per task, native German speakers, Sci Rep 2021): a seven-alternative choice with intensity and authenticity ratings, per-emotion clarity ratings, and valence and arousal ratings with authenticity. The individual responses are on OSF (osf.io/jmh5t), not in the Zenodo archive.


## What is used

The `full_set` folder of the archive: 1,085 recordings, 0.27 hours, from 11
speakers (`S01`–`S11`). The `core_set` folder is a subset of the same files and is
not used (see "Core set" below). Each recording is 0.40 to 2.00 seconds long (median 0.81) and is 44.1 kHz,
16-bit mono; `create_dataset` converts it to 16 kHz mono.

The speaker, emotion and intensity come from the file name,
`<speaker>_<emotion>_<intensity>_<number>.wav` (for example `S04_surprise_peak_10.wav`).
The emotion word is the label, except that `anger` is written `angry` as in the other
recipes.

The recordings are 0.40 s or longer, so `remove_long_short` drops none for being
short.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `VIVAE` — the directory holding the extracted audio, or where to download it to
- `VIVAE_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `VIVAE.zip` (93 MB) from Zenodo, checks it against the
MD5 Zenodo lists, and extracts only the `full_set` WAV files. The corpus is
CC BY-NC 4.0 (non-commercial use only). Please cite Holz, N., Larrouy-Maestri, P.,
& Poeppel, D. (2022), *Emotion* 22(1), 213–225.

## Splits

The corpus publishes no partition. Recordings are split 70/10/20 at random within
each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all 11
speakers, and the same speaker's recordings of the same emotion at different
intensities appear across splits.

| split | recordings | after `remove_long_short` |
|---|---|---|
| train | 761 | 761 |
| valid | 108 | 108 |
| test | 216 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of the 93 MB `VIVAE.zip` from Zenodo, MD5 check, extraction) | 2 min 24 s (a 30 GB download was running at the same time) |
| `create_dataset remove_long_short prepare_labels` (conversion to 16 kHz, one file at a time) | 1 min 15 s (includes starting `pixi` and Python) |

The extracted audio takes 85 MB and the converted data 33 MB. `prepare_source`
deletes the zip after a successful extraction.

## Core set

The archive has two folders: `full_set` (1,085 files, used here) and `core_set` (480 files,
all of them also in `full_set`). The core set is a fully crossed subsample (6 emotions x 4
intensities x 10 speakers x 2 items) that the authors selected "based on judged
authenticity" (Zenodo description); the paper that introduces the corpus (Holz et al.,
2022) gives the validation, and a companion paper (Holz et al., 2021, Sci Rep 11:9663) says
the core set was developed "following a technical validation" and "based on authenticity
ratings". So membership in the core set was decided by listeners' perceived authenticity of
each recording, not by the intended emotion. The number of raters and the cut-off of that
selection are not stated in the sources read here (the 2022 paper was not available).

The core set contains speakers `S01` to `S10` with 48 files each; `S11` (54 files) is not in
it, and the other speakers have 87 to 111 files in `full_set`. The recipe does not mark which
files belong to the core set. They are the file names in the archive's `core_set` folder, so
they can be picked out later (for example to train or score on the 480 recordings the authors
consider authentic).

## Listening test

**The recordings were evaluated by listeners.** The 480 core-set recordings were rated by 90
participants recruited at the Max Planck Institute for Empirical Aesthetics (all native German
speakers with normal hearing), 30 per task (Holz et al., 2021, Sci Rep 11:9663):

- Experiment 1: a seven-alternative choice among the six emotions and "none of the specified
  emotions", plus a rating of the speaker's emotional intensity and of the authenticity of the
  expression. Every emotion was classified above chance (16.67%) by the 30 listeners. The most
  frequent confusion was between `achievement` and `surprise`. Accuracy depends on the
  intensity: for four of the six emotions (anger, pleasure, pain, surprise) it is lower at
  peak intensity than at low, moderate and strong. The overall accuracy is described as
  relatively low compared with earlier studies; per-emotion values are given in the
  paper's figures and supplementary table.
- Experiment 2: ratings of how clearly each of the six emotions is perceived (one scale per
  emotion), and of the intensity.
- Experiment 3: ratings of valence and arousal, and of authenticity.

The full set of 1,085 used here also contains the 605 recordings that were not selected. The
individual responses are not in the Zenodo archive; the paper says its data are available on
the Open Science Framework (https://osf.io/jmh5t/). The recipe uses the intended emotion
of the file name.

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
