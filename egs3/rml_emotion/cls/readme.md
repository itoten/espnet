# RML Emotion Database speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the RML Emotion Database alone and scores it on its own test split,
classifying each utterance into one of six emotions.

**License: Unknown.** The Kaggle page, published under the account Ryerson
Multimedia Lab, lists the license as "Unknown" and states no terms of use.
An archived copy (2015) of the lab's original page
([Wayback Machine](https://web.archive.org/web/20150514235638/http://www.rml.ryerson.ca/rml-emotion-database.html))
says the data is for research and non-commercial use only; it is a reference, not
an authoritative source. Check with the lab before using the data beyond
internal research.

[RML Emotion Database](https://www.kaggle.com/datasets/ryersonmultimedialab/ryerson-emotion-database)
(Ryerson Multimedia Lab, Kaggle) is a set of 720 acted audio-visual clips, 120
in each of six emotions: `angry`, `disgust`, `fear`, `happy`, `sad` and
`surprise` (there is no neutral class). The Kaggle description says that "about
500 samples" were selected by a listening test; the download has no marker for
which clips these are, so this recipe uses every clip except six exact duplicates
(714 in all). The Kaggle release is avi
video with a stereo audio track (44.1 kHz in 669 clips, 48 kHz in 51), 2.13 to
8.37 s long, 0.99 hours in total; the Kaggle description says 22,050 Hz and a
single channel. `create_dataset` extracts the audio as 16 kHz mono wav with ffmpeg.

Files are laid out as `s<N>/s<N>/[f<M><lang>/]<emo><take>.avi`. The emotion is
the two-letter code in the file name (`an`, `di`, `fe`, `ha`, `sa`, `su`). `s1`
to `s8` are treated as 8 speakers, which the Kaggle page does not confirm. Subjects
`s3` to `s8` have language folders (`eng`, `chi`, `per`, `urd`, `pjb`, `sdh`, `hnh`); `s1`
and `s2` have none. The meaning of the language codes is taken from their
spelling, not from any documentation. Eleven clips form five groups of byte-identical files
(same subject, folder and emotion). Only the lowest take number of each group is
kept; the other six are listed in `exclude_files` in `dataset/config.yaml`, and
`create_dataset` stops if identical files are left.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `RML_EMOTION` — the directory holding the extracted `s1/` ... `s8/` folders
- `RML_EMOTION_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `ryersonmultimedialab/ryerson-emotion-database` (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 720 | 8 (see below) | 0.99 |

How these were counted: every `.avi` file under `s1/` to `s8/` (video with a stereo audio track, 44.1 kHz in 669 files and 48 kHz in 51; 2.13 to 8.37 s); hours from the audio lengths. The release does not say that a folder `s1` to `s8` is one person; 8 is the number of folders (the Kaggle page and the original page also speak of 8 subjects).

### Files per label

| emotion code in the file name | samples | hours |
|---|---|---|
| an | 120 | 0.157 |
| di | 120 | 0.177 |
| fe | 120 | 0.162 |
| ha | 120 | 0.161 |
| sa | 120 | 0.175 |
| su | 120 | 0.156 |

How these were counted: the two-letter emotion code at the start of each file name. The release has no neutral class and no label table; the codes are read as angry, disgust, fear, happy, sad and surprise. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The Kaggle description says the audio is 22,050 Hz and single-channel; the files are 44.1 kHz or 48 kHz stereo. The left and right channels differ in all 720 files (they contain nearly the same sound). The original page says 3 to 6 s per clip; the files are 2.13 to 8.37 s.
- The Kaggle description says "about 500 samples" were selected by a listening test; the 720 files carry no marker of which ones.
- 11 files form 5 groups of byte-identical content (same subject, folder and emotion, differing only in the take number).
- Observation by the recipe author (2026-10-20 at the latest): the 5 duplicate groups were listened to; each group is the same utterance.
- The language folders of `s3` to `s8` use 7 codes (`eng`, `chi`, `per`, `urd`, `pjb`, `sdh`, `hnh`) while the description names 6 languages; what `sdh` and `hnh` mean is not documented. `s1` and `s2` have no language folder.


## Download

The recipe does not download the data. Download the zip from the Kaggle page
above (3.4 GB, 4.5 GB unpacked, free Kaggle account) and unzip it into `$RML_EMOTION`.
If the folders are missing, `create_dataset` stops with an error that gives the
URL. The download time was not measured; `create_dataset`, `remove_long_short`
and `prepare_labels` together took about 61 seconds.

## Splits

The dataset publishes no partition. Whole subject folders are assigned at random
(`split_seed` 0) to test, then valid, then train, so no subject spans two
splits. Because the folders differ in size (25 to 182 clips), the ratios are only
approximate and the labels are not balanced across splits.

| split | utterances | subjects |
|---|---|---|
| train | 415 | s1, s4, s7, s8 |
| valid | 80 | s3 |
| test | 219 | s2, s5, s6 |

No clip is shorter than the 0.2 s floor of `remove_long_short`, and none is longer than 30 s.

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
