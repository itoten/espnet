# AkaCE (Akan Cinematic Emotions) speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on AkaCE alone and scores it on AkaCE's own test split, classifying each
utterance into one of nine emotion labels.

AkaCE ([GitHub](https://github.com/sasudavid/AkaCE), Sasu et al., Findings of ACL 2025)
is Akan (a West African language) dialogue cut from 21 Akan movies. The paper says the
movies were downloaded from the Internet Archive. Utterances were transcribed and timed by
hand and given an emotion label by watching the movie. The recipe uses the audio only.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the AkaCE release from the GitHub repository
`sasudavid/AkaCE` (commit `1a556f591d668691cfe3e3b3113421648c49ddfd`, 2025-05-24) and the
public Google Drive folder it links to (fetched 2026-10-28/29). All counts cover every sample
in the release, before any filtering by this recipe.

### Access

Direct download from the public Google Drive folder, no sign-in. The folder holds several
versions of the table and of the audio; the recipe uses
`excel_files/Akan Speech Emotion Dataset cleaned.csv` and
`audio/movie_audio_segments_wav/`. A bulk download with `gdown` stopped part-way because of
Google's limits; the files were fetched one by one by their IDs.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 6,138 | unknown | 15.36 |

How these were counted: every `.wav` under `audio/movie_audio_segments_wav/` (21 movie folders;
all 16 kHz mono); hours from the file lengths (1.51 s to 498.51 s). The table
`Akan Speech Emotion Dataset cleaned.csv` has 6,162 rows, one per utterance; 3 of its rows
have no segment file (see below).
Speakers: the `Speaker` column of the table is only unique within a dialogue (for example
`Speaker one man`), so speakers cannot be counted across dialogues; the number is unknown
from the release.

### Files per label

| label (as in the release) | table rows | hours |
|---|---|---|
| Neutral | 2,984 | 6.597 |
| Anger | 1,109 | 4.009 |
| Sadness | 811 | 1.956 |
| Joy | 569 | 1.099 |
| Surprise | 390 | 0.868 |
| Disgust | 162 | 0.593 |
| Fear | 135 | 0.242 |
| Frustration | 1 | 0.002 |
| Confusion | 1 | 0.001 |

How these were counted: the `Emotion` column of all 6,162 rows of the table. Hours per label are measured in the same way as the total hours above and summed over the rows of that label. 1 row(s) of `Neutral` have no readable audio file; 1 row(s) of `Anger` have no readable audio file; 1 row(s) of `Surprise` have no readable audio file and add nothing to the hours.

### Issues in the release

- The times of the movie `me_sofo_anyim_me_yere` (33 rows, all in `test`) are written one
  unit off: `00m:01s:12ms` where the other movies have the form `01m:04s:00ms`, so
  `start_seconds` is about 1 s. The segment files of this movie are all cut from the first
  1 to 3.5 s of the movie. Checked from the table and file names; the true times are not
  known.
- The segments are cut 1.5 s after the annotated end, so a segment can contain the start of
  the next utterance (rule derived from the file names: 6,159 of the 6,162 rows match a
  file). 3 rows have no segment file.
- Annotated times are whole seconds: in the other movies 582 rows have an end not after the
  start, and the longest segment is 498.51 s.
- Rows pointing to the same segment: 11 groups (24 rows) have different emotion labels;
  6 further rows have the same label as an earlier row.
- 72 distinct transcriptions occur with more than one emotion label (259 rows).
- The `split` column has 4,498 / 935 / 729 rows for train / val / test; the paper reports
  3,888 / 816 / 834 and the GitHub table has a separate `split1` column with other counts.
- The table has nine emotion labels; the paper names seven (`Frustration` and `Confusion`
  occur once each).
- Observation by the recipe author (2026-10-29): two 1.5 s segments cut from rows of
  length zero (`Joy` and `Surprise`) were listened to; something like an utterance can be
  heard, but it is hard to tell.
- Observation by the recipe author (2026-10-29): two 75 s candidate regions of
  `me_sofo_anyim_me_yere` were cut to find the true times; the author could not
  understand Akan and could not decide, so the true times remain unknown.

### Listening test ratings in the release

None: the individual annotators' votes are not in the release.

## Labels

The authors' Google Drive folder holds several tables. The recipe reads
`Akan Speech Emotion Dataset cleaned.csv` (6,162 rows, one per utterance, with a
`split` column). The label is its `Emotion` column, mapped to lower-case names (`Joy` is
`happy`; the paper calls it Happy).

The paper (Sections 3.2, 3.4, 3.5): seven annotators (five men, two women, native Akan
speakers) labelled the movies while watching them, each movie assigned at random to one
annotator; two of them then gave a second label to the utterances they had not labelled, and
the final label is the majority vote, with an external Akan-speaking expert deciding
disagreements. Agreement over the whole corpus is Fleiss' kappa 0.488. The individual
annotators' votes are not in the download. Because the labels were given while watching the
video, they are not judgements of the voice alone.

The table has nine labels, not the seven of the paper: `Frustration` and `Confusion` occur
once each; they are kept.

## Audio

The recipe uses the authors' segment files, `audio/movie_audio_segments_wav/<movie>/
<movie>_<start ms>_<end ms>.wav` (16 kHz mono). A table row with annotated times S and E
seconds corresponds to the file `<movie>_<S*1000-10>_<E*1000+1500>`: the segments end 1.5
seconds after the annotated end, so a segment can contain the start of the next
utterance. This rule was derived from the file names (6,159 of the 6,162 rows match a
file).

Things measured in the table and segments (not discussed in the paper):

- **One movie is left out.** In `me_sofo_anyim_me_yere` (33 rows, all test) the times
  are written one unit off (`00m:01s:12ms`, where every other movie has the form
  `01m:04s:00ms`), so `start_seconds` becomes 1.012 and the authors' segments are all
  cut from the first 1 to 3.5 s of the movie, not from the annotated utterances. The
  movie is listed in `excluded_movies` in `dataset/config.yaml`. The recipe therefore uses
  20 of the 21 movies.
- In the other movies the annotated times are whole seconds. 582 of their rows have an end
  not after the start, and a few are far too long; the 6,096 converted segments are 1.51
  to 498.5 s long, 332 are longer than 30 s.
- 3 rows have no segment file in the release and are not used.
- 24 rows (11 groups) point to the same segment with different labels; they are not used
  (`conflicting_ids` in `dataset/config.yaml`). 6 more rows point to the same segment with
  the same label as an earlier row; they are used unless `exclude_duplicates: true`.
- The paper reports 3,888 / 816 / 834 utterances for train / validation / test after
  cleaning; the `split` column of the table has 4,498 / 935 / 729.
- Of the 5,896 distinct transcriptions in the table, 72 occur with more than one emotion
  label (259 rows).

## What the recipe uses

After leaving out the excluded movie, the 3 rows without a segment and the 24 rows with
conflicting labels, 6,102 utterances (15.3 hours) are used. Labels:

| label | utterances |
|---|---|
| neutral | 2,954 |
| angry | 1,099 |
| sad | 802 |
| happy | 564 |
| surprise | 386 |
| disgust | 161 |
| fear | 134 |
| frustration | 1 |
| confusion | 1 |

The speaker column is only unique within a dialogue (for example `Speaker one man`), so
speakers are not identified across dialogues.

## Environment variables

Both are optional.

- `AKACE` - a directory holding the `excel_files/` and `audio/movie_audio_segments_wav/`
  folders of the authors' Google Drive folder (default `download/` under the recipe)
- `AKACE_OUTPUT` - where the converted audio and manifests are written (default `data/`)

## Download

The files are in a public Google Drive folder,
<https://drive.google.com/drive/folders/1T2pVy_A6kD5owaAmFyPqgCtFTmjtB0b4> (linked from the
GitHub README); no sign-in is needed. The recipe does not download them. Fetch
`excel_files/Akan Speech Emotion Dataset cleaned.csv` and the segment files the table points
to (6,107 distinct files outside the excluded movie, 1.7 GB on disk) into `$AKACE`. Measured: about 36 minutes for the
segments with four parallel downloads; `create_dataset`, `remove_long_short` and
`prepare_labels` together took about 4 minutes.

## Splits

The `split` column of the table is used as published (`val` is the validation split).

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 4,481 | 4,393 |
| valid | 929 | 881 |
| test | 692 | 496 |

`remove_long_short` drops the segments longer than 30 s from all three splits: 88 train, 48
valid and 196 test (up to 498.5 s). Test is filtered too: those
segments come from wrong annotated times, and the longest ones make inference run out of GPU
memory. Test scores therefore cover 496 of the 692 test utterances.

## License and source material

The repository README says: "This dataset is released for non-commercial research purposes
only." Its `LICENSE.txt` (AkaCE Dataset License Agreement) allows non-commercial academic
research, teaching and educational use, prohibits sharing or redistributing the dataset or
any processed form of it without the authors' written permission, and requires citing the
paper (Sasu, Wu, Gong, Chen, Shi, Ai, Hirschberg and Schluter, "Akan Cinematic Emotions: A
Multimodal Dataset for Speech Emotion Recognition in Movie Dialogues", Findings of ACL 2025).
GitHub shows no license for the repository. The movies are third-party works; the paper says only that they were downloaded from
the Internet Archive and does not discuss the rights in the films, and neither does the
README.

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
