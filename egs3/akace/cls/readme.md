# AkaCE (Akan Cinematic Emotions) speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on AkaCE alone and scores it on AkaCE's own test split, classifying each
utterance into one of nine emotion labels.

AkaCE ([GitHub](https://github.com/sasudavid/AkaCE), Sasu et al., Findings of ACL 2025)
is Akan (a West African language) dialogue cut from 21 Akan movies. The paper says the
movies were downloaded from the Internet Archive. Utterances were transcribed and timed by
hand and given an emotion label by watching the movie. The recipe uses the audio only.

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

## Data

6,102 utterances (15.3 hours) are used. Labels:

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
