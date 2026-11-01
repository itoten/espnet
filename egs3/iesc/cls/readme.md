# IESC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on IESC alone and scores it on IESC's own test split, classifying each
utterance into one of five emotions.

IESC (Indian Emotional Speech Corpora) is an acted English corpus of eight
north Indian speakers, recorded in a closed room with a speech-recorder app on a
mobile phone, with a headset microphone. The paper (Singh and Goel, 2023, Table 2)
lists the speakers: S1, S2, S3, S6 and S7 are men and S4, S5 and S8 women, aged 23
to 35. The paper does not say whether they are professional actors. Each speaker reads
two sentences taken from RAVDESS ("Kids are talking by the door" / "Dogs are
sitting by the door") several times in each emotion. The paper reports a listening
check on 60 of the 600 files (10%) by more than 20 people using Google Forms, with an
average accuracy of 89.96%; the label is the emotion each file was recorded in (its
folder), not the listeners' answer. The folder layout is
`Speaker-<n>/<Emotion>/<E>-<take>-<n>-<sentence>.wav`, and the folder names give
the label.

Measured on the copy used here: 600 wav files, 0.56 hours in total, 1.77 to
5.50 seconds long; 598 are used (two duplicates are left out, see below).
Emotions: angry 119, fear 120, happy 120, neutral 120, sad 119 (`Anger` is
written `angry` here, as in the other recipes). Speakers 1 to 4 have
50 files each and speakers 5 to 8 have 100. The recordings are in mixed formats
as distributed (16, 32, 44.1 or 48 kHz; mono or stereo; all 16-bit) and are
converted to 16 kHz mono.

Things to know when reading the scores:

- Only two sentences exist, so the text carries no emotion information but the
  model sees the same two sentences in every split.
- Two pairs of files are byte-identical copies (`Speaker-3/Sad/S-1-3-1.wav` =
  `S-3-3-1.wav`, `Speaker-8/Anger/A-10-8-2.wav` = `A-7-8-2.wav`). The one with
  the higher take number is left out of each pair (`exclude_files` in
  `dataset/config.yaml`). One file name looks mistyped
  (`Speaker-8/Sad/S-8-8-18.wav`, where the other takes end in `-1` or `-2`); a Whisper transcript reads sentence
  1, so it is probably `S-8-8-1.wav`. It is used as is.
- The recording format differs by speaker, and whole speakers are held out, so
  the splits also differ in recording device.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset IESC (Indian Emotional Speech Corpora; `Speaker-<n>/<Emotion>/`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 600 | 8 | 0.56 |

How these were counted: every `.wav` under `Speaker-<n>/<Emotion>/` (all 16-bit; 1.77 to 5.50 s); hours from the file lengths. Speakers are the 8 speaker folders (speakers 1 to 4 have 50 files each and 5 to 8 have 100 files). Formats as distributed: 44.1 kHz mono 200 files, 48 kHz stereo 200, 32 kHz mono 50, 16 kHz mono 50, 44.1 kHz stereo 50, 48 kHz mono 50.

### Files per label

| label (folder name in the release) | samples |
|---|---|
| Anger | 120 |
| Fear | 120 |
| Happy | 120 |
| Neutral | 120 |
| Sad | 120 |

How these were counted: the emotion folder each file is in; the file name `<E>-<take>-<n>-<sentence>.wav` carries the emotion letter, the take, the speaker and the sentence (1 or 2).

### Issues in the release

- Two pairs of files are byte-identical copies of the same speaker, emotion and sentence, differing only in the take number: `Speaker-3/Sad/S-1-3-1.wav` = `S-3-3-1.wav` and `Speaker-8/Anger/A-10-8-2.wav` = `A-7-8-2.wav`.
- Observation by the recipe author (2026-10-25): both identical pairs were listened to and are the same sound.
- One file name looks mistyped: `Speaker-8/Sad/S-8-8-18.wav`; the other takes end in `-1` or `-2`, and a Whisper transcript reads sentence 1, so it is probably `S-8-8-1.wav`.
- The Kaggle license field says Unknown; the release asks for a citation.

### Not stated in the documentation

- Only two sentences ('Kids are talking by the door' and 'Dogs are sitting by the door') occur in all 600 files, in every emotion and every speaker (checked from the file names and a Whisper transcription).


## Provenance

Kaggle dataset `ybsingh/indian-emotional-speech-corpora-iesc` (version 1,
2022-01-28). The licence field on Kaggle is Unknown, and the description gives
only a request: "For using This Dataset Kindly Cite this Paper". This recipe
treats that request as the terms of use, trusting the authors, and so counts the
corpus as open for use with citation. The description asks users to cite
Singh, Y.B. and Goel, S., "A lightweight 2D CNN based approach for
speaker-independent emotion recognition from speech with new Indian Emotional Speech
Corpora", Multimedia Tools and Applications 82, 23055-23073 (2023),
[doi:10.1007/s11042-023-14577-w](https://doi.org/10.1007/s11042-023-14577-w). The
authors are at Bennett University and KIET Group of Institutions (India).

Kaggle requires a login, so the recipe does not download. Download the zip
(138 MB) and extract it:

```bash
unzip indian-emotional-speech-corpora-iesc.zip -d "$IESC"
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `IESC` — the directory holding (or containing, at any depth) the `Speaker-<n>/` folders
- `IESC_OUTPUT` — where the converted audio and manifests are written

## Download

The zip took 6 seconds to download (unzip not timed separately).
`create_dataset`, `remove_long_short` and `prepare_labels` together took about
60 seconds on CPU.

## Splits

No partition is published. Whole speakers are held out: the eight speakers are
shuffled with `split_seed` (`dataset/config.yaml`) and two go to test, one to
valid, five to train. This gives the following splits.

| split | speakers | utterances | after `remove_long_short` |
|---|---|---|---|
| train | 1, 3, 4, 7, 8 | 348 | 348 |
| valid | 6 | 100 | 100 |
| test | 2, 5 | 150 | 150 (not filtered) |

Each speaker has the same number of files per emotion, so the classes are
balanced in every split (train has 69 angry and 69 sad after the two exclusions).

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
