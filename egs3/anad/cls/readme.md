# ANAD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ANAD alone and scores it on ANAD's own test split, classifying each
utterance into one of three emotions.

ANAD (Arabic Natural Audio Dataset) is a natural Arabic corpus built from eight
recordings of live phone calls on Arabic talk shows. The release cuts them into
1-second chunks (`<call>_<turn> (<chunk>).wav`, e.g. `V7_1 (145).wav`) and
`ANAD.csv` gives each chunk one label: `angry`, `happy` or `surprised`
(written `surprise` here, as in the other recipes).

Measured on the copy used here: 1,334 chunks that have both a wav file and a
label, 0.05 to 1.0 seconds long. They are 44.1 kHz stereo as distributed and are
converted to 16 kHz mono. 768 of them are excluded (see below), leaving 566:
angry 404, happy 154, surprise 8. The remaining chunks come almost entirely
from two calls: V7 (angry, 376) and V8 (happy, 149).

Things to know when reading the scores:

- Every label is constant within a call: calls V1, V3, V8 are `happy`, V2 and
  V6 are `surprise`, V4, V5 and V7 are `angry`. A model can therefore learn the
  call (speaker, line, topic) instead of the emotion.
- The release describes the labels as given by 18 listeners per video (call),
  not per chunk. Whether each chunk's audio matches its label has not been
  checked.
- 768 chunks are skipped (`exclude_files` in `dataset/config.yaml`): they fall in
  two groups of near-identical audio (386 and 382 chunks, each about 1 s) that
  appear under different calls and labels. Six of them, listened to, were the
  same audio.
- No speaker id is published. The splits below keep a call's turn together but
  do not hold speakers out.
- The release is not fully consistent: `ANAD.csv` lists 1,383 chunks, but 49 of
  them have no wav file, and 86 wav chunks (plus 8 full-call recordings, not
  used) have no label in the table. Only chunks with both are used.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset ANAD (Arabic Natural Audio Dataset; `ANAD.csv`, `ANAD_Normalized.csv` and the three `1sec_segmented_part*` folders, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,420 | unknown | 0.36 |

How these were counted: every `.wav` under the three `1sec_segmented_part*` folders (44.1 kHz stereo, 0.05 to 1.0 s; one file could not be read); hours from the file lengths. The release has no speaker ID: the clips come from eight recordings (`V1` to `V8`) of live calls on Arabic talk shows, so the number of speakers is unknown.

### Files per label

| label (the `Emotion` column of `ANAD.csv`) | rows of `ANAD.csv` | hours |
|---|---|---|
| angry | 741 | 0.191 |
| happy | 505 | 0.114 |
| surprised | 137 | 0.036 |

How these were counted: the `Emotion` column of `ANAD.csv` (1,383 rows). The label is constant within a call: calls `V1`, `V3`, `V8` are happy, `V2` and `V6` surprised, `V4`, `V5` and `V7` angry. Of the 1,420 wav files, 86 have no row in the table and 49 rows have no wav file; 1,334 have both. Hours per label are measured in the same way as the total hours above and summed over the rows of that label. 18 row(s) of `angry` have no readable audio file; 31 row(s) of `happy` have no readable audio file and add nothing to the hours.

### Issues in the release

- The labels were given by 18 listeners per video (call), not per clip; whether each clip matches its label has not been checked.
- `ANAD.csv` (1,383 rows) and the wav files (1,420) do not match: 49 rows have no wav, 86 wavs have no row, and 1 wav is unreadable. 8 full-call recordings are also in the release.
- Of the 1,334 clips with both a wav and a label, 768 fall into two groups of near-identical audio (386 and 382 clips; waveform correlation above 0.9) that appear under different calls and different labels; Whisper's transcripts of them are meaningless strings.
- Observation by the recipe author (2026-10-23 at the latest): six clips from these groups were listened to and are the same sound.
- Observation by the recipe author (2026-10-23 at the latest): 15 of the remaining 566 clips were listened to; in each the same female voice (sounding like "ava smi...") can be heard, with slightly different background sound. The cause is not known.
- The release has no speaker ID: the clips of one call cannot be assigned to speakers.


## Provenance

Kaggle dataset `suso172/arabic-natural-audio-dataset` (version 11, 2017-12-01),
licence shown on Kaggle: CC BY-NC-SA 4.0 (non-commercial, share-alike). The
paper that introduces it is Klaylat et al., "Emotion recognition in Arabic
speech", Analog Integrated Circuits and Signal Processing 96, 337-351 (2018).
Whether a citation is required was not checked; see the Kaggle page.

Kaggle requires a login, so the recipe does not download. Download the zip and
extract only the audio and the table:

```bash
unzip arabic-natural-audio-dataset.zip '*.wav' '*.csv' -x 'V?.wav'
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ANAD` — the directory holding `1sec_segmented_part{1,2,3}/` and `ANAD.csv`
- `ANAD_OUTPUT` — where the converted audio and manifests are written

## Download

The zip (587 MB) took 24 seconds to download. `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 90 seconds.

## Splits

No partition is published and the labels cannot be tied to speakers. Chunks of
the same call and turn (`V7_1`) are kept in one split, and within each label the
groups are assigned largest first to the split furthest below its target
(70/10/20, `split_seed` in `dataset/config.yaml`).

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 396 | 349 |
| valid | 56 | 33 |
| test | 114 | 114 (not filtered) |

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
