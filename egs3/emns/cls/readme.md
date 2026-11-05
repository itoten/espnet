# emns speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on emns alone, and scores it on emns's own test split (8 classes:
`angry`/`disgust`/`excited`/`happy`/`neutral`/`sad`/`sarcastic`/`surprise`).

[EMNS](https://www.openslr.org/136/) is a single-speaker narrative-storytelling
corpus published as 48 kHz WebM; `create_dataset` transcodes it to 16 kHz mono
WAV with ffmpeg. The label comes from the pipe-separated `metadata.csv`.
`EMNS` points at an existing corpus tree (`cleaned_webm/` and `metadata.csv`),
or where to download it to.

The corpus has one speaker, so no split can hold speakers out; the partition
is stratified by emotion instead (20% valid, 20% test), which
`dataset/config.yaml`'s ratios and seed control.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the OpenSLR resource 136 (EMNS; `cleaned_webm/` and `metadata.csv`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from OpenSLR, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,181 (audio files) | 1 | 1.91 |

How these were counted: every `.webm` in `cleaned_webm/` (48 kHz mono Opus, 1.22 to 12.08 s); hours from the decoded audio length of each file (the container durations are longer, because the first audio packet of each file starts after an initial offset, for example 7.755 s in `recorded_audio_fCBtcy4.webm`). `metadata.csv` (pipe-separated) has 1,205 rows; the 1,181 rows with `status` = `Complete` have audio and 24 rows (`Needs Updating`) do not. The speaker is one woman in her twenties (the `gender` and `age` columns).

### Files per label

| label (the `emotion` column, rows with audio) | samples | hours |
|---|---|---|
| Disgust | 162 | 0.296 |
| Happy | 157 | 0.240 |
| Surprised | 151 | 0.250 |
| Excited | 150 | 0.221 |
| Neutral | 149 | 0.209 |
| Sad | 147 | 0.283 |
| Sarcastic | 139 | 0.223 |
| Angry | 126 | 0.188 |

How these were counted: the `emotion` column of the 1,181 `Complete` rows. The 24 rows without audio have Angry 7, Excited 5, Disgust 3, Sarcastic 3, Sad 3, Surprised 2 and Happy 1; their dates run from 2022-07-27 to 2022-08-19. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- 24 rows of `metadata.csv` (status `Needs Updating`) point to `wavs/*.webm` files that are not in `cleaned_webm/`; the reason is not stated. The OpenSLR page also offers the unprocessed recordings (`raw_webm.tar.xz`), which were not checked for these rows.
- The corpus has one speaker, so speaker-dependent analysis is not possible.


## What the recipe uses

EMNS is acted British English speech by one female speaker in her twenties,
reading narrative sentences "for games, television and graphic novels" in eight
styles. `metadata.csv` also holds an intensity `level` (0 to 10), the
sentence text, a `description` (the reading instruction, for example emphasis on given
words), a `status` and the path of the recording. The recipe uses only the label.
The release used here has 1,181 recordings (1.91 hours, 1.22 to 12.08 seconds, 48 kHz
mono Opus). The classes are disgust 162, happy 157, surprised 151, excited 150,
neutral 149, sad 147, sarcastic 139 and angry 126.

`metadata.csv` has 1,205 rows, but 24 are marked `Needs Updating` and their recordings
(`wavs/*.webm`) are not in `cleaned_webm/`; only the 1,181 `Complete` rows have audio. The
24 rows cannot be used and are not read. Why they have no audio is not stated (the
OpenSLR page also offers the unprocessed recordings, `raw_webm.tar.xz`, which was not
checked).

## License and citation

OpenSLR lists the corpus under Apache 2.0. Cite the corpus as given on
[OpenSLR 136](https://www.openslr.org/136/): "EMNS /Imz/ Corpus: An emotive
single-speaker dataset for narrative storytelling in games, television and graphic
novels".

## Quick start

```bash
# 1) Write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`EMNS_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
