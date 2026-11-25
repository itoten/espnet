# SantaliESC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SantaliESC alone and scores it on SantaliESC's own test split,
classifying each utterance into one of five emotions.

[SantaliESC](https://zenodo.org/records/20702265) (Santali Emotional Speech
Corpus) is acted Santali speech recorded in Koraput, Odisha, India. The Zenodo
record states 2,000 recordings from 80 speakers aged 18 to 60: five sentences
(Ol Chiki script) spoken in five emotions, `angry`, `happy`, `neutral`, `sad`,
`surprise`, 400 recordings per emotion, about 1 h 53 min in all, cut into
3-5 s WAV files at 48 kHz.

The record does not describe how the files inside the archive are named. The
archive unpacks to `<Emotion>/Statement <n>/03-01-01-<EE>-<SS>-<NN>.wav` (see
"The data"); the label is read from the emotion folder and the speaker from
`NN`, with the rules `emotion_aliases` and `speaker_re` in
`dataset/config.yaml`. If a file cannot be labelled, `create_dataset` stops
and prints a sample of the paths it found. Every file found under the source
root is listed in `source_files.txt`, and the members of a downloaded archive
in `archive_members.txt`, both under the output directory. `create_dataset`
converts all recordings to 16 kHz mono 16-bit PCM with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SANTALIESC` — an existing unpacked copy of the archive, or where to download it to
- `SANTALIESC_OUTPUT` — where the resampled audio, manifests and file lists are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-18 against the Zenodo record 20702265, the archive `SantaliESC (Santali Emotional Speech Corpus)-20260615T095634Z-3-001.zip` (892 MB, md5 `774b4335cfc4da2f2ed29e09cfb26847`, downloaded 2026-11-18). All counts cover every recording in the release, before any filtering by this recipe.

### Access

Zenodo, direct download, no sign-in.

### Size of the release

| recordings | speakers | total hours |
|---|---|---|
| 2,000 | 80 | 2.27 |

How these were counted: every `.wav` in the archive (no other files); hours from the WAV headers (2.32 to 7.98 s). All are stereo: 1,975 at 48 kHz and 25 at 44.1 kHz. The archive unpacks to `SantaliESC (Santali Emotional Speech Corpus)/<Emotion>/Statement <n>/03-01-01-<EE>-<SS>-<NN>.wav`, 80 files in each of the 25 emotion/statement folders. Speakers are the values of `NN` (01 to 80), which the record does not explain; they are read as speakers because each value occurs once in every folder, matching the record's 80 speakers x 5 statements x 5 emotions.

### Files per label

| emotion folder (code `EE` in the file name) | recordings | hours |
|---|---|---|
| Angry (01) | 400 | 0.424 |
| Happy (02) | 400 | 0.445 |
| Neutral (03) | 400 | 0.376 |
| Sad (04) | 400 | 0.552 |
| Surprise (05) | 400 | 0.477 |

How these were counted: the emotion folder of each file; the code `EE` in the file name agrees with the folder for every file. Hours per label are measured in the same way as the total hours above and summed over the recordings of that label.

### Issues in the release

- Twelve pairs of recordings have byte-identical content (listed in `dataset/config.yaml`):
  - Four pairs have different labels: `Neutral/Statement 4` NN 09 to 12 are identical to `Sad/Statement 4` NN 77 to 80 (`conflicting_ids`).
  - Eight pairs have the same label, each within one folder, with two different NN (`duplicate_ids`; the second of each pair is listed): Happy/5 NN 21 and 22; Sad/1 39 and 40; Sad/4 03 and 75; Sad/5 46 and 47; Surprise/3 54 and 55; Surprise/4 47 and 48, 59 and 60; Surprise/5 63 and 67.
  - Since the copies carry different NN, either the copy or the speaker number is wrong for these files.
- The record gives the total length as about 1 hour 53 minutes; the WAV headers add up to 2.27 hours.


## Download

`prepare_source` downloads
`SantaliESC (Santali Emotional Speech Corpus)-20260615T095634Z-3-001.zip`
(892 MB) from [Zenodo](https://zenodo.org/records/20702265), checks it against
the MD5 the record lists, unpacks it and deletes the archive. The corpus is
CC BY 4.0.

## Splits

The corpus publishes no partition. The speakers (`NN` in the file name, see
"The data") are shuffled with `split_seed` and split 80/10/10, so no speaker
is in two splits:

| split | speakers | utterances |
|---|---|---|
| train | 64 | 1,600 |
| valid | 8 | 200 |
| test | 8 | 200 |

Each split has the same number of utterances of every emotion. If no speaker
can be read, the utterances are split 70/10/20 at random within each label
instead, and the splits share speakers; `split_mode` in `dataset/config.yaml`
forces either one. The byte-identical copies listed in `dataset/config.yaml`
carry different `NN`, so with the default settings a copy can be in a
different split from its original.

## Listening test

The Zenodo record states that the emotion labels were validated by ten human
evaluators, with an average recognition accuracy of about 74.6% for the
intended emotions. Per-recording ratings are not part of the download; the
recipe uses the intended emotion.

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
