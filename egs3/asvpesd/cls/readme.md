# ASVP-ESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ASVP-ESD alone and scores it on ASVP-ESD's own test split, classifying each
utterance into one of seven emotions.

[ASVP-ESD](https://zenodo.org/records/4782712) (Audio, Speech and Vision
Processing Lab Emotional Sound Database; Tientcheu Touko Landry Dejoli, Qianhua
He and Wei Xie) is a collection of
emotional sounds taken from websites, YouTube and films, labelled with 12
emotions. This recipe uses the speech recordings only (3,555 of the 9,092
files; the rest are non-speech vocalizations such as laughs and cries) and the
seven emotions shared with the other corpora, `neutral`, `happy`, `sad`,
`angry`, `fear`, `disgust` and `surprise`. The 314 speech recordings labelled
boredom, excited, pleasure, pain or disappointment are left out, which leaves
3,241 recordings, 5.48 hours, from 76 `actor_N` folders (a folder is a source folder, not a verified single speaker). Each is 0.4 to 30.6 seconds
long. The speech is in several languages, mostly Chinese and English. The
classes are very uneven, from 997 `neutral` down to 76 `disgust`.

The label comes from the file name,
`<modality>-<channel>-<emotion>-<intensity>-<statement>-<actor>-<age>-<source>-<language>[-...].wav`
(e.g. `03-01-06-01-02-12-02-01-01-16-04.wav`); the emotion codes are 02 neutral,
03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust and 08 surprised, and the
vocal channel is 01 for speech, as in the Zenodo description. The number of
fields varies between files, so only the first four are read. The `speaker` field is
the `actor_N` folder number. The recordings are already 16 kHz mono;
`create_dataset` still passes them through ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ASVPESD` — the directory holding the extracted audio, or where to download it to
- `ASVPESD_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 4782712 (ASVP-ESD update, `ASVP_UPDATE/Audio` and `ASVP_UPDATE/Bonus`; downloaded around 2026-10-15). The record describes 9,092 files in all; only the speech files (channel code 01) were obtained and counted here. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 3,555 (speech files) | unknown (76 source folders) | 6.08 |

How these were counted: every `.wav` under `ASVP_UPDATE/` (all speech channel 01, 16 kHz mono, 0.38 to 30.56 s; 3,530 in `Audio/` and 25 in `Bonus/`); hours from the file lengths. The non-speech vocalizations and the infant sounds described on the record page were not counted. The folders `actor_N` (76) are source folders, not verified single speakers, so the number of speakers is unknown; the record says odd folders are female and even folders male.

### Files per label

| emotion code in the file name (third field) | samples | hours |
|---|---|---|
| 02 | 997 | 1.885 |
| 05 | 935 | 1.669 |
| 03 | 459 | 0.715 |
| 04 | 432 | 0.828 |
| 08 | 244 | 0.192 |
| 12 | 194 | 0.428 |
| 06 | 98 | 0.119 |
| 07 | 76 | 0.073 |
| 09 | 62 | 0.086 |
| 11 | 26 | 0.045 |
| 10 | 17 | 0.019 |
| 01 | 15 | 0.018 |

How these were counted: the third field of each file name `<modality>-<channel>-<emotion>-<intensity>-<statement>-<actor>-<age>-<source>-<language>[-...].wav`. The record gives 02 neutral, 03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust and 08 surprised; the remaining codes (01, 09, 10, 11, 12) are the five emotions boredom, excited, pleasure, pain and disappointment, and their order was not checked. The number of fields varies between files (8 to 12). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The infant-sound folder `Bonus/` (25 files here, all labelled with the same emotion code) is in the speech set; 16 of them carry the age field of a newborn. The Whisper transcripts of these files are mostly repetitions and meaningless words, which suggests infant vocalizations (inferred, not listened to).
- Pairs with the same decoded audio or nearly: 22 groups (44 files) among the files with the seven shared emotions; in 8 groups (16 files) the emotion labels differ (for example 6 groups in the `a35` folder).
- The language field has values the record does not describe (3,241 files with the seven shared emotions: Chinese 1,585, English 989, code 03 in 467, French 87 and others); the intensity, age and source fields also contain undescribed values.
- The folders include dialogues and multi-person audio although the record describes them as actors; the individual raters' judgements (five raters for the first 52 folders and three others later) are not in the release.


## Download

`prepare_source` downloads `ASVP_ESD.zip` (1.26 GB) from Zenodo, checks it
against the MD5 Zenodo lists, and extracts only the speech WAV files. The
corpus is CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). The `actor_N` folders are not held out, so the splits share all
76 folders.

| split | utterances |
|---|---|
| train | 2248 |
| valid | 322 |
| test | 641 |

## Listening test

The labels are the majority vote of raters who listened and chose the emotion they perceived: five raters for the first 52 folders and three other raters for the folders added later. The individual ratings are not distributed.

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

**Duplicates.** 22 pairs of recordings have the same sound (identical after decoding, or a waveform correlation of 0.998 or more). In 8 pairs (16 recordings) the two files carry different emotions; none of them is used (`conflicting_ids`). In the other 14 pairs the emotion is the same; with `exclude_duplicates: true` (the default) one of each pair is dropped (`duplicate_ids`), and `false` keeps both. This leaves 3,211 recordings (3,225 with `false`).

**Infant voices.** The 25 speech recordings under `Bonus/` (folders `actor_50` and `actor_150`) are described as infant voices; the Whisper transcripts are repeated or meaningless words, so they look like babbling by young children (not listened to). They are used by default; set `exclude_bonus: true` to leave them out.

**Source.** Part of the corpus is taken from films, videos and websites, so the rights of the original recordings may differ from the CC BY 4.0 of the release.
