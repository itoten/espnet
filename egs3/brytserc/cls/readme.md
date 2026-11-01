# BRYTSERC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BRYTSERC alone and scores it on its own test split, classifying each
utterance into one of seven emotions.

BRYTSERC (Bangla Realistic YouTube Speech Emotion Recognition Corpus) is a Bangla
corpus of clips collected from publicly available YouTube videos. The archive holds
5,215 wav files named `<Emotion>_<n>_<Gender>.wav`, exactly 745 per emotion
(`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`; the recipe lowercases
them), 4.35 hours in total. Every clip is 3.0 seconds long, 44.1 kHz mono 16-bit,
so `create_dataset` converts them to 16 kHz with ffmpeg. The gender in the file name
(3,446 male and 1,769 female clips) is the only speaker information; no
speaker id, video id or transcript is published.

Things to know when reading the scores:

- The clips are not all distinct. 49 pairs (98 clips) are near-identical: the
  waveforms correlate above 0.98 (mostly 0.9998 or higher) within 10 ms of each
  other, and the bytes differ. They look like the same 3 seconds saved under two
  numbers. 3 pairs carry different labels (`Happy_005_Male` / `Neutral_225_Male`,
  `Fear_646_Male` / `Neutral_024_Male`, `Fear_645_Male` / `Neutral_293_Male`), and
  20 pairs fall into different splits. Only clips of the same length that line up
  are found; windows cut at an offset from the same video and overlapping in part
  were not searched for. All 5,215 clips are used as they are.
- How the labels were given (listeners or the video context) is not stated.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the BRYTSERC archive (Bangla Realistic YouTube Speech Emotion Recognition Corpus; `audio/`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download of the archive, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 5,215 | unknown | 4.35 |

How these were counted: every `.wav` named `<Emotion>_<n>_<Gender>.wav` (all 3.0 s, 44.1 kHz mono 16-bit); hours from the file lengths. The only speaker information is the gender in the file name (3,446 male and 1,769 female clips); no speaker ID or video ID is published, so the number of speakers is unknown.

### Files per label

| label (emotion in the file name) | samples |
|---|---|
| Angry | 745 |
| Disgust | 745 |
| Fear | 745 |
| Happy | 745 |
| Neutral | 745 |
| Sad | 745 |
| Surprise | 745 |

How these were counted: the emotion field of each file name.

### Issues in the release

- 49 pairs (98 clips) are near-identical copies: the waveforms correlate above 0.98 (mostly 0.9998 or higher) within 10 ms of each other while the bytes differ, as if the same 3 seconds were saved under two numbers. In 3 pairs the labels differ (`Happy_005_Male` / `Neutral_225_Male`, `Fear_646_Male` / `Neutral_024_Male`, `Fear_645_Male` / `Neutral_293_Male`). Only clips of equal length that line up were searched; windows cut at an offset from the same video and overlapping in part were not.
- The description says the clips come from public YouTube videos; how they were cut and who gave the labels is not stated, and the release has no video ID, speaker ID or transcript.


## Provenance

figshare article 30218872 (version 3, published 2026-03-17), DOI
`10.6084/m9.figshare.30218872.v3`. No login is needed. The record licenses the
curation, annotations and documentation under CC BY 4.0 but states that this does
not extend to the audio, which is third-party YouTube content: the audio is
provided for non-commercial academic and educational research only, and its use is
subject to the YouTube Terms of Service. Check these terms before any use beyond
research or any redistribution. The record lists Jonathon Lenny Baroi, Masiat
Momshad, Raisa Tahasen and Tasnia Hossain as authors; cite the figshare record.
`BRYTSERC.zip` (1,121,541,358 bytes) is checked against its MD5
`988af91f7397d65c8d0753b880a9e0d6`.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BRYTSERC` — the directory holding the extracted wav files, or where to download them to
- `BRYTSERC_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `BRYTSERC.zip`, checks its MD5 and extracts the wav
files. Measured on one run: about 10 minutes to download (curl) and 8 seconds to
extract. `create_dataset`, `remove_long_short` and `prepare_labels` together took
about 5 minutes (the conversion took 280 seconds).

## Splits

No partition, speaker id or source-video id is published, so utterances are split
70/10/20 at random within each label (`train_ratio`, `valid_ratio`, `test_ratio`,
`split_seed` in `dataset/config.yaml`). The clips are 3-second pieces of longer
videos, so clips from the same speaker or video probably appear in several
splits and scores are optimistic. Every class has the same count in each split.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 3,654 | 3,654 |
| valid | 518 | 518 |
| test | 1,043 | 1,043 (not filtered) |

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
