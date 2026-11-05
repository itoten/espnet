# SAVEE speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SAVEE alone and scores it on SAVEE's own test split, classifying
each utterance into one of seven emotions.

[SAVEE](https://cvssp.org/savee/) (Surrey Audio-Visual Expressed Emotion) is
English read speech from four male speakers (`DC`, `JE`, `JK`, `KL`): 480
utterances labelled `neutral` (120), `angry`, `disgust`, `fear`, `happy`, `sad`
and `surprise` (60 each).

According to the corpus's pages (cvssp.org/savee), the four speakers are native
English-speaking men, postgraduate students and researchers at the University of Surrey
aged 27 to 31. The text is 15 TIMIT sentences per emotion (3 common to all emotions, 2
emotion-specific and 10 generic, phonetically balanced); the common and emotion-specific
ones were also recorded as `neutral`, which gives 30 neutral sentences per speaker and 120
utterances per speaker. The recordings were made in a 3D vision laboratory (44.1 kHz audio
with video and facial markers), and utterances in which the actor did not express the
emotion properly were repeated.

The label comes from the file name, `<speaker>_<emotion code><index>.wav`
(e.g. `DC_a01.wav`); no annotation file is read. The recordings are 44.1 kHz,
so `create_dataset` resamples them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SAVEE` — a directory holding the WAV files, or the downloaded zip
- `SAVEE_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the SAVEE (Surrey Audio-Visual Expressed Emotion) audio download from cvssp.org (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Download from the project page after a free registration (the registration is a prerequisite stated on the page).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 480 | 4 | 0.51 |

How these were counted: every `.wav` in the download (44.1 kHz mono 16-bit, 1.63 to 7.14 s); hours from the file lengths. Speakers are `DC`, `JE`, `JK` and `KL` in the file names `<speaker>_<emotion code><index>.wav` (four men).

### Files per label

| emotion code in the file name | read as | samples | hours |
|---|---|---|---|
| n | neutral | 120 | 0.120 |
| a | anger | 60 | 0.062 |
| d | disgust | 60 | 0.066 |
| f | fear | 60 | 0.062 |
| h | happiness | 60 | 0.063 |
| sa | sadness | 60 | 0.075 |
| su | surprise | 60 | 0.063 |

How these were counted: the letters after the speaker in each file name. Neutral has twice as many files because the common and emotion-specific sentences were also recorded neutral (30 neutral sentences per speaker, 120 utterances per speaker). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- No byte-identical, decoded-identical or near-identical recordings were found.

### Listening test ratings in the release

The corpus's evaluation page reports a perception test (Evaluation page, not part of the audio download): each actor's data were evaluated by 10 subjects who chose one of the seven emotions under audio, visual and audio-visual conditions; the individual responses are not in the download.


## Download

**Downloading requires registering.** The corpus is "available free of charge for research
purposes" and a registration at
[cvssp.org/savee/Download.html](https://cvssp.org/savee/Download.html) is needed before
the files can be downloaded, so it cannot be fetched from a script. Put either the extracted WAV files or a single
downloaded zip in `$SAVEE`; `prepare_source` extracts the zip.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all three splits contain
all four speakers.

| split | utterances |
|---|---|
| train | 336 |
| valid | 48 |
| test | 96 |

## Listening test

**The recordings were evaluated by listeners** (the corpus's Evaluation page). Each
actor's data were evaluated by 10 subjects: students at the University of Surrey aged 21 to
29 (mean 24), half of them women; 5 were native English speakers and 5 had lived in the UK
for more than a year. The 120 clips of each actor were split into 10 groups of 12, ordered
per subject with a balanced Latin square, and the subjects chose one of the seven emotions
under three conditions: audio, visual and audio-visual, after training with example
slides, sounds and a video clip for each emotion. The mean accuracy over the 10 subjects
and 4 actors was 66.5% for audio (per actor KL 53.2, JE 67.7, JK 71.2, DC 73.7), 88.0% for
visual and 91.8% for audio-visual; for 441 of the 480 utterances at least 8 of the 10
subjects were right under the audio-visual condition. In the audio condition, disgust was
often confused with neutral, fear with sad and surprise, and happy and surprise with each
other. The individual responses are not in the download; the recipe uses the intended
emotion of the file name.

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
