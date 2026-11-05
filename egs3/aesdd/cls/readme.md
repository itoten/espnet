# AESDD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on AESDD alone and scores it on AESDD's own test split, classifying each
utterance into one of five emotions.

[AESDD](https://m3c.web.auth.gr/research/aesdd-speech-emotion-recognition/)
(Acted Emotional Speech Dynamic Database v1.0, Aristotle University of
Thessaloniki, 2017) is acted Greek speech from actors (the original release used professional actors), six
speakers in the files this recipe downloads. Each speaker acts utterances of
ambiguous, out-of-context content in five emotions: 604 recordings with audio,
`angry`, `disgust`, `fear`, `happy` and `sad` (119 to 122 each), 0.69 hours in
total, each 1.0 to 12.7 seconds long. There is no `neutral` or `surprise`.

The emotion comes from the folder (`anger`, `disgust`, `fear`, `happiness`,
`sadness`); the file name `<code><utterance> (<speaker>)[<take>].wav` (e.g.
`a03 (4).wav`) starts with the same one-letter code, and `create_dataset` fails
if the two disagree. One recording, `sadness/s05 (3).wav`, is a WAV header with
no samples, so it is skipped. The sources are 44.1 kHz mono (float32 or 24-bit
PCM), so `create_dataset` resamples them to 16 kHz 16-bit with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `AESDD` — the downloaded corpus, or where to download it to
- `AESDD_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the AESDD download (Acted Emotional Speech Dynamic Database, from the project page of Aristotle University of Thessaloniki; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from the project page, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 605 | 6 (see below) | 0.69 |

How these were counted: every `.wav` under the five emotion folders (44.1 kHz mono, float32 in 504 files and 24-bit in 100; 0.98 to 12.74 s); hours from the file lengths (one file, `sadness/s05 (3).wav`, has a header and no samples). Speakers: the file names `<code><utterance> (<speaker>)[<take>].wav` carry a speaker number 1 to 6, and five files carry `(01)`, read as speaker 1 (an inference). The v1.0 description speaks of five actors (three women, two men, aged 25 to 30); the attributes of a sixth are not given.

### Files per label

| label (folder name in the release) | samples | hours |
|---|---|---|
| anger | 121 | 0.121 |
| disgust | 122 | 0.155 |
| fear | 120 | 0.125 |
| happiness | 119 | 0.133 |
| sadness | 123 | 0.156 |

How these were counted: the emotion folder each file is in; the first letter of the file name agrees with the folder. The release has no neutral and no surprise class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- `sadness/s05 (3).wav` is a 36-byte file that holds only a WAV header.
- The v1.0 description says five speakers; the files carry six speaker numbers.


## Download

`prepare_source` downloads the corpus from the public MEGA folder linked on the
corpus's own page (about 410 MB, with its documentation). MEGA encrypts files
client-side, so the builder decrypts them itself with the folder key from that
link. The corpus page says the corpus is publicly available for research purposes
(the words "research non-commercial" there are about an extended version built from
recordings sent to the authors' web site); there is no licence file.

## Listening test

The corpus page reports a subjective evaluation of the database (Vryzas et al.,
Audio Mostly 2018, "Subjective Evaluation of a Speech Emotion Recognition
Interaction Framework"): human listeners recognised the intended emotion about 74% of
the time. This figure is a rough human reference for the five classes. The number of
listeners and the per-utterance answers are not in the release, and the labels used here
are the intended emotions.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 424 |
| valid | 60 |
| test | 120 |

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
