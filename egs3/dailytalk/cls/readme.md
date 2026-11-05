# DailyTalk speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on DailyTalk alone and scores it on DailyTalk's own test split, classifying each
utterance into one of seven emotions.

[DailyTalk](https://github.com/keonlee9420/DailyTalk) is an English conversational
speech corpus built for text-to-speech (Lee, Park and Kim, 2023,
[arXiv:2207.01063](https://arxiv.org/abs/2207.01063)): two speakers recorded
2,541 dialogues taken from the DailyDialog text corpus, 23,773 utterances and 21.7
hours in total, 0.5 to 17.8 seconds long. The recordings are 44.1 kHz; 23,720 are
mono and 53 are stereo. One utterance is left out (see below), so 23,772 are used.

**Why this corpus matters.** Its utterances are the turns of 2,541 two-person
conversations (acted from DailyDialog scripts), and `metadata.json` keeps the turn
order, topic and dialogue act, with the text of every turn. The recipe does
not use that structure yet, but future work on dialogue-level emotion recognition, context-aware SER or
conversational speech models would need exactly this kind of data, so keep the
dialogue information in mind when changing the recipe. The split below keeps each
dialogue in one split for that reason.

The label is the per-utterance `emotion` field of `metadata.json`, inherited from
DailyDialog's text annotation, so it was not assigned by listening to the audio.
`dataset/config.yaml` maps it to `neutral` (`no emotion`), `happy`, `sad`, `angry`,
`fear`, `disgust` and `surprise`. The distribution of the 23,772 used utterances is
highly skewed:

| label | utterances |
|---|---|
| neutral | 18,965 |
| happy | 3,856 |
| surprise | 407 |
| sad | 301 |
| angry | 159 |
| disgust | 66 |
| fear | 18 |

One utterance is left out: `dailytalk-d1253-u009` (in the valid split) has audio
byte-identical to `dailytalk-d1121-u009`, which is in the test split
(`exclude_files` in `dataset/config.yaml`). Dialogues 1121 and 1253 are almost the
same dialogue in DailyDialog, and only this utterance reuses the same recording.
Four other pairs of dialogues have identical text (177 and 532, 288 and 1698,
507 and 1383, 1069 and 2047) but were recorded separately.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `DAILYTALK` — the directory holding the extracted `metadata.json` and `data/` (also found one level down, in `dailytalk/`), or where to download them to
- `DAILYTALK_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the DailyTalk release (`metadata.json` and the audio of the GitHub project `keonlee9420/DailyTalk`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from the project's shared storage, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 23,773 | 2 | 21.67 |

How these were counted: every utterance in `metadata.json` (2,541 dialogues; 44.1 kHz, 23,720 mono and 53 stereo, 0.5 to 17.8 s); hours from the audio lengths. Speakers are the 2 distinct `speaker` values (both appear in every dialogue).

### Files per label

| label (the `emotion` field of `metadata.json`) | samples | hours |
|---|---|---|
| no emotion | 18,966 | 17.142 |
| happiness | 3,856 | 3.532 |
| surprise | 407 | 0.374 |
| sadness | 301 | 0.349 |
| anger | 159 | 0.187 |
| disgust | 66 | 0.071 |
| fear | 18 | 0.020 |

How these were counted: the `emotion` field of every utterance of `metadata.json`. The field is inherited from the text annotation of the DailyDialog corpus; the voice actors performed accordingly. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- `d1121-u009` (test) and `d1253-u009` (validation in the recipe's split) are byte-identical recordings with transcripts that differ only in punctuation; dialogues 1121 and 1253 are almost the same dialogue.
- Four dialogues have the same full text as another dialogue; their audio was recorded separately.
- Observation by the recipe author (2026-10-24 at the latest): the identical pair `d1121-u009` and `d1253-u009` was listened to and is the same sound.

### Not stated in the documentation

- The labels are not the result of listening to the audio: they come from the text-dialogue annotation. The release also keeps the turn order, topic and dialogue act of every utterance (`metadata.json`).


## Download

`prepare_source` downloads `dailytalk.zip` (5.3 GB, md5-checked) from the public
Google Drive folder linked in the GitHub README and extracts the wav files and
`metadata.json`. Measured with `gdown` on one run: about 3.3 minutes to download,
45 seconds to extract, and 6 seconds for the md5. `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 3.5 minutes (ffmpeg
resampling with 8 threads).

## Splits

The corpus publishes no partition. Both speakers (`0` and `1` in `metadata.json`)
take part in every dialogue, so speakers cannot be held out. Whole dialogues are
assigned at random to 70/10/20 (`split_seed` in `dataset/config.yaml`), so the
utterances of one conversation never fall in different splits. The splits share
both speakers, and the rarest classes are tiny in valid and test.

| split | dialogues | utterances | after `remove_long_short` |
|---|---|---|---|
| train | 1779 | 16,694 | 16,694 |
| valid | 254 | 2,395 | 2,395 |
| test | 508 | 4,683 | 4,683 (not filtered) |

Test labels: neutral 3715, happy 791, surprise 83, sad 52, angry 25, disgust 11,
fear 6.

## License and citation

The GitHub README states CC BY-SA 4.0. Cite Lee, Park and Kim,
"DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech",
arXiv:2207.01063.

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
