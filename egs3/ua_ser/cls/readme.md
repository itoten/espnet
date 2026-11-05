# UA-SER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on UA-SER alone and scores it on UA-SER's own test split, classifying each
utterance into one of four emotions.

[UA-SER](https://huggingface.co/datasets/OlhaHavryliuk/UA-SER) is a Ukrainian
emotional speech corpus (CC BY 4.0): 952 short clips (0.52 to 5.5 s, 0.55 hours)
labelled `angry`, `happy`, `neutral` or `sad`, with a speaker id for each clip
(400 speakers). The clips are already 16 kHz mono; `create_dataset` copies them
through ffmpeg unchanged in format.

The card says three native Ukrainian speakers labelled each clip and the majority
vote (at least two of three) is the label; clips without a majority were removed
(Fleiss' kappa 0.46). The card does not say where the audio comes from.

`h_213_1` is the same recording as `h_212_1` (same speaker and label, confirmed by
listening), so it is left out (`exclude_files` in `dataset/config.yaml`).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `UA_SER` — the directory holding `dataset.csv` and `clips/`, or where to download them to
- `UA_SER_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `OlhaHavryliuk/UA-SER` (repository revision `c3fc0b0ef8d40f133d7a74e804ab6f376ef53aa9`, last modified 2026-04-30; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 952 | 400 | 0.55 |

How these were counted: every row of `dataset.csv` (equal to the clips in `clips/`; 16 kHz mono, 0.52 to 5.5 s, mean 2.07 s); hours from the clip lengths. Speakers are the 400 distinct `speaker_id` values (an anonymous number; 593 clips by female and 359 by male voices judged from F0).

### Files per label

| label (as in the release) | samples | hours |
|---|---|---|
| angry | 259 | 0.136 |
| happy | 244 | 0.130 |
| sad | 227 | 0.144 |
| neutral | 222 | 0.136 |

How these were counted: the `emotion` column of `dataset.csv`. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The card does not say where the audio comes from; the recording source and the consent of the 400 speakers are not described.
- `h_212_1` and `h_213_1` are the same recording (the sound is the same; the transcripts differ only in a full stop; both happy, speaker 106).
- Observation by the recipe author (2026-10-20 at the latest): `h_212_1` and `h_213_1` were listened to and are the same.
- The file name prefix is described on the card as an identifier of the corpus source; it is in fact the initial of the emotion. Four speaker IDs occur with both genders (judged from F0).
- Only the majority label is released; the votes and confidences of the three raters are not.
- Whisper identified Ukrainian in 112 and Russian in 65 of the transcribed clips; the transcripts are Ukrainian.


## Download

`prepare_source` downloads `dataset.csv` and the 952 clips (63 MB) from Hugging Face, pinned
to one commit; every file is checked against the size and SHA-256 (or git blob id) the
listing gives. Measured on one run, download plus `create_dataset`, `remove_long_short`
and `prepare_labels` took about 3 minutes 18 seconds.

## Splits

`dataset.csv` marks each clip `train` or `test`; the two share no speaker. The published
split is kept (test 181 clips). A valid split is cut from train by whole speakers
(`valid_speaker_ratio` 0.1, `split_seed` in `dataset/config.yaml`).

| split | utterances |
|---|---|
| train | 707 |
| valid | 63 |
| test | 181 |

No clip is shorter than the 0.2 s floor of `remove_long_short`.

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
