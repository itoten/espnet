# AlbEmo speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on AlbEmo alone and scores it on AlbEmo's own test split, classifying each
utterance into one of seven emotions.

[AlbEmo](https://zenodo.org/records/21638085) (an emotional speech corpus for
Albanian, University of Tirana and University of Victoria, 2026) is acted
Albanian read speech from 10 native speakers (5 female, 5 male, aged 18 to 55+),
each reading the same six sentences in seven emotions: 420 utterances, 60 per
emotion (`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`),
0.37 hours in total, each 1.2 to 10.1 seconds long.

The label comes from the emotion folder,
`recordings/<speaker>/<emotion>/sentence_<n>.wav`, whose name is a
transliterated Albanian emotion word (e.g. `recordings/sp02/trishtim/sentence_4.wav`
is sad); `dataset/config.yaml` maps the seven folder names to labels. The
recordings are 48 kHz mono, so `create_dataset` resamples them to 16 kHz with
ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ALBEMO` — the extracted `albemo-v1.0.zip`, or where to download it to
- `ALBEMO_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 21638085 (AlbEmo v1.0, `albemo-v1.0.zip`; downloaded around 2026-10-25). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 420 | 10 | 0.37 |

How these were counted: every `.wav` under `recordings/<speaker>/<emotion>/` (48 kHz mono 16-bit, 1.20 to 10.07 s); hours from the file lengths. Speakers are the 10 folders `sp01` to `sp10` (five women, five men, aged 18 to 55+; `speaker-demographics.csv`).

### Files per label

| label (emotion folder, transliterated Albanian) | read as (recipe mapping) | samples |
|---|---|---|
| neutral | neutral | 60 |
| gezim | happy | 60 |
| trishtim | sad | 60 |
| zemerim | angry | 60 |
| frike | fear | 60 |
| habi | surprise | 60 |
| neveri | disgust | 60 |

How these were counted: the emotion folder of each recording (six sentences, read by every speaker in all seven emotions).

### Issues in the release

- The description names `scripts/`, `tools/` and `README.md`, which are not in the zip.
- In `evaluations/`, the files `audit_ev01_sp04.csv` to `audit_ev01_sp08.csv` are named ev01 but their `rater` column says ev02, so counting by that column gives the recordings of speakers sp05 to sp08 (168 clips) only two raters (ev02 and ev03). Some files have extra columns that look like rater notes (`Emocioni`, `Gjetjet`, `Rastet random`).
- The notes in `corpus-sentences.json` mention a speaker `sp12` who has no recordings.

### Listening test ratings in the release

`evaluations/` holds a blind forced-choice listening test of every clip: four native Albanian raters (`ev01` to `ev04`) gave 1,260 judgements, three independent ones per clip (a rater never judged a recording of their own). Counted on the files: the raters picked the intended emotion in 73% of the judgements; the strict majority differs from the intended emotion for 63 clips (disgust 23, fear 10, neutral 7, happy 7, surprise 6, angry 5, sad 5) and none of the three raters picked it for 31 clips (disgust 15).


## Download

`prepare_source` downloads `albemo-v1.0.zip` (92 MB) from
[Zenodo](https://zenodo.org/records/21638085). The corpus is CC BY 4.0.

## Listening test

The release holds a blind forced-choice listening test of every clip: four native Albanian
raters (`ev01` to `ev04`) gave 1,260 judgements, three independent ones per clip
(a rater never judged a recording of their own). The raters picked the intended emotion
in 73% of the judgements (chance is 14%; kappa 0.59 per the Zenodo record), and the
per-rater files are in `evaluations/`, with a summary. The strict majority differs from the
intended emotion for 63 clips and no rater picked it for 31. These ratings are a useful
human reference and for later analysis; the recipe does not use them, and the label is
the intended emotion (the folder name). In the files, `audit_ev01_sp04.csv` to
`audit_ev01_sp08.csv` are named ev01 but carry `rater` ev02, so counting by that column
gives sp05 to sp08 only two raters.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 294 |
| valid | 42 |
| test | 84 |

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
