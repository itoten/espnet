# BASER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BASER alone and scores it on BASER's own test split, classifying each
utterance into one of five emotions.

BASER (Bangla Adolescent Speech Emotion Recognition) is an acted Bengali corpus.
The Kaggle description says it was recorded by 12 non-professional adolescent
actors aged 12 to 19, six male and six female. The copy used here has **10**
speakers (five female, five male), each reading 5 sentences in five emotions
(`angry`, `fear`, `happy`, `neutral`, `sad`) with 3 takes. That gives 750
recordings, 150 per emotion (75 per speaker), 0.61 hours in total, 1.71 to
5.01 seconds long. They are 48 kHz mono and are converted to 16 kHz. Why the
copy has 10 speakers instead of 12 is not known.

58 of the 750 files are duplicates and are excluded (see below), so 692
recordings are used: angry 139, fear 137, happy 139, neutral 139, sad 138.

The label and the speaker are taken from the file name
`<F|M>_<n>_<NAME>_S_<sentence>_<EMOTION>_<take>.wav` (for example
`M_4_RIAD_S_2_ANGRY_3.wav`); `dataset/config.yaml` maps the emotion to the
manifest label. No transcripts are provided, so what each sentence number says
is unknown.

Things to know when reading the scores:

- Duplicates are excluded (`exclude_files` in `dataset/config.yaml`). 692 of the
  750 files have distinct content. Speaker RIAD's three takes of every sentence
  and emotion are byte-identical (50 redundant files); JUHAYER has 3 redundant
  files, and MAHI, MAHMUD, MIFTA and TABASSUM one each, and the two takes
  `M_2_MIFTA_S_1_SAD_2` and `_3` have different bytes but the same waveform.
  From each group only the lowest take is kept, 58 files are left out in all.
  The duplicates were confirmed by listening on 2026-10-23.
- The file names have a few irregularities: three have no take number
  (`F_1_CHAITY_S_5_SAD_.wav`, `F_2_IRA_S_1_HAPPY_.wav`,
  `M_3_MAHMUD_S_1_NEUTRAL_.wav`), one has take 21
  (`M_5_JUHAYER_S_4_ANGRY_21.wav`), and one speaker is written `Tahsin` instead
  of upper case.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `jabedcse/baser-dataset` (BASER; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 750 | 10 | 0.61 |

How these were counted: every `.wav` in the archive (48 kHz mono, 1.71 to 5.01 s); hours from the file lengths. Speakers are the 10 distinct speaker names in the file names `<F|M>_<n>_<NAME>_S_<sentence>_<EMOTION>_<take>.wav` (five female, five male; 15 files per emotion per speaker from 5 sentences and 3 takes).

### Files per label

| emotion in the file name | samples |
|---|---|
| ANGRY | 150 |
| FEAR | 150 |
| HAPPY | 150 |
| NEUTRAL | 150 |
| SAD | 150 |

How these were counted: the emotion field of each file name.

### Issues in the release

- The Kaggle description says 12 actors (six male, six female); the files have 10 speakers. The reason is not known.
- 58 files have the same sound as another file of the same speaker and emotion: speaker `RIAD` has three byte-identical takes in all 25 (sentence, emotion) combinations (50 redundant files); `JUHAYER` has 3 redundant files and `MAHI`, `MAHMUD`, `MIFTA` and `TABASSUM` one each; the takes `M_2_MIFTA_S_1_SAD_2` and `_3` differ in bytes but have the same waveform.
- Observation by the recipe author (2026-10-23): the duplicate groups were listened to and are the same sound.
- Three file names have no take number (`F_1_CHAITY_S_5_SAD_.wav`, `F_2_IRA_S_1_HAPPY_.wav`, `M_3_MAHMUD_S_1_NEUTRAL_.wav`), one has take 21 (`M_5_JUHAYER_S_4_ANGRY_21.wav`), and one speaker is written `Tahsin` without upper case.
- The release has no transcripts, so what each sentence number says is unknown; the page does not describe the recording method.


## Provenance

Kaggle dataset `jabedcse/baser-dataset` (version 1, 2023-12-16), licence shown on
Kaggle: Apache 2.0. The required citation was not checked; see the Kaggle page.

Kaggle requires a login, so the recipe does not download. Download the zip and
extract it into `$BASER` so that `BASER_DATASET/` holds the 750 wav files.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BASER` — the directory holding `BASER_DATASET/`
- `BASER_OUTPUT` — where the converted audio and manifests are written

## Download

The zip (149 MB) took 8 seconds to download. `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 63 seconds.

## Splits

No partition is published. Whole speakers are held out
(`split_seed` in `dataset/config.yaml`): one female (`F4`, TABASSUM) and one male
(`M4`, RIAD) speaker for test, one more (`F1`, CHAITY) for valid, and the other
seven for training. Scores are therefore speaker-independent, but the
sentences are shared across splits. The duplicates are removed before the split,
so RIAD (test) has 25 files and TABASSUM (test) 74.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 518 | 518 |
| valid | 75 | 75 |
| test | 99 | 99 (not filtered) |

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
