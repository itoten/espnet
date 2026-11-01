# NonverbalTTS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on NonverbalTTS alone and scores it on NonverbalTTS's own test split,
classifying each utterance into one of seven emotions.

[NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS) is an
English corpus of clips cut from VoxCeleb (interviews, 16 kHz) and Expresso
(acted, 48 kHz), built for text-to-speech with nonverbal vocalizations
(breathing, laughter and so on) marked in the transcripts. Each clip also has an
`Emotion` tag (one per clip): the dataset card says it was first predicted
automatically with emotion2vec+ and then checked by human annotators.

The Hugging Face dataset has 6,256 clips (17.6 hours, 0.5 to 116.7 seconds).
Two different things in it are called `other`; this readme keeps them apart:

- the **`other` split**: a fourth published split (2,210 clips) next to
  `train` (3,641), `dev` (46) and `test` (359);
- the **`other` emotion tag**: one of the eight values of the `Emotion` column,
  meaning none of the seven named emotions (1,177 clips in all splits).

What the recipe uses and leaves out:

| part of the data | clips | used | reason |
|---|---|---|---|
| `train`, `dev`, `test` splits | 4,046 | yes | the splits used in the paper |
| `other` split | 2,210 | no, not downloaded | not described by the dataset card or the paper, and not among the splits the paper reports (its Table 5 lists only `train`, `dev` and `test`). Its clips do carry annotations |
| `other` emotion tag, in `train`/`dev`/`test` | 833 | no | not one of the seven emotion classes |
| no emotion tag, in `train` | 12 | no | no emotion label (the paper says about 8.5% of clips have none because the annotators did not agree) |
| nonverbal-vocalization marks in the transcripts | — | no | a clip can hold several marks of different types, so they are multi-label, while this recipe predicts one label per clip. No audio is lost by this: the marks are in the same clips as the emotion tags |

The log reports the number of clips dropped for each tag. The remaining 3,201
clips (8.1 hours, 0.57 to 29.8 seconds) carry the seven labels `angry`,
`disgust`, `fear`, `happy`, `neutral`, `sad` and `surprise`.

The label distribution is very skewed, and the test split has only four labels:

| split | speakers | utterances | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|---|---|
| train | 1,159 | 2,867 | 16 | 66 | 8 | 1,170 | 1,274 | 310 | 23 |
| valid | 44 | 44 | 0 | 0 | 0 | 12 | 29 | 3 | 0 |
| test | 140 | 290 | 1 | 0 | 0 | 66 | 217 | 6 | 0 |

The source also confounds emotion: in `train`, the Expresso clips hold all 66
`disgust`, 262 of the 310 `sad` and 686 of the 1,170 `happy` clips, while 1,190
of the 1,274 `neutral` clips are VoxCeleb.

`create_dataset` decodes the audio stored in the parquet files and converts it to
16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `NONVERBALTTS` — the directory holding `default/<split>/*.parquet`, or where to download them to
- `NONVERBALTTS_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the 8 parquet files of `train`, `dev` and `test`
(about 3.1 GB) from Hugging Face at a pinned commit, with no login, and checks
each SHA-256. Measured on one run that downloaded all 11 published parquet files
(4.2 GB): about 4.7 minutes to download; `create_dataset`, `remove_long_short`
and `prepare_labels` together took about 3 minutes.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `deepvk/NonverbalTTS` (11 parquet files; repository revision `947139b9003316e1d369ae2012a3bded83fd7128`, 2025-10-04). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 6,256 | 2,296 | 17.59 |

How these were counted: every row of the 11 parquet files (indexes are all unique); hours from the decoded audio lengths (0.5 to 116.7 s); speakers are the distinct `speaker_id` values (1,380 male, 916 female; 2,292 from VoxCeleb and 4 from Expresso, by the `data_name` column).

### Files per label

| label (the `Emotion` column of the release) | samples |
|---|---|
| neutral | 2,406 |
| happy | 1,605 |
| other | 1,177 |
| sad | 387 |
| disgusted | 78 |
| surprised | 34 |
| angry | 25 |
| fearful | 10 |
| (no tag) | 534 |

How these were counted: the `Emotion` column of all rows. `other` here is an emotion tag, not the `other` split (see below).

### Issues in the release

- The release has four published splits: `train` 3,641, `dev` 46, `test` 359 and `other` 2,210 clips. The `other` split is not described on the dataset card or in the paper (whose Table 5 lists only train, dev and test; its train count is 3,642). Among the 1,378 clips without a nonverbal vocalization, 1,373 are in the `other` split; among the 534 clips without an emotion tag, 522 are; transcripts from all three annotators exist for 47% of its clips, against 75 to 84% in the other splits.
- The Hugging Face metadata field says `apache-2.0`; the card text says the annotations are CC BY-NC-SA 4.0 and the audio follows the licenses of VoxCeleb and Expresso.
- The paper's Table 7 (emotion counts per split) does not match the counts of the `train` and `test` splits; its Table 6 (all clips) matches.
- Nonverbal vocalizations are marked as symbols inside the transcripts (10 types, 4,878 clips contain at least one). The release's own table of symbols could not be fetched; the symbol-to-type mapping was inferred by matching the counts of the paper's Table 2.

### Not stated in the documentation

- The clips come from two sources, separable by the `data_name` column: VoxCeleb 4,452 clips (13.68 hours, 16 kHz) and Expresso 1,804 clips (3.91 hours, 48 kHz).


## Splits

The published `train`, `dev` and `test` are used as they are, with `dev` as the
validation set. According to the paper, the test clips come from VoxCeleb
speakers with at least two recordings and the training data from the remaining
speakers, so the splits do not share speakers (no overlap among the three in
the data). `dev` is tiny (44 clips after filtering), so validation scores are
noisy. All clips are between 0.5 and 30 s, so `remove_long_short` drops nothing.

## License and citation

The Hugging Face metadata says Apache-2.0, but the dataset card says the
annotations are CC BY-NC-SA 4.0 and the audio follows the licenses of VoxCeleb and
Expresso. Check these before redistributing or using the data commercially.
Please cite:

```
@inproceedings{borisov25_ssw,
  title     = {{NonverbalTTS: A Public English Corpus of Text-Aligned Nonverbal Vocalizations with Emotion Annotations for Text-to-Speech}},
  author    = {Maksim Borisov and Egor Spirin and Daria Diatlova},
  year      = {2025},
  booktitle = {{13th edition of the Speech Synthesis Workshop}},
  pages     = {104--109},
  doi       = {10.21437/SSW.2025-16},
}
```

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
