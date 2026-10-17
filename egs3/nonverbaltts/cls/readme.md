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
of the 1,274 `neutral` clips are VoxCeleb. A model can partly use the recording
condition (sampling rate, acting style) rather than the emotion.

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
