# NonverbalTTS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on NonverbalTTS alone and scores it on NonverbalTTS's own test split,
classifying each utterance into one of seven emotions.

[NonverbalTTS](https://huggingface.co/datasets/deepvk/NonverbalTTS) is an
English corpus of clips cut from VoxCeleb (interviews, 16 kHz) and Expresso
(acted, 48 kHz), built for text-to-speech with nonverbal vocalizations
(breathing, laughter and so on) marked in the transcripts. Each clip also has an
`Emotion` tag: the dataset card says it was first predicted automatically with
emotion2vec+ and then checked by human annotators. This recipe uses only that
tag.

The Hugging Face dataset has 6,256 rows (17.6 hours, 0.5 to 116.7 seconds) in
four published splits: `train` (3,641), `dev` (46), `test` (359) and `other`
(2,210). The recipe uses `train`, `dev` and `test` and does not download
`other`: neither the dataset card nor the paper's text describes it, its speakers
overlap with `train` and `test`, and 522 of its rows have no emotion tag. Rows
tagged `other` (833) or without a tag (12) are not emotion classes and are
dropped, and the log reports their numbers. The remaining 3,201 clips (8.1
hours, 0.57 to 29.8 seconds) carry the seven labels `angry`, `disgust`, `fear`,
`happy`, `neutral`, `sad` and `surprise`.

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
each SHA-256. Measured on one run (all 12 files, 4.3 GB): about 4.7 minutes to
download; `create_dataset`, `remove_long_short` and `prepare_labels` together took
about 3 minutes.

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
