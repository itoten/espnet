# RML Emotion Database speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the RML Emotion Database alone and scores it on its own test split,
classifying each utterance into one of six emotions.

**License: Unknown.** The Kaggle page lists the license as "Unknown", and no
terms of use were found. Check with the Ryerson Multimedia Lab before using the
data beyond internal research.

[RML Emotion Database](https://www.kaggle.com/datasets/ryersonmultimedialab/ryerson-emotion-database)
(Ryerson Multimedia Lab, Kaggle) is a set of 720 acted audio-visual clips, 120
in each of six emotions: `angry`, `disgust`, `fear`, `happy`, `sad` and
`surprise` (there is no neutral class). The Kaggle release is avi video with a
stereo audio track (44.1 kHz in 669 clips, 48 kHz in 51), 2.13 to 8.37 s long,
0.99 hours in total. `create_dataset` extracts the audio as 16 kHz mono wav with ffmpeg.

Files are laid out as `s<N>/s<N>/[f<M><lang>/]<emo><take>.avi`. The emotion is
the two-letter code in the file name (`an`, `di`, `fe`, `ha`, `sa`, `su`). `s1`
to `s8` are treated as 8 speakers, which the Kaggle page does not confirm. Subjects
`s3` to `s8` have language folders (`eng`, `chi`, `per`, `urd`, `pjb`, `sdh`, `hnh`); `s1`
and `s2` have none. The meaning of the language codes is taken from their
spelling, not from any documentation. Eleven clips form five groups of byte-identical files
(same subject, folder and emotion); they are kept.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `RML_EMOTION` — the directory holding the extracted `s1/` ... `s8/` folders
- `RML_EMOTION_OUTPUT` — where the converted audio and manifests are written

## Download

The recipe does not download the data. Download the zip from the Kaggle page
above (3.4 GB, 4.5 GB unpacked, free Kaggle account) and unzip it into `$RML_EMOTION`.
If the folders are missing, `create_dataset` stops with an error that gives the
URL. The download time was not measured; `create_dataset`, `remove_long_short`
and `prepare_labels` together took about 61 seconds.

## Splits

The dataset publishes no partition. Whole subject folders are assigned at random
(`split_seed` 0) to test, then valid, then train, so no subject spans two
splits. Because the folders differ in size (25 to 182 clips), the ratios are only
approximate and the labels are not balanced across splits.

| split | utterances | subjects |
|---|---|---|
| train | 420 | s1, s4, s7, s8 |
| valid | 80 | s3 |
| test | 220 | s2, s5, s6 |

No clip is shorter than the 0.5 s floor of `remove_long_short`, and none is longer than 30 s.

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
