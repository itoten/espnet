# UTeMo speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on UTeMo alone and scores it on UTeMo's own test split, classifying each
utterance into one of seven emotions.

UTeMo (Universidad Tecnologica de la Mixteca Emotion Dataset) is an acted
audiovisual corpus in Mexican Spanish: 14 actors (7 female, 7 male, as stated by
the authors) performed sentences in the six basic emotions plus neutral. The
release (Mendeley Data version 2) has 1,801 AVI videos, which are 2.01 hours of audio in total and
1.9 to 9.9 seconds each; 1,780 are used (21 duplicates are left out, see below). The audio stream of every video is 48 kHz mono 16-bit PCM.

The label is the emotion word in the file name
`<Emotion>_Actor<n>_P<m>_F<l>.avi` (actor 1-14, repetition 1-3, phrase 1-10);
`dataset/config.yaml` maps `Anger`, `Happiness` and `Sadness` to `angry`, `happy`
and `sad` and keeps `disgust`, `fear`, `neutral` and `surprise`. Whether the
emotion is the prompted or a validated one is not stated beyond the authors
calling the videos "validated".

Class distribution of all 1,801 recordings: 254 neutral, 249 angry, 244 disgust,
244 fear, 242 happy, 289 sad and 279 surprise. The corpus is not balanced
per actor (110 to 144 recordings each). 20 groups of videos with identical video
and audio streams exist (21 surplus files), always the same actor, emotion and
phrase in a different repetition. Only the lowest repetition of each group is
kept; the 21 others are listed in `exclude_files` in `dataset/config.yaml`. The
1,780 recordings used are 252 neutral, 246 angry, 244 disgust, 241 fear, 237 happy,
289 sad and 271 surprise.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `UTEMO` — the directory holding the extracted `Utemo/` videos, or where to download them to
- `UTEMO_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Mendeley Data record of UTeMo (Universidad Tecnologica de la Mixteca Emotion Dataset), version 2 (`Utemo.zip` and `README_Mendeley.md`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Mendeley Data, no sign-in. Version 2 has the same `Utemo.zip` as version 1 (same SHA-256, checked by downloading) plus the README file.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,801 | 14 | 2.01 |

How these were counted: every `.avi` video in `Utemo/` (1920x1080, 30 fps; the audio stream is 48 kHz mono 16-bit PCM, 1.9 to 9.9 s); hours from the decoded audio length of each file. Speakers are the 14 actor numbers in the file names `<Emotion>_Actor<n>_P<m>_F<l>.avi` (7 female, 7 male, as stated by the authors; 110 to 144 recordings each).

### Files per label

| label (emotion word in the file name) | samples | hours |
|---|---|---|
| Neutral | 254 | 0.271 |
| Anger | 249 | 0.246 |
| Disgust | 244 | 0.269 |
| Fear | 244 | 0.272 |
| Happiness | 242 | 0.247 |
| Sadness | 289 | 0.390 |
| Surprise | 279 | 0.312 |

How these were counted: the emotion word at the start of each file name. The description calls the videos "validated"; the validation procedure is not described. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- 20 groups (41 files) have identical video and audio streams: always the same actor, emotion and phrase in a different repetition.
- The Mendeley description gives about 105 minutes; the audio of the files totals 2.01 hours.
- Version 2 differs from version 1 only by the added README file: the phrase "the first database" was dropped and "balanced" became "approximately balanced" in the description.


## Download

`prepare_source` downloads `Utemo.zip` (2.5 GB, SHA-256 checked) from
[Mendeley Data](https://data.mendeley.com/datasets/x5rmd28h73/2) (version 2) through its public
file link, with no login, extracts the AVI files and deletes the archive. Measured
on one run: 17 minutes 53 seconds to download (about 2.3 MB/s) and 14 seconds to
extract; `create_dataset`, `remove_long_short` and `prepare_labels` together took
131 seconds (create_dataset alone 107 s, almost all of it extracting the audio
with ffmpeg).

## Splits

The release publishes no partition. Whole actors are held out so that no actor
appears in two splits: the 14 actors are shuffled with `split_seed` in
`dataset/config.yaml`, then 3 go to test (actors 2, 10, 11), 2 to valid (6, 12) and
9 to train. Gender and emotion balance are not enforced, so the neutral class is
smaller in valid and test than in train.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 1,176 | 1,176 |
| valid | 240 | 240 |
| test | 364 | 364 (not filtered) |

| split | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| train | 158 | 156 | 155 | 150 | 202 | 186 | 169 |
| valid | 35 | 35 | 33 | 34 | 22 | 40 | 41 |
| test | 53 | 53 | 53 | 53 | 28 | 63 | 61 |

## License and citation

Mendeley Data lists the dataset under CC BY-NC 3.0; this was taken from the
dataset listing and its text was not checked here. The dataset README states that it is
intended strictly for research and academic purposes. Please cite:

Carrasco-Jimenez, J. A., Tellez-Velazquez, A., Miranda-Luna, R., Orantes-Molina,
A., Garcia-Vazquez, J. P., & Cruz-Barbosa, R. (2026). An audiovisual dataset for
emotion recognition (UTeMo) in Spanish language. Data in Brief (in revision).

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
