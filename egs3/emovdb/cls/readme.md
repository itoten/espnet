# EmoV-DB speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoV-DB alone and scores it on EmoV-DB's own test split, classifying
each utterance into one of five emotions.

[EmoV-DB](https://github.com/numediart/EmoV-DB) is English read speech from
four actors (`bea` and `jenie` female, `josh` and `sam` male), recorded for
emotional speech synthesis. It has 6893 utterances in five emotions: `amused`,
`angry`, `disgust`, `neutral`, `sleepy`. `josh` recorded only `amused`,
`neutral` and `sleepy`.

The corpus comes as one archive per speaker and emotion, and the file names
inside carry neither (the same name appears in several archives). Each archive
is therefore extracted into its own `<speaker>_<Emotion>/` directory, and the
label is read from that directory name. The recordings mix 44.1 kHz and
16 kHz, so `create_dataset` resamples all of them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOVDB` — an existing `<speaker>_<Emotion>/` tree, or where to download it to
- `EMOVDB_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the EmoV-DB archives (18 archives, one per speaker and emotion, from OpenSLR / the `numediart/EmoV-DB` project; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download of the 18 archives, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 6,893 | 4 | 9.49 |

How these were counted: every `.wav` in the 18 archives (mono; 16 kHz in 5,106 files and 44.1 kHz in 1,787; 0.59 to 13.16 s); hours from the file lengths. Speakers are `bea` and `jenie` (female) and `josh` and `sam` (male).

### Files per label

| label (as in the release) | samples | hours |
|---|---|---|
| Amused | 1,317 | 1.889 |
| Angry | 1,268 | 1.563 |
| Disgusted | 1,019 | 1.632 |
| Neutral | 1,568 | 1.694 |
| Sleepy | 1,721 | 2.712 |

By speaker and label (one row per archive):

| speaker | label | samples | hours |
|---|---|---|---|
| bea | Amused | 296 | 0.341 |
| bea | Angry | 304 | 0.318 |
| bea | Disgusted | 333 | 0.489 |
| bea | Neutral | 357 | 0.390 |
| bea | Sleepy | 497 | 0.854 |
| jenie | Amused | 222 | 0.306 |
| jenie | Angry | 496 | 0.728 |
| jenie | Disgusted | 189 | 0.260 |
| jenie | Neutral | 417 | 0.525 |
| jenie | Sleepy | 466 | 0.633 |
| josh | Amused | 298 | 0.336 |
| josh | Neutral | 302 | 0.290 |
| josh | Sleepy | 263 | 0.349 |
| sam | Amused | 501 | 0.906 |
| sam | Angry | 468 | 0.517 |
| sam | Disgusted | 497 | 0.884 |
| sam | Neutral | 492 | 0.488 |
| sam | Sleepy | 495 | 0.877 |

How these were counted: the speaker and emotion name of each archive; the file names inside carry neither (the same name appears in several archives), so each archive must be extracted to its own directory. `josh` recorded only Amused, Neutral and Sleepy. Hours per label are measured in the same way as the total hours above and summed over the rows of that label. In this second table the hours are those of the files of each archive.

### Issues in the release

- The dataset page and OpenSLR list 7,000 utterances; the archives hold 6,893.
- File names do not identify speaker or emotion.


## Download

`prepare_source` downloads the 18 archives from
[OpenSLR SLR115](https://www.openslr.org/115/) (about 1.4 GB, no account
needed). Use is limited to non-commercial purposes; see the corpus's
[LICENSE.md](https://github.com/numediart/EmoV-DB/blob/master/LICENSE.md).

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all three splits contain
all four speakers.

| split | utterances |
|---|---|
| train | 4824 |
| valid | 690 |
| test | 1379 |

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
