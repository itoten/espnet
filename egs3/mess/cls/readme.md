# MESS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MESS alone and scores it on MESS's own test split, classifying each
utterance into one of four emotions.

[MESS](https://zenodo.org/records/3813437) (Morgan Emotional Speech Set;
Morgan, 2019) is English speech from 6 talkers (3 female, 3 male) in four
emotional speaking styles, `angry`, `calm`, `happy` and `sad`: 1800 utterances,
450 per style, 1.33 hours in total, each 1.7 to 3.6 seconds long. Each
utterance is one of the sentence frames "Michael / Theo / Victor discussed the X
and the Y today" with different words filled in. `calm` is not one of the seven
labels shared by the other recipes (`angry`, `disgust`, `fear`, `happy`,
`neutral`, `sad`, `surprise`) and is kept as its own label.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 3813437 (MESS, Morgan Emotional Speech Set; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,800 | 6 | 1.33 |

How these were counted: every `.wav` in the archive (24.4 kHz mono 16-bit, 1.73 to 3.65 s); hours from the file lengths. Speakers are 6 talkers (3 female, 3 male), encoded in the file names `<emotion><gender><speaker><frame><item>_SCR.wav`.

### Files per label

| emotion letter in the file name | read as | samples |
|---|---|---|
| A | angry | 450 |
| C | calm | 450 |
| H | happy | 450 |
| S | sad | 450 |

How these were counted: the first letter of each file name (225 files per emotion and gender). The documentation describes the labels as emotional speaking styles, with `calm` used instead of `neutral`.

### Issues in the release

- The paper states a 36 kHz recording; the released files are 24.4 kHz.
- The meaning of the suffix `_SCR` in the file names is not documented. The release has no ratings; the author provides rating data on request.


## About the corpus

MESS was made for experiments on the auditory perception of emotion, not for
training classifiers (Morgan, 2019, Journal of Speech, Language, and Hearing Research
62(11), 4015-4029; the paper's author manuscript is open access).

- **Talkers.** Six students of the Department of Theater, University of Utah (3 men,
  3 women, 19-21 years, native speakers of American English), who had completed all
  the vocal production coursework of their program. They were chosen for their
  training in emotional vocal production, not for naturalness.
- **Emotions.** Four speaking styles, each one quadrant of the activation by
  pleasantness plane: angry (high activation, low pleasantness), happy (high, high),
  sad (low, low) and calm (low activation, high pleasantness). The author used `calm`
  instead of `neutral` because neutral speech has been found to carry a negative
  valence.
- **Sentences.** Frames of the Theo-Victor-Michael sentences of Helfer and Freyman
  (2009), "[Michael / Theo / Victor] discussed the [word 1] and the [word 2] today",
  with one- or two-syllable nouns. They are semantically neutral relative to the
  emotions; words that primed an emotion (e.g. "joy") were replaced by neutral ones
  (e.g. "lint"). Men and women read the same sets of sentences, and within a gender
  each talker read different sentences.
- **Recording.** Two sessions per talker (rehearsal, then recording on another day), in a
  sound-treated room, Shure SM10 head-mounted microphone and a Marantz digital recorder,
  36 kHz (the release is 24.4 kHz). Misread, clipped or noisy sentences were re-recorded.
  Of the 30 sentences recorded for each cue name, emotion and talker, 5 were removed
  (errors first, then those farthest from the mean activation and pleasantness of their
  emotion), leaving 25: 25 x 3 cue names x 4 emotions x 3 talkers x 2 genders = 1,800.

The label comes from the file name,
`<emotion><gender><speaker><frame><item>_SCR.wav` (e.g. `AF1M01_SCR.wav` is
angry, female talker 1, frame M, item 01). The recordings are 24.4 kHz mono,
so `create_dataset` resamples them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MESS` — the extracted WAV files, or where to download them to
- `MESS_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads `MESS compressed.zip` (192 MB) from
[Zenodo](https://zenodo.org/records/3813437). The corpus is CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Talkers are not held out, so all three splits contain
all 6 talkers.

| split | utterances |
|---|---|
| train | 1260 |
| valid | 180 |
| test | 360 |

## Listening test

**This corpus comes with a human perception study** (Morgan, 2019). Ten young adult
listeners (5 men, 5 women, 19-28 years, mean 24; native speakers of American English, 9,
or Canadian English, 1; all passed a hearing screening) rated **all 2,160 recorded
sentences**, including the ones later removed, in six sessions of about 90 minutes. For
each sentence a listener chose an emotion category, rated the activation and the
pleasantness on a 0-100 slider, and rated the goodness of fit of the category.
The share of judgments that chose the intended emotion (the unbiased hit rate of Wagner,
1993, in parentheses) was calm 0.88 (0.54), sad 0.70 (0.63), happy 0.85 (0.79) and angry
0.86 (0.83); the most common confusion was sad judged as calm (0.28). The paper notes that
`calm` was over-selected, so its true recognition rate is uncertain. The ratings are
available from the author on request and are not in the release; the recipe uses only the
intended emotion of the file name.

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
