# EmoTale speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoTale alone and scores it on EmoTale's own test split, classifying each
utterance into one of five emotions.

[EmoTale](https://github.com/snehadas/EmoTale) is an enacted corpus in Danish and
English recorded at DTU with wireless microphones in a quiet room. 18 speakers (12
women, 6 men, aged 9 to 39) read five emotionally neutral sentences in five
emotions: `angry`, `boredom`, `happy`, `neutral` and `sad`. 14 of the speakers also
recorded the English sentences, so there are 450 Danish and 350 English files,
800 in total, 160 per emotion. Clips are 1.07 to 7.44 seconds long, 0.617 hours in
total, and are 48 kHz stereo; `create_dataset` converts them to 16 kHz mono with
ffmpeg.

The label is the enacted emotion, taken from the file name
`<DK|EN>_<speaker>_<code>_<sentence>.wav` (`A`, `B`, `H`, `N`, `S`), which equals the
`gt_emotion` column of `annotations.csv` for every file; `dataset/config.yaml` maps
the codes to labels. The repository also holds `speaker_info.csv` and the datasheet,
which this recipe does not use.

**Perceived-emotion labels exist but are not used.** `annotations.csv` also holds, for
every utterance, the judgements of three listeners: a categorical emotion and
arousal, valence and dominance (1 to 5 in steps of 0.5). (The repository README says
two annotators, the datasheet and the file have three.) This recipe trains on the enacted
emotion only.
Measured on the file: the listeners chose the enacted emotion for 639 (a1), 706 (a2)
and 507 (a3) of the 800 utterances; all three agree on 465; two or more agree on
749, and for 113 of those the agreed emotion differs from the enacted one; for 51 the
three disagree.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOTALE` — the directory holding `wav/` and `annotations.csv`, or where to download them to
- `EMOTALE_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the GitHub repository `snehadas/EmoTale` (`wav/`, `annotations.csv`, `speaker_info.csv` and the datasheet; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from GitHub, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 800 | 18 | 0.62 |

How these were counted: every `.wav` in `wav/` (48 kHz stereo, 1.07 to 7.44 s; 450 Danish and 350 English files); hours from the file lengths. Speakers are the 18 distinct speaker codes in the file names `<DK|EN>_<speaker>_<code>_<sentence>.wav` (12 women, 6 men, aged 9 to 39; 14 of them also recorded the English sentences).

### Files per label

| label (code in the file name) | read as | samples |
|---|---|---|
| A | angry | 160 |
| B | boredom | 160 |
| H | happy | 160 |
| N | neutral | 160 |
| S | sad | 160 |

How these were counted: the emotion code in the file name; it equals the `gt_emotion` column of `annotations.csv` for every file (the enacted emotion).

### Issues in the release

- The repository README says two annotators; the datasheet and `annotations.csv` have three.

### Listening test ratings in the release

`annotations.csv` holds, for every one of the 800 utterances, the judgements of three listeners (a1, a2, a3): a categorical emotion and arousal, valence and dominance (1 to 5 in steps of 0.5). Counted on the file: the listeners chose the enacted emotion for 639 (a1), 706 (a2) and 507 (a3) of the 800 utterances; all three listeners agree on 465; two or more agree on 749, and for 113 of those the agreed emotion differs from the enacted one; for 51 the three disagree.


## Download

`prepare_source` downloads the 800 wav files (409 MB) and the two tables from the
public GitHub repository, pinned to commit `41e7d00f`; no login is needed. Measured on
one run: about 42 seconds to download; `create_dataset`, `remove_long_short` and
`prepare_labels` together took about 75 seconds.

## Splits

The release publishes no partition. Whole speakers are held out (`split_seed` in
`dataset/config.yaml`). The speaker id is shared by a speaker's Danish and English
files, which are kept in one split. Of the 18 speakers, 12 are in train (3, 5, 6, 7,
8, 9, 10, 12, 14, 15, 16, 19), 2 in valid (4, 17) and 4 in test (1, 11, 13, 18). Every
speaker recorded each emotion equally often, so every split is balanced over the five
emotions. The test speakers are three women and one man, all adults, and the train
speakers include the three children (ages 9 to 12).

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 525 | 525 |
| valid | 100 | 100 |
| test | 175 | 175 (not filtered) |

## License and citation

**The terms of use are not confirmed.** The repository has no licence file and GitHub
shows no licence. The datasheet answers the question about licences with "The data
will be distributed under a copyright. There is no license, but there is a request to
cite the corresponding paper if the dataset is used." Without a licence, no
permission to copy, redistribute or use the data is written down. The exact terms are
to be confirmed by contacting the authors; until then, treat the terms as unknown and
do not redistribute the audio. Please cite: Maja J. Hjuler, Harald V. Skat-Rørdam,
Line H. Clemmensen and Sneha Das, "EmoTale: An Enacted Speech-emotion Dataset in
Danish", ASRU 2025 (https://arxiv.org/abs/2508.14548).

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
