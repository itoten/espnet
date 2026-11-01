# jlcorpus speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on jlcorpus alone, and scores it on jlcorpus's own test split (10
native classes).

[JL-Corpus](https://github.com/tli725/JL-Corpus) is New Zealand English read
by four actors; five of its ten emotions are the usual primary ones, the
other five secondary (`anxious`/`apologetic`/`assertive`/`concerned`/
`encouraging`). The label comes from the file name; no annotation file is
read. The audio is 44.1 kHz, so `create_dataset` resamples it to 16 kHz.

The corpus is distributed on
[Kaggle](https://www.kaggle.com/datasets/tli725/jl-corpus), which needs an API
token (`pip install kaggle`, then a token from
https://www.kaggle.com/settings/api in `~/.kaggle/access_token` or
`$KAGGLE_API_TOKEN`). `JLCORPUS` points at an existing corpus tree, or where
to download it to.

Seven pairs of clips have the same or nearly the same sound: six are the `a` and `b`
versions of one sentence by `female1` in `excited` (`5a`/`5b`, `6a`/`6b` and `9a`/`9b`,
takes 1 and 2), and one is `male2_neutral_15b_1`/`15b_2`. They are kept as they are.

The corpus publishes no partition, and it has four speakers. One is held out
for test; validation is drawn from the remaining three, because a second
held-out speaker would leave two for training.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Kaggle dataset `tli725/jl-corpus` (JL-Corpus; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Kaggle download with a free account and API token.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 2,400 | 4 | 1.41 |

How these were counted: every `.wav` in `Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/` (44.1 kHz mono, 1.2 to 3.75 s); hours from the file lengths. Speakers are the 4 speaker names in the file names `<speaker>_<emotion>_<sentence><session>_<repetition>.wav` (`female1`, `female2`, `male1`, `male2`). The same folder appears twice in the download (see below); the counts are those of one copy. Each copy also holds a `Perception test material on Qualtrics` folder with 2,440 further `.wav` files and 1,240 `.mp3` files, which were not counted.

### Files per label

| emotion in the file name | samples |
|---|---|
| angry | 240 |
| anxious | 240 |
| apologetic | 240 |
| assertive | 240 |
| concerned | 240 |
| encouraging | 240 |
| excited | 240 |
| happy | 240 |
| neutral | 240 |
| sad | 240 |

How these were counted: the emotion field of each file name (four speakers x ten emotions x 60 files). Five are the usual primary emotions (`angry`, `excited`, `happy`, `neutral`, `sad`) and five are secondary (`anxious`, `apologetic`, `assertive`, `concerned`, `encouraging`, as in the README; the names of the secondary emotions differ between documents).

### Issues in the release

- The download holds two folders whose names differ only in case: `Raw JL corpus (unchecked and unannotated)` and `raw jl corpus (unchecked and unannotated)`, and `Picture Stimuli` and `picture stimuli`; each pair has the same content, so every file is present twice.
- Seven pairs of clips have the same or nearly the same sound: six are the `a` and `b` versions of one sentence by `female1` in `excited` (`5a`/`5b`, `6a`/`6b`, `9a`/`9b`, takes 1 and 2) and one is `male2_neutral_15b_1` / `15b_2`; the labels agree within each pair.
- The folder is named "unchecked and unannotated" although the file names carry the emotion.


## License and citation

The Kaggle listing gives the licence CC0 1.0. The corpus README asks users to cite Jesin
James, Li Tian and Catherine Watson, "An Open Source Emotional Speech Corpus for Human
Robot Interaction Applications", Interspeech 2018. The corpus was evaluated in a
perception test with 120 participants; the survey links are in the README and the
ratings are not bundled.

## Quick start

```bash
# 1) Write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`JLCORPUS_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.

**Duplicates.** 7 recording(s) are byte-identical to another one (one per group is kept). They are listed in `duplicate_ids` and are dropped only when `exclude_duplicates` in `dataset/config.yaml` is `true` (default `false`).
