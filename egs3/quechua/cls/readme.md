# quechua speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on quechua alone, and scores it on quechua's own test split (9
classes: `angry`/`boredom`/`calm`/`excited`/`fear`/`happy`/`neutral`/`sad`/
`sleepy`, spelled to match the other corpora here; the corpus's own spelling
is `anger`/`sadness`).

The [Quechua Collao corpus](https://doi.org/10.6084/m9.figshare.20292516) is
12,420 recordings by six actors performing 230 script items in nine emotions.
Labels come from `Data/Data/Data.xlsx`, read as a zip of XML rather than
adding a spreadsheet dependency. `QUECHUA` points at an existing corpus tree,
or where to download it to; the figshare archive is checked against its
published MD5.

No partition is published; speakers are dealt out in order (a1-a4 train, a5
valid, a6 test) after the metadata repairs `dataset/config.yaml` documents
(two mis-spelled emotions, two mis-typed speaker ids, 42 mis-attributed
`happy` recordings).

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the figshare record 20292516 (Quechua Collao emotional speech corpus; `Audios/`, `Data/Data/Data.xlsx`, `Labels/` and `Script/`, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from figshare, no sign-in (the archive is checked against its published MD5).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 12,420 | 6 | 15.24 |

How these were counted: every `.wav` in `Audios/` (mostly 44.1 kHz mono: 12,409; 7 files at 16 kHz mono and 4 at 48 kHz stereo; 0.75 to 21.83 s); hours from the file lengths. Speakers are the 6 actor codes `a1` to `a6` of `Data.xlsx` (the release has no actor attributes).

### Files per label

| label (the `Emotion` column of `Data.xlsx`, as written) | samples | hours |
|---|---|---|
| happy | 1,380 | 1.866 |
| sleepy | 1,380 | 1.621 |
| sadness | 1,380 | 1.829 |
| calm | 1,380 | 1.523 |
| fear | 1,380 | 1.596 |
| excited | 1,380 | 1.654 |
| neutral | 1,380 | 1.735 |
| anger | 920 | 0.983 |
| angry | 460 | 0.495 |
| boredom | 920 | 1.327 |
| bored | 460 | 0.611 |

How these were counted: the `Emotion` column of all 12,420 rows of `Data.xlsx`. `anger` and `angry` are two spellings of one emotion (1,380 together), as are `boredom` and `bored` (1,380 together), so each of the nine emotions has 1,380 recordings (6 actors x 230 script items). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- `Data.xlsx` contains errors: two spellings of an emotion (above), two mistyped actor IDs (`2_`, `6-`), and 42 `happy` recordings of actor `a4` that belong to actor `a5` (the grid of 6 actors x 9 emotions x 230 items is complete only after the repairs).
- Besides the discrete emotion, `Labels/` holds valence, arousal and dominance ratings by four raters.

### Listening test ratings in the release

`Labels/` holds valence, arousal and dominance ratings by four raters for the recordings (not counted here).


## Data, ratings and licence

The 12,420 recordings (about 15 hours; 0.75 to 21.8 seconds; mostly 44.1 kHz mono) are
the full grid of 6 actors, 9 emotions and 230 script items (words and sentences, with
Spanish translations in `Script.xlsx`). Besides the discrete emotion, `Labels/` holds
valence, arousal and dominance ratings by four raters; they are not used here. The
licence is CC BY 4.0 (figshare 20292516). Cite the Scientific Data paper of the corpus
(Nature Scientific Data 2022, [doi:10.1038/s41597-022-01855-9](https://www.nature.com/articles/s41597-022-01855-9)).

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

`QUECHUA_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
