# Quechua Collao speech emotion recognition recipe

Classifies each recording into one of nine emotions (`anger`, `boredom`,
`calm`, `excited`, `fear`, `happy`, `neutral`, `sadness`, `sleepy`) with a
frozen [WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm)
frontend, a Transformer encoder and a linear head.

The [Quechua Collao corpus](https://doi.org/10.6084/m9.figshare.20292516) is
12,420 recordings by six actors, each performing 230 script items in all nine
emotions. The script mixes single words with sentences, so recordings run from
0.75 to 21.8 seconds. Most of the audio is 44.1 kHz mono, but seven files are
16 kHz and four are 48 kHz stereo, so `create_dataset` puts every one through
ffmpeg to 16 kHz mono.

The file names are serial numbers and carry nothing, so `Data/Data/Data.xlsx`
is the only link between a recording and its label. The builder reads that
workbook with the standard library — an xlsx file is a zip of XML — rather than
adding a spreadsheet dependency for one table.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `QUECHUA` — an existing corpus tree, or where to download it to
- `QUECHUA_OUTPUT` — where the resampled audio and manifests are written

The archive comes from figshare, which needs no account, and is checked against
the published MD5 before it is unpacked. Beyond the labels the corpus also
ships valence, arousal and dominance ratings from four annotators, and the
recording script with Spanish translations; the manifests use none of that.

## The metadata needs three repairs

**Two emotions are spelled two ways.** `anger`/`angry` and `boredom`/`bored`
name the same emotions and would otherwise become four classes for two. The
majority spelling wins. This is fixing a data-entry inconsistency, not renaming
a label.

**Two rows lost the `a` from their speaker id**, because the file names they
were derived from are malformed (`6-N093..wav`, `2_N 089.wav`). Both are
`neutral`, and a6 and a2 are each one `neutral` short without them, so where
they belong is not in doubt.

**42 `happy` recordings are filed under the wrong speaker.** Script items
H001–H042 appear twice under a4 and are missing from a5 entirely; the `Actor`
column agrees with the file name because it was derived from it, so both
columns are wrong together. Each of those 42 items has exactly two recordings,
one per speaker, so the question is only which of the two is a5's. They were
told apart by an LDA over WavLM layers trained on the two speakers' other eight
emotions, which

- is right on all 376 held-out `happy` recordings whose speaker is known,
- gives the same answer when the reference set is rebuilt without any one
  emotion, or halved,
- was confirmed by listening to all 42 pairs, and
- is corroborated by the `Audio` ids, which separate cleanly: everything
  assigned to a5 falls below 13633 and everything left with a4 above 13703.

**This is a reconstruction, not something the corpus states.** The 42 ids are
listed in `dataset/config.yaml`; empty `reassign_to_a5` to use the table as
published.

With all three applied the corpus becomes the even grid it was designed as:
six speakers × nine emotions × 230 script items, 2,070 recordings each.

## Splits

No partition is published — the corpus paper describes the recordings and a
listening test and defines no train/test division. Speakers are dealt out in
order, which is enough to balance every split across emotions once the repairs
above are in place.

| split | speakers | recordings |
|---|---|---|
| train | a1–a4 | 8280 |
| valid | a5 | 2070 |
| test | a6 | 2070 |

With six speakers, one each for validation and test, both numbers carry a lot
of speaker-specific variance. `dataset/config.yaml` carries a TODO: the
distribution ships no speaker demographics, so there is nothing here to balance
against either.

`remove_long_short` is set past both ends of the duration range and drops
nothing. An emotional speech corpus is small enough that losing even a handful
of recordings moves the numbers, so the filter is here to catch a corrupt
resample, not to reshape the data.

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

## Results

| WA | UA | macro F1 | mAP | AUC |
|---|---|---|---|---|
| 36.38 | 36.38 | 30.96 | 34.81 | 75.32 |

WA and UA are identical because every split holds the same number of recordings
per emotion. Nine classes make chance 11.1 %.

Four classes hold up — `anger` (F1 90.2), `calm` (74.9), `happy` (47.7) — while
`neutral` and `sadness` collapse to zero and `excited` and `fear` sit near it.
The low-arousal emotions run together: `boredom` draws far more predictions
than it should and absorbs much of `sleepy`, `sadness` and `calm`.

**Treat a single number here with suspicion.** Validation, on a different
single speaker, scores 52.6 — sixteen points above test. The training curve
plateaus by the fifth epoch and then oscillates, and two runs of this recipe
differing only in two training recordings landed 2.6 points apart. Nothing has
been tuned per corpus; the hyperparameters come from the MELD recipe unchanged,
and nine classes over largely single-word recordings is the hardest setting of
any recipe here.
