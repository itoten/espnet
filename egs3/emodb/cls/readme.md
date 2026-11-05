# EmoDB speech emotion recognition recipe

Classifies each utterance into one of seven emotions (`angry`, `boredom`,
`disgust`, `fear`, `happy`, `neutral`, `sad`) with the `cls` system: a
fine-tuned [WavLM Large](https://github.com/microsoft/unilm/tree/master/wavlm)
encoder, attentive statistics pooling and a fully connected head.

This is the baseline of the [Interspeech 2025 Speech Emotion Recognition in
Naturalistic Conditions
challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.html).
Unlike the `esp2_cls` recipes, the encoder is not frozen: only its
convolutional feature extractor is, leaving 314 M of 319 M parameters trainable.
EmoDB is small, so this recipe exists to check the system runs end to end; the
results that matter come from MSP-Podcast and the multi-corpus setting.

[EmoDB 2.0](https://doi.org/10.5281/zenodo.17651657) is already distributed as
16 kHz mono WAV, so nothing is converted: `create_dataset` only writes one
manifest per split. The labels come from the corpus metadata, which ships as
plain CSV in [audformat](https://audeering.github.io/audformat/), so `audb` is
not a dependency.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMODB` — an existing corpus tree, or where to download it to
- `EMODB_OUTPUT` — where the manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against EmoDB 2.0 (Zenodo DOI 10.5281/zenodo.17651657; `emodb_2.0/` with the audformat tables, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 816 | 10 | 0.63 |

How these were counted: every row of `db.emotion.csv` (equal to the `.wav` files in `wav/`; 16 kHz mono 16-bit, 1.21 to 8.98 s); hours from the file lengths. The release also holds a second set of 816 recordings in `laryngo/` (laryngograph signals; `db.laryngo.*` tables), which are not counted here. Speakers are the 10 rows of `db.speaker.csv` (five women, five men; the tables give age, gender and language).

### Files per label

| label (the `emotion` column of `db.emotion.csv`) | samples | hours |
|---|---|---|
| anger | 137 | 0.101 |
| fear | 122 | 0.076 |
| sadness | 121 | 0.118 |
| happiness | 115 | 0.080 |
| boredom | 112 | 0.082 |
| disgust | 105 | 0.098 |
| neutral | 104 | 0.070 |

How these were counted: the `emotion` column of all 816 rows. Each row also has `emotion.confidence` and `emotion.naturalness` (listener scores). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- Version 2.0 adds 281 utterances (`db.emotion.categories.ambiguous.csv`) to the 535 of the earlier EmoDB releases: fewer than 80% of the raters agreed on the intended emotion for them (the added ones reach down to 10% agreement).
- The individual ratings of the 20 raters are not in the release; only the averaged `emotion.confidence` and `emotion.naturalness` remain.

### Listening test ratings in the release

`db.emotion.csv` gives for each recording the confidence (the share of listeners who heard the intended emotion) and the naturalness, averaged over the listening test of the original EmoDB (20 raters); the single raters' answers are not included.


## Splits

The corpus publishes a speaker-independent partition: 6 speakers for training
and 4 for evaluation, proposed in [Burkhardt et al.,
Interspeech 2025](https://www.isca-archive.org/interspeech_2025/burkhardt25_interspeech.html).
The recipe reads it from the corpus tables rather than repeating it, so it
cannot drift from the published split, and the test speakers are never touched.

No validation split is published, so `dataset/config.yaml` holds out one
training speaker. The default is speaker 13.

| split | speakers | utterances |
|---|---|---|
| train | 3, 8, 9, 10, 11 | 403 |
| valid | 13 | 81 |
| test | 12, 14, 15, 16 | 332 |

## License and citation

The Zenodo record of EmoDB 2.0 (Burkhardt, Paeschke, Kienast, Sendlmeier and Weiss,
published 2025-11-19) lists the licence CC BY 4.0, which this recipe follows. The
older EmoDB pages may state other terms; the 2.0 release is what is downloaded here.
Ten actors (five women, five men) acted ten German sentences (five short, five
longer) in seven emotions in an anechoic chamber. Cite the Zenodo record
[10.5281/zenodo.17651657](https://doi.org/10.5281/zenodo.17651657) and the original
EmoDB paper, [Burkhardt et al., Interspeech 2005](https://www.isca-archive.org/interspeech_2005/burkhardt05_interspeech.html).

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

Test scores with `ambiguous: all`, the default:

| WA | UA | macro F1 | mAP | AUC |
|---|---|---|---|---|
| 79.52 | 79.43 | 78.56 | 87.72 | 97.60 |

WA and UA sit 0.09 apart because every split is near uniform across the seven
emotions. Per class, `sad` (F1 98.9), `disgust` (96.3) and `fear` (83.1) are
solved; `happy` is not, at 34.3 — the model predicts it rarely and confuses it
with `angry`, whose recall is 98.3 against a precision of 62.0.

For reference, Table 1 of [Burkhardt et al.][paper] reports UAR for a support
vector machine over embeddings from a 12-layer wav2vec2-large-robust model
fine-tuned for dimensional emotion on MSP-Podcast:

| training data | UAR on gold test | UAR on all of test |
|---|---|---|
| gold standard | .89 | .83 |
| all | .95 | .90 |

The bottom right, .90, is the cell this recipe's default corresponds to. Two
differences matter before reading much into the 10-point gap: that model is a
much larger frontend used as a fixed feature extractor with a classical
classifier on top, where this recipe trains a Transformer encoder on 403
utterances, and it trains on all six speakers where this recipe holds one out
for validation.

[paper]: https://www.isca-archive.org/interspeech_2025/burkhardt25_interspeech.html

## Notes

### The ambiguous utterances, and what "EmoDB" names

Version 2.0 adds 281 utterances that fewer than 80 % of the raters agreed on,
which earlier releases withheld. This recipe is for version 2.0, so it uses
them by default. `ambiguous` in `dataset/config.yaml` changes that.

| `ambiguous` | train | valid | test | |
|---|---|---|---|---|
| `all` (default) | 403 | 81 | 332 | the whole of version 2.0 |
| `train` | 403 | 61 | 231 | training changes, scores stay on the gold standard |
| `none` | 243 | 61 | 231 | the corpus published EmoDB numbers describe |

**Only `none` is EmoDB as the literature uses the name.** Every label here is
the emotion the actor was asked to perform, and `emotion.confidence` records
how many listeners heard it that way. The 535 gold-standard utterances are
those at least 80 % agreed on, and that threshold is what the name has meant
since 2004. The added utterances reach down to 10 % agreement, so the default
both trains and scores on labels a majority of listeners rejected. Set
`ambiguous: none` before comparing against a published EmoDB number.

The default is the harder task and the more balanced one: every split sits near
uniform across the seven emotions, where the gold standard leaves training only
20 `disgust` utterances. Burkhardt et al. report the same choice as a 2×2 —
train on gold or on everything, score on gold or on everything — and find that
adding the ambiguous utterances to training helps on both test sets, `disgust`
most of all.

All ten actors read the same ten sentences, so lexical content carries no
information about the label, which makes this a cleaner test of paralinguistic
modelling than corpora whose scripts differ by emotion.
