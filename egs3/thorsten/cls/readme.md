# Thorsten-Emotional speech emotion recognition recipe

Classifies each recording into one of eight labels (`amused`, `angry`,
`disgusted`, `drunk`, `neutral`, `sleepy`, `surprised`, `whisper`) with a
frozen [WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm)
frontend, a Transformer encoder and a linear head.

[Thorsten-Emotional](https://www.openslr.org/110/) is one German speaker
reading the same 300 sentences eight times over, once per label. It was
recorded and released for text-to-speech, not for emotion recognition, and the
difference shows — read the results section before using the numbers. The audio
is 22.05 kHz mono, normalised to -24 dB with leading and trailing silence
removed, so `create_dataset` resamples it to 16 kHz.

Labels come from the directory names and are used exactly as the corpus writes
them. Three of the eight are speaking styles rather than emotions, and the
corpus is candid about one of them:

> "drunk", recorded sober without being drunk, just pronounced it that way :-)

Renaming or dropping those would be this recipe deciding what the corpus means.
Merging labels across corpora is a cross-corpus decision, not a per-recipe one.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `THORSTEN` — an existing corpus tree, or where to download it to
- `THORSTEN_OUTPUT` — where the resampled audio and manifests are written

## Download and licence

The archive comes from OpenSLR 110, which needs no account. The same corpus is
on [Zenodo](https://doi.org/10.5281/zenodo.5525023), and only Zenodo publishes
a checksum — the two files are byte-identical, so `prepare_source` verifies the
OpenSLR download against the Zenodo digest.

The licence is **CC0**, stated by OpenSLR, by the Zenodo description, and by the
[project's repository](https://github.com/thorstenMueller/Thorsten-Voice).
Zenodo's own licence *field* says CC-BY-4.0, which contradicts the description
on the same page and looks like a slip. There is no paper; the dataset is cited
by its Zenodo DOI.

## The corpus is 2,399 recordings, not 2,400

Every description of this corpus says 300 sentences × 8 labels = 2,400. The
archive holds 2,399: `whisper` is one short.

```
2cc2cc4a34b961ef1657cc82dbd18875
  "Herr Kapitän, ich stehe Ihnen mit all meinen Kräften zu Diensten!"
```

That sentence is present under the other seven labels. No other distribution
has it either — Zenodo ships the identical archive, there is no earlier
version, and the official 44 kHz release on Hugging Face
(`Thorsten-Voice/TV-44kHz-Full`) is a different, smaller curation at 2,020
recordings. `create_dataset` names the gap in its log rather than quietly
rounding it away.

## Splits

One speaker, so a speaker-independent split cannot be built and **scores from
this recipe are speaker-dependent**.

The split works on sentences instead: all eight recordings of a sentence go to
the same split. Splitting recordings would put the same German text in both
training and test, and the model would be scored on sentences it had already
heard, just spoken differently. Sentences are sorted by hash before a seeded
shuffle, so the partition depends on the corpus and `split_seed` alone.

| split | sentences | recordings |
|---|---|---|
| train | 180 | 1440 |
| valid | 60 | 479 |
| test | 60 | 480 |

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
| 99.79 | 99.79 | 99.79 | 99.93 | 99.99 |

One of the 480 test recordings is wrong, a `disgusted` read as `drunk`.

**This is not a hard emotion-recognition task, and the number should not be
compared with one.** Three things stack up. There is one speaker, so nothing
has to generalise across voices. Three of the eight labels — `whisper`,
`sleepy`, `drunk` — are speaking styles whose acoustics differ before prosody
enters into it; whispering drops voicing outright. And the recordings are
single-session, single-microphone, normalised to a fixed level with the silence
trimmed, because they were prepared for TTS training.

What 99.79 measures is whether one person's eight deliberate manners of
speaking can be told apart. That is a real question, and the answer is yes, but
it is not the question CREMA-D or MELD ask.

For cross-corpus work the corpus is awkward in both directions: it carries three
labels no other corpus here has, and a model trained on it has heard exactly one
voice.
