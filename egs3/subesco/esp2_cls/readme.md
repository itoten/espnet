# SUBESCO speech emotion recognition recipe

Classifies each utterance into one of seven emotions (`angry`, `disgust`,
`fear`, `happy`, `neutral`, `sad`, `surprise`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[SUBESCO](https://doi.org/10.5281/zenodo.4526477) is 7,000 Bangla utterances:
20 professional actors, 10 male and 10 female, each performing ten sentences in
seven emotions with five takes apiece. It was recorded in a sound studio, with
trained assistants watching from an adjacent control room. The audio is 48 kHz
32-bit mono, so `create_dataset` resamples it to 16 kHz 16-bit with ffmpeg.

Every field comes from the file name — no annotation file is read:

```
F_01_OISHI_S_1_ANGRY_1.wav
│  │    │    │ │   │     └ take
│  │    │    └─┴ sentence
│  │    └ speaker name
│  └ speaker number
└ gender
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SUBESCO` — an existing corpus tree, or where to download it to
- `SUBESCO_OUTPUT` — where the resampled audio and manifests are written

The archive comes from Zenodo, which needs no account, and is checked against
the MD5 the record publishes before it is unpacked.

## Splits

The corpus publishes no partition: its paper describes the recordings and a
listening test and defines no train/test division. Speakers are dealt out in
numeric order, **the same number of each gender**, so the split is balanced by
construction and reproducible from the corpus alone.
`dataset/config.yaml` holds the per-gender counts.

| split | speakers | utterances |
|---|---|---|
| train | 6 F + 6 M | 4200 |
| valid | 2 F + 2 M | 1400 |
| test | 2 F + 2 M | 1400 |

Every split holds the same number of utterances per emotion, which is why WA
and UA come out identical below.

Two things the builder has to get right. **Speaker numbers restart within each
gender**, so `F_01` and `M_01` are different people and the speaker id joins the
two — keying on the number alone would put one "speaker" in two splits. And
each speaker recorded **five takes of every sentence-emotion pairing**, so the
split has to work on whole speakers: splitting utterances would scatter
near-identical takes across train and test.

One file is named `F_02_MONIKA_S_2_SURPRISE_3].wav`. The builder's pattern
tolerates the stray bracket.

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
| 58.43 | 58.43 | 54.43 | 63.73 | 89.82 |

**This is the one corpus here where the model lands well below its listeners.**
The corpus paper reports 71 % raw accuracy from 50 raters; this recipe reaches
58.4 %. WavLM is pretrained overwhelmingly on English, and nothing in this
recipe is tuned per corpus — the MELD hyperparameters are reused unchanged —
so the gap is a reasonable place to look for how much a frontend's pretraining
language matters.

`disgust` and `fear` carry the loss, at 17.0 % recall each against 43 % and
63 % on validation. The model is not confusing them so much as declining to
predict them: `fear` draws 43 predictions against 200 true instances, at 79 %
precision, while `surprise` draws 326.

Per speaker, the same emotion is not the same problem:

| | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---|---|---|---|---|---|---|
| F_09 | 4.0 | 12.0 | 28.0 | 62.0 | 94.0 | 84.0 | 88.0 |
| F_10 | 56.0 | 42.0 | 2.0 | 86.0 | 98.0 | 54.0 | 70.0 |
| M_09 | 100.0 | 4.0 | 16.0 | 80.0 | 76.0 | 46.0 | 96.0 |
| M_10 | 98.0 | 10.0 | 22.0 | 76.0 | 72.0 | 74.0 | 86.0 |

`angry` swings from 4 % to 100 % across four speakers. Twenty professional
actors do not converge on one way of performing an emotion, and with four
speakers per split the test score inherits that variance: validation scores
9.7 points higher on an identically built partition. Read a single number here
with that in mind — a leave-speakers-out cross-validation would say more than
this one split does.
