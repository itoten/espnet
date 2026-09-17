# MSP-Podcast speech emotion recognition recipe

Classifies each speaking turn into one of eight emotions (`angry`, `contempt`,
`disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[MSP-Podcast](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
release 2.0 is 267,905 speaking turns and 409 hours of English taken from
podcasts — natural speech, not acted, and by a wide margin the largest corpus
here. Every turn carries a categorical label and arousal/valence/dominance
ratings from at least five annotators, plus a speaker id, gender and a human
transcript.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MSPPODCAST` — an existing corpus tree
- `MSPPODCAST_OUTPUT` — where the resampled audio and manifests are written
  (about 47 GB)

## The corpus has to be placed by hand

It is released under a licence agreement with UT Dallas, so this recipe never
downloads it. Request it from
[the lab's page](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
and point `MSPPODCAST` at a directory holding `Audios/MSP-PODCAST_*.wav` and
`Labels/Labels/labels_consensus.csv`; `create_dataset` says as much if it
cannot find them.

`labels_consensus.csv` carries everything the manifests need — label, speaker,
gender, attributes and the published partition — so it is the only table read.
Nearly all of the audio is already 16 kHz mono, but not quite all, so every
recording goes through ffmpeg.

One row names a recording the release does not ship
(`MSP-PODCAST_1909_1017.wav`: podcast 1909 has 650 rows and 649 files). It is
skipped with a warning, and `max_missing_audio` in `dataset/config.yaml` caps
how many such gaps are tolerated, so a copy that is genuinely incomplete still
fails.

## Classes and splits

The eight categories are those of the
[Interspeech 2025 SER challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.html),
which drops `O` (the annotator wrote their own class) and `X` (no plurality
winner). Those are states of the annotation rather than emotions, and they are
a fifth of the corpus — 51,896 of 264,705 labelled turns.

The partition comes from the `Split_Set` column.

| split | source | turns |
|---|---|---|
| train | Train | 137,168 |
| valid | Development | 27,205 |
| test | Test1 | 36,768 |
| test2 | Test2 | 11,667 |

**Test1 and Test2 are both scored**, in one `measure` run, because they are
skewed differently: Test2 is 58 % `neutral` against Test1's 34 %, and the same
model scores eight macro-F1 points apart on them.

Test3 is not usable here. Its labels are withheld for the challenge's
[submission interface](https://lab-msp.com/MSP-Podcast_Competition/SERB/), so
it appears in neither the table nor the manifests. Results on it can only be
obtained by submitting predictions.

The corpus readme says train and development speakers do not overlap with the
test sets. Counted from the table, `train ∩ Test1` is empty, but
`train ∩ Test2` shares two speakers (37 turns on the train side) and
`Development` shares one speaker with Test1 and three with Test2. Separately,
44,436 training turns have `SpkrID = Unknown`, which the readme notes may
belong to test speakers.

## Quick start

```bash
# 1) Write the manifests (about an hour: 212,808 files through ffmpeg)
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train (about five hours on one A6000)
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

| split | WA | UA | macro F1 | mAP | AUC |
|---|---|---|---|---|---|
| test (Test1) | 61.22 | 32.87 | 32.81 | 36.54 | 81.28 |
| test2 (Test2) | 58.82 | 25.90 | 24.41 | 25.69 | 72.31 |

Macro F1 is the figure to compare: it is what the challenge and the corpus
paper report.

| | release | audio only | Test1 | Test2 |
|---|---|---|---|---|
| corpus paper, WavLM [1] | 2.0 | yes | .297 | .206 |
| corpus paper, HuBERT [1] | 2.0 | yes | .285 | .192 |
| **this recipe** | **2.0** | **yes** | **.328** | **.244** |
| entropy-aware curriculum [2] | 2.0 | yes | **.348** | **.318** |

[1] [The MSP-Podcast Corpus](https://arxiv.org/abs/2509.09791) — fine-tunes
WavLM-large (310M) with focal loss and a two-layer head.
[2] [Learning from Annotation Uncertainty](https://arxiv.org/abs/2606.27536),
Interspeech 2026 — fine-tunes WavLM with distribution-valued targets and an
entropy-based curriculum.

This recipe sits above the published baselines and below the best audio-only
result found, with a frozen frontend a third the size and no handling of class
imbalance at all. Do not read the ordering as a claim about method: only arXiv
was searched, and the papers reporting on release 2.0 with these splits are few.

Multimodal systems score higher still, but on Test3 and with text: Crab reports
macro F1 .431 there, and the 2025 challenge's leading entries .41–.43. Those
are not comparable with the table above.

## Where the score comes from

Class frequency and recall track each other almost exactly on Test1:

| | turns | recall | precision | F1 |
|---|---|---|---|---|
| happy | 10,947 | 78.4 | 61.5 | 68.9 |
| angry | 6,985 | 67.2 | 66.6 | 66.9 |
| neutral | 12,457 | 62.8 | 63.9 | 63.3 |
| sad | 3,041 | 42.0 | 40.9 | 41.4 |
| surprise | 1,206 | 7.0 | 40.1 | 12.0 |
| disgust | 744 | 3.8 | 26.7 | 6.6 |
| contempt | 1,040 | 1.2 | 19.7 | 2.2 |
| fear | 348 | 0.6 | 40.0 | 1.1 |

`fear` is right on 2 of its 348 turns, at 40 % precision: the model has learned
to almost never predict it rather than to predict it badly. That is the whole
gap between WA 61.22 and UA 32.87, and it is what the published systems address
and this recipe does not — the loss here is unweighted cross-entropy, where the
corpus paper uses focal loss and the entropy-aware work reweights by annotator
agreement. Validation accuracy plateaus by the fourth epoch, which is the model
settling into predicting the frequent classes.
