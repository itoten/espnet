# Emozionalmente speech emotion recognition recipe

Classifies each utterance into one of seven emotions (`angry`, `disgust`,
`fear`, `joy`, `neutral`, `sad`, `surprise`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[Emozionalmente](https://doi.org/10.5281/zenodo.12616095) is 6,902 Italian
utterances crowdsourced from 431 non-professional actors, each reading from the
same 18 sentences. It is already 16 kHz mono WAV, so `create_dataset` converts
nothing and only writes the manifests. Labels come from the metadata: the file
names are recording timestamps and carry no information. Only the word form is
normalised -- `anger` to `angry`, `sadness` to `sad` -- so no label is replaced
by a different word; `joy` stays `joy` rather than becoming `happy`, because
deciding that two corpora mean the same thing belongs to cross-corpus work.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOZIONALMENTE` — an existing corpus tree, or where to download it to
- `EMOZIONALMENTE_OUTPUT` — where the manifests are written

The archive is fetched from Zenodo, which needs no account, and checked against
the MD5 the record publishes before it is unpacked.

## Splits

**The corpus publishes a speaker-independent train/dev/test split**, stratified
by emotion, gender and age, and the recipe reads it from
`metadata/split/*.csv` rather than rebuilding it. Nothing about the partition
is decided here. It is the split the paper evaluates on — "these splits are
shared as part of the dataset for reproducibility" — so the numbers below and
the paper's are measured on the same test set.

| split | speakers | utterances |
|---|---|---|
| train | 274 | 4366 |
| valid | 69 | 1202 |
| test | 88 | 1334 |

No speaker appears in two splits. The stratification holds up: every emotion
sits between 13.3 % and 15.9 % of each split against a uniform 14.3 %, the
gender ratio is 69.7/69.6/68.2 % female, and the median speaker age is 27/26/28.

Utterances per speaker are very uneven — 44 speakers contribute one recording
and the busiest contributes 105 — but the official split spreads that evenly,
leaving a median of 7, 8 and 8. The paper names this imbalance as a limitation:
not every actor performed every sentence-emotion pairing.

## What the corpus ships that the manifest does not use

| file | contents |
|---|---|
| `metadata/samples.csv` | the Italian sentence for each recording |
| `metadata/users.csv` | gender, age and mother tongue for all 938 participants |
| `metadata/evaluations.csv` | five listener judgements per recording |

`evaluations.csv` is the interesting one. **829 evaluators rated every
recording five times**, giving both a perceived emotion and an audio-quality
verdict, and the coverage is exact: 6,902 recordings, five ratings each, none
missing.

**Listeners reach 66.4 % UAR against the intended labels** — recomputed here
from the table, matching the 66 % the corpus reports. They are weakest on
`disgust` (52.4 %) and `fear` (54.0 %) and strongest on `neutrality` (83.9 %).
That is a useful ceiling to keep in mind: the labels record what the actor
meant, not what listeners hear, and a third of the corpus reads as something
else to the average listener.

The audio-quality flag is not worth filtering on. Only 3.8 % of judgements say
`bad`, they are spread evenly across emotions, and **no recording collects even
three `bad` verdicts out of five**.

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
| 73.16 | 73.03 | 73.08 | 81.05 | 95.10 |

No class collapses: per-class F1 runs from 65.7 (`fear`) to 82.4 (`neutral`).
The model over-predicts `angry` and `sad` and under-predicts `disgust` and
`fear` — **the same two classes listeners find hardest**, which suggests the
errors follow the ambiguity in the labels rather than a modelling failure.

Two reference points, both on this same test set. Listeners score 66.4 % UAR
on the same labels. The corpus authors report 82.45 % UAR, so the 73.0 % here
sits between the two. The gap to the paper is in the model, not the data: they
fine-tune wav2vec 2.0 end to end and tune the learning rate with Optuna against
the dev set, where this recipe freezes WavLM, trains a Transformer encoder on
top, and reuses the MELD recipe's hyperparameters unchanged.

## Notes

The 18 sentences are everyday statements with no emotional content, and all 18
are used for all seven emotions, so the words say nothing about the label. The
paper reports they were written ad hoc to cover every Italian phoneme, taking
the approach from English and Portuguese sentence sets, and checked by expert
review.

Three of the 18 — *Gli operai si alzano presto*, *La cascata fa molto rumore*
and *Vorrei il numero telefonico del Signor Piatti* — are word-for-word the
first, third and seventh sentences of
[EMOVO](http://www.lrec-conf.org/proceedings/lrec2014/pdf/591_Paper.pdf), the
other acted Italian corpus, which the paper does not mention. They account for
1,166 recordings, 16.9 % of the corpus. **Anyone pairing these two corpora for
cross-corpus work should know their text is not independent.**
