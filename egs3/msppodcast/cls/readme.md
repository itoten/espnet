# MSP-Podcast speech emotion recognition (8 classes)

Trains `cls` on MSP-Podcast's Train partition alone (angry, contempt,
disgust, fear, happy, neutral, sad, surprise), validates and early-stops on
its Development partition, and scores the best checkpoint against MSP-Podcast's
own Test1/Test2 plus the same 5 out-of-domain corpora
`egs3/multicorpus_ood20/cls` uses (EmoV-DB, IEMOCAP, TESS, SAVEE,
RAVDESS, each read there in full).

MSP-Podcast's own standard is eight classes, so contempt is a class like any
other. `O` (free-text) and `X` (no annotator majority) are dropped: neither is
a hard emotion label to train on.

Compared with `multicorpus_ood20`, this recipe asks whether a single large,
naturalistic corpus (MSP-Podcast, ~137k training utterances) generalizes
better to unseen corpora than ten smaller, mostly acted corpora used together
(~55k utterances). It also differs in using real validation and early
stopping: MSP-Podcast is far larger than the other corpora, so a held-out
slice of it is a reasonable proxy for its own held-out data.

All outputs (manifests, token list, class weights, 16 kHz WAV files) go to
`$MSPPODCAST_OUTPUT`; the Train and Development partitions are used for
training and validation, Test1/Test2 only as extra test sets.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the MSP-Podcast corpus as obtained from the MSP lab under the signed agreement (`Audios/`, `Labels/Labels/labels_consensus.csv`, `Partitions.txt`, Release 2.0 of 2025-10-24 according to its `readme.txt`). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Request to the MSP lab and a signed agreement (internal non-commercial research only; no redistribution). The data cannot be downloaded without the agreement.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 267,905 | 3,714 (distinct `SpkrID` values) | 409.51 |

How these were counted: every `.wav` in `Audios/` (16 kHz mono in 263,225 files and 44.1 kHz mono in 4,680; 1.91 to 11.94 s); hours from the file lengths. `labels_consensus.csv` has 264,705 rows, so 3,200 audio files have no row in it. Speakers are the distinct `SpkrID` values of that table.

### Files per label

| `EmoClass` code in `labels_consensus.csv` | read as (corpus documentation) | rows |
|---|---|---|
| N | neutral | 78,718 |
| H | happy | 58,284 |
| X | no agreed label | 48,177 |
| A | angry | 35,860 |
| S | sad | 24,230 |
| U | surprise | 5,845 |
| C | contempt | 5,585 |
| O | other (free text) | 3,719 |
| D | disgust | 2,744 |
| F | fear | 1,543 |

How these were counted: the `EmoClass` column of `labels_consensus.csv` (the consensus of the crowd workers' votes). The table also holds arousal, valence and dominance scores and a `Split_Set` column: Train 169,190, Development 34,399, Test1 46,294, Test2 14,822 rows.

### Issues in the release

- 2,760 groups (8,180 files) of the audio are byte-identical (MD5); their content was not examined.
- 3,200 audio files have no row in `labels_consensus.csv`.


## Data, annotators and licence

MSP-Podcast (Lotfian and Busso, 2019) is natural English speech cut from podcasts, with
emotion annotated by crowd workers: the consensus class comes from their votes, and
arousal, valence and dominance scores are annotated as well (arXiv 2509.09791 reports
1,446,270 annotations by 13,280 annotators). The recipe uses only the consensus class. Access needs a signed agreement with the MSP
lab: use is limited to internal non-commercial research, the data must not be
redistributed, and Lotfian and Busso (2019) must be cited. The raw release holds 8,180
files in 2,760 groups of byte-identical audio; they have not been examined and are kept.

## Running

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
    --training_config conf/training.yaml
python run.py --stages train --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`dataset.test` (in both `conf/training.yaml` and `conf/inference.yaml`) reads
`multicorpus_ood20`'s already-built manifests via `data_src`, so both models
are scored on byte-identical evaluation data. It is read-only: running this
recipe never writes into `multicorpus_ood20`'s directory.

## Scoring

`conf/metrics.yaml` computes WA, UA and MacroF1, without `token_list`: the
evaluation corpora carry labels this model cannot predict (EmoV-DB's
`amused`/`sleepy`, IEMOCAP's `excited`, RAVDESS's `calm`), and passing
`token_list` would make UA/MacroF1 raise on those instead of scoring them as
always-wrong. mAP and AUC are not computed here since they require
`token_list` with no such fallback.
