# Dusha speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Dusha alone and scores it on Dusha's own test split, classifying each
recording into one of four emotions.

[Dusha](https://github.com/salute-developers/golos/tree/master/dusha) (Kondratenko et
al., arXiv:2212.12266) is Russian speech, annotated on a
crowd-sourcing platform. The release has a crowd
subset (`crowd.tar`) and a podcast subset. The paper's abstract calls the two
subsets acted and real-life (podcasts), so the crowd subset is presumably the
acted one. Only the crowd subset is used. The labels are `angry`, `sad`,
`neutral` and `happy` (the release calls happiness `positive`; it is renamed to
match the other recipes). There is no
`disgust`, `fear` or `surprise`.

## What is used

The labels are the release's own aggregated labels, read from the two files in its
`paper_setups.tgz` (`crowd_large.jsonl` for train, `crowd_test.jsonl` for test).
How the paper aggregates the annotations, and with what agreement threshold, was
not checked. These files list 146,715 train and 13,651 test utterances (160,366,
202.5 hours); the rest of `crowd.tar` (utterances that are in neither file) is not
extracted. Durations range from 0.86 to 29.88 seconds (median 4.34), so
`remove_long_short` drops nothing. Every recording is already 16 kHz mono 16-bit
WAV, so `build()` references the files where they lie and converts nothing. A file
in any other format would be converted into `$DUSHA_OUTPUT/wav/`.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `DUSHA` — the directory holding the extracted `crowd_train/`, `crowd_test/` and `paper_setups/`, or where to download it to
- `DUSHA_OUTPUT` — where the manifests are written

## Download

`prepare_source` downloads `paper_setups.tgz` (79 MB) and `crowd.tar` (29.97 GB,
about 28.6 GiB) from the release's CDN, checks each by byte count (no checksum is
published), and extracts only the listed WAVs and the two annotation files
(`raw_crowd_train.tsv`, `raw_crowd_test.tsv`). The archive is deleted afterwards
(`remove_archive`). Make sure the disk has about 30 GB free during the download and
23 GB afterwards.

The release is under the "Public license with attribution and conditions
reserved". It reworks CC BY-SA 4.0 and is not a Creative Commons license. It has no
non-commercial clause; it requires attribution and an indication of changes, and
requires adapted material to be shared under the same license elements or a
BY-SA-compatible license. Please read the license text in the repository
before redistributing anything derived from it, and cite Kondratenko, Sokolov, Karpov,
Kutuzov, Savushkin and Minkin, "Large Raw Emotional Dataset with Aggregation
Mechanism" (arXiv:2212.12266).

## Splits

The release's train/test partition is kept. The validation set is carved out of
train: whole `source_id` groups (the last column of the annotation files; what it
identifies, probably the speaker, is not confirmed) are moved to valid until 10% of the train
utterances are reached (`valid_ratio`, `split_seed` in `dataset/config.yaml`), so valid shares no
`source_id` with train. Utterances without a `source_id` (1,517 of the 146,715
listed train utterances; none in test) are treated as groups of one. The listed
train utterances carry 1,870 distinct `source_id`s, test carries 190, and the two
sets share none.

| split | utterances | after `remove_long_short` | angry | happy | neutral | sad |
|---|---|---|---|---|---|---|
| train | 132006 | 132006 | 12790 | 12558 | 87223 | 19435 |
| valid | 14709 | 14709 | 1597 | 1619 | 9894 | 1599 |
| test | 13651 | (not filtered) | 1332 | 1215 | 8948 | 2156 |

`neutral` is 66% of train, so the class weights (written by `prepare_labels`)
range from 0.378 (neutral) to 2.63 (happy). Because the data are unbalanced, report
Macro-F1 rather than accuracy.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`, 16 threads). The
download depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of 30 GB, size check, extraction of 160,366 WAVs) | 23 min 43 s |
| `create_dataset` (checks the format of each file; nothing is converted) | 32.6 s |
| `remove_long_short` | 22.7 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 3 s (includes starting `pixi` and Python) |

The download and the extraction were not timed separately in the recipe run. A
separate manual `curl` of `crowd.tar` took 24 min 42 s, and a plain `tar xf` of the
whole archive took 31 s. The extracted audio takes 23 GB and the manifests 50 MB.

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
