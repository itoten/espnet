# SyntAct speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SyntAct alone and scores it on SyntAct's own test split, classifying each
utterance into one of six emotions.

[SyntAct](https://zenodo.org/records/6573016) is a German synthetic corpus
([Burkhardt et al., 2022](https://aclanthology.org/2022.dclrl-1.1.pdf)): six MBROLA
diphone voices (`de1`-`de4`, `de6`, `de7`) speak sentences with prosody modified by
rules to sound `angry`, `boredom`, `fear`, `happy`, `neutral` or `sad`. The speech is
not human. The release has 6,001 wav files, all 16 kHz mono, 0.82 to 30.51 seconds
long, 11.15 hours in total. The label is the emotion the synthesizer was asked to
produce, taken from the official `db.emotion.categories.*.desired.csv` tables.

Thirty `neutral` files are byte-identical to the `sad` file with the preceding
number (same voice, all in the official train set). The cause is not known. Only
the `sad` file of each pair is kept; the thirty `neutral` files are listed in
`exclude_files` in `dataset/config.yaml`, and `create_dataset` stops if identical
files are left. The recipe therefore uses 5,971 of the 6,001 files.

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `SYNTACT` — the directory holding the extracted `syntact_cat/`, or where to download it to
- `SYNTACT_OUTPUT` — where the copied audio and manifests are written

## Download

`prepare_source` downloads `syntact_cat.zip` (987 MB, md5-checked) from
[Zenodo](https://zenodo.org/records/6573016) and extracts the wav and csv files. The
download time was not measured. With the archive already extracted, `create_dataset`,
`remove_long_short` and `prepare_labels` together took 5 minutes 6 seconds.

## Splits

The official split is voice-disjoint: `de1`-`de4` train and `de6`/`de7` test. No
validation set is published, so voice `de3` moves from train to valid.
Test voices are never seen in training; the valid voice is not seen either.

The per-class columns count the utterances left after `remove_long_short` (the test
split is not filtered).

| split | utterances | hours | after `remove_long_short` | angry | boredom | fear | happy | neutral | sad |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,723 | 4.97 | 2,722 | 365 | 532 | 408 | 509 | 487 | 421 |
| valid | 867 | 1.55 | 867 | 94 | 170 | 133 | 170 | 168 | 132 |
| test | 2,381 | 4.54 | 2,381 (not filtered) | 394 | 410 | 388 | 396 | 408 | 385 |

`remove_long_short` drops the one train clip longer than 30 s.

## License and citation

The Zenodo record states CC BY 4.0. Please cite Burkhardt, Eyben and Schuller (2022), the
DCLRL workshop paper at LREC 2022 linked above.

## Listening test

The label is the emotion specified for the synthesis, not a human or classifier judgement. The paper reports a separate forced-choice listening experiment with 20 people (accuracy 0.35 to 0.6); its results are not in the release.

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
