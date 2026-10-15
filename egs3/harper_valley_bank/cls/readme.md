# Harper Valley Bank speech sentiment recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Harper Valley Bank alone and scores it on Harper Valley Bank's own test split,
classifying each utterance into one of three polarity classes: `negative`, `neutral`
and `positive`.

[Harper Valley Bank](https://github.com/cricketclub/gridspace-stanford-harper-valley)
(Gridspace and Stanford) is a corpus of 1,446 simulated English contact-center calls
between a bank agent and a caller, with 59 speakers across the agent and caller
roles. Each call is stored as two 8 kHz mono wav files, one per side. The
transcript of a call is a list of segments with a human transcript, a dialog act
and a softmax over `negative`, `neutral` and `positive`. This recipe makes one
utterance per segment: 25,730 segments, 12.0 hours in total, 0.03 to 18.7
seconds long (the 47.3 hours of audio in the repository count both sides of each
call in full).

The label is the arg-max of the segment's softmax. The README of the corpus
describes it as the output of Gridspace's emotion model ("emotional valence"), so
the labels are model outputs, not human judgements, and they measure polarity,
not discrete emotions. Over the 25,730 segments: 17,149 neutral, 8,148 positive,
433 negative.

`start_ms` of a segment is its position in the per-side wav. `offset_ms`, which the
corpus README describes as the offset from the start of the recording, does not
match the audio (measured by scanning the energy of the audio around both
fields); this recipe uses `start_ms`. Thirteen segments start after the end of
their recording and are dropped. `create_dataset` cuts each segment out of its
call and converts it to 16 kHz mono with ffmpeg. About a quarter of the
segments are shorter than 0.5 s (6,626 of 25,717), so `remove_long_short` removes
many train and valid utterances.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `HARPER_VALLEY_BANK` — the directory holding the extracted repository (`data/audio`, `data/transcript`), or where to download it to
- `HARPER_VALLEY_BANK_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the repository tarball (676 MB compressed, 2.6 GB
extracted) from GitHub without a login and extracts it. Measured on one run:
about 3 minutes to download; `create_dataset`, `remove_long_short` and
`prepare_labels` together took 3 minutes 5 seconds (8 ffmpeg workers).

## Splits

The corpus publishes no partition. Whole calls are assigned at random
(`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`),
so no call spans two splits. Speakers are not held out: the same agents and
callers appear in several calls and therefore in several splits.

| split | utterances | after `remove_long_short` | negative / neutral / positive (filtered) |
|---|---|---|---|
| train | 17,973 | 13,346 | 282 / 7,487 / 5,577 |
| valid | 2,583 | 1,909 | 54 / 1,084 / 771 |
| test | 5,161 | 5,161 (not filtered) | 96 / 3,415 / 1,650 (unfiltered) |

The label is skewed to neutral and `negative` is rare, so `prepare_labels` writes class
weights.

## License and citation

The repository's `LICENSE` file is the Creative Commons Attribution 4.0
International license. Cite: Mike Wu, Jonathan Nafziger, Anthony Scodary and
Andrew Maas, "HarperValleyBank: A Domain-Specific Spoken Dialog Corpus",
[arXiv:2010.13929](https://arxiv.org/abs/2010.13929).

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
