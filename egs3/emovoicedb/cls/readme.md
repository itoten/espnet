# EmoVoice-DB speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoVoice-DB alone and scores it on the corpus's own test split,
classifying each utterance into one of seven emotions.

[EmoVoice-DB](https://huggingface.co/datasets/yhaha/EmoVoice-DB) is an English
emotional speech dataset **synthesised entirely with GPT-4o-audio** from texts
and emotion descriptions written by GPT-4o; samples with a high word error rate
were filtered out (dataset card). It is not human speech. The lists name 22,100
recordings, 40.44 hours in total, 1.05 to 27.75 seconds long, all 24 kHz mono
16-bit wav (converted to 16 kHz here). The emotions are `angry`, `disgust`
(`disgusted` in the lists), `fear` (`fearful`), `happy`, `neutral`, `sad` and
`surprise` (`surprised`), roughly balanced (2,950 to 3,486 recordings each). The
card says five voices were used; the voice name is the last part of each key
(`ash`, `ballad`, `coral`, `sage`, `verse`) and every voice appears in every split.

The label is the `emotion` field of the official lists. Each recording also has a
free-text emotion description, which this recipe does not use.

## Provenance and licence

The files are downloaded from the Hugging Face dataset `yhaha/EmoVoice-DB` (no
login; not gated), pinned to commit `e0e61055fc5c2b52bc20d8b45f8ce3037228a61d`
and checked against the SHA-256 digests in `dataset/config.yaml`: the three
split lists (`train.jsonl`, `val.jsonl`, `test.jsonl`) and the seven
`audio/<emotion>.zip` archives, 5.8 GB in total. The card states the MIT licence.
The audio was generated with OpenAI's GPT-4o-audio, whose terms of use may apply
to the generated speech; this was not checked. Cite Yang et al., "EmoVoice:
LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting",
arXiv:2504.12867.

The archives hold 24,923 recordings, 2,823 more than the lists name (for example
414 more `angry` than listed). Only the listed ones are used; why the others are
in the archives is not documented.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOVOICEDB` — the directory holding the three lists and the extracted `<emotion>/` folders, or where to download them to
- `EMOVOICEDB_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the lists and archives, checks them and extracts the
wav files. Fetching the 5.8 GB with `curl` took 11 min 33 s, and extracting the
archives took about 50 s. `create_dataset`, `remove_long_short` and
`prepare_labels` together took 3 min 1 s on one run (8 conversion threads; the
download was already done).

## Splits

The official lists are used as they are. `train.jsonl` has three rows per
recording (one for each of three paraphrased emotion descriptions), 63,150 rows
for 21,050 recordings; rows are de-duplicated by key. No recording is in more than
one split. The five voices are shared by all splits, so scores are not
voice-independent. The test and validation sets have the same count in every class.

| split | utterances | hours | after `remove_long_short` |
|---|---|---|---|
| train | 21,050 | 38.50 | 21,050 |
| valid | 350 | 0.64 | 350 |
| test | 700 | 1.30 | 700 (not filtered) |

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
