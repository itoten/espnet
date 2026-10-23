# CASIA speech emotion recognition (single-corpus baseline)

> **Warning: do not use this recipe's data source.** The audio it downloads is an
> unofficial copy of a corpus that is licensed for a fee, and no permission to
> redistribute it was found. The recipe is kept for reference only. Use it only
> with a copy you have licensed yourself.

Summary of the problem (checked on 2026-10-23):

- The CASIA Chinese emotional speech corpus (CASIA汉语情感语料库, catalogue number
  CLDC-SPC-2005-010) is listed by the Chinese Linguistic Data Consortium
  (ChineseLDC) as developed by the Institute of Automation, Chinese Academy of
  Sciences, and licensed for a fee. Its resource page (archived 2020-05-01)
  gives 4 speakers, 6 emotions and 9,600 utterances. No document from the
  corpus authors themselves was found, and the current terms could not be
  checked (the site was not reachable).
- The widely used 1,200-utterance version, linked from the EmoBox benchmark, is a
  re-upload by an individual user to Baidu AI Studio (dataset 209512, 2023-04-16).
  The Hugging Face mirror used here appears to be the same files: the 20 files
  listed on that page match in size and modification time.
- Papers often cite "Design of Speech Corpus for Mandarin Text to Speech"
  (Tao et al., Blizzard Challenge 2008) for this corpus. That paper describes a
  different corpus, recorded by one female speaker for speech synthesis, and does
  not mention emotion.

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on CASIA alone and scores it on CASIA's own test split, classifying each
utterance into one of six emotions.

CASIA is a Mandarin Chinese acted corpus from the Institute of Automation,
Chinese Academy of Sciences. The copy used here has 4 speakers (`liuchanhg`,
`ZhaoZuoxiang`, `zhaoquanyin`, `wangzhe`) reading 50 sentences in six emotions:
`angry`, `fear`, `happy`, `neutral`, `sad` and `surprise`. That gives 1,200
recordings, 200 per emotion (50 per speaker), 0.64
hours in total, 0.96 to 4.93 seconds long. They are 16 kHz mono as distributed.

The label is the emotion in the file name `<sentence>-<emotion>-<speaker>.wav`;
`dataset/config.yaml` maps it to the manifest label. The sentence texts are shared
across speakers and emotions.

## Provenance

See the warning above. The recipe uses an
**unofficial Hugging Face mirror** (`Bill13579/casia-mirror`, pinned to commit
`a6699530434a32a5ce8f49e037016ba58e934abd`, `CASIA.zip`, SHA-256 checked). The
mirror's licence field is `other` and states no terms; check the terms of the
original corpus before redistributing or using it commercially.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CASIA` — the directory holding the extracted `CASIA/` wav files, or where to download them to
- `CASIA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `CASIA.zip` from the mirror above, checks its SHA-256
and extracts the wav files. `create_dataset`, `remove_long_short` and
`prepare_labels` together took about 77 seconds on one run (the archive was
already downloaded; its download time was not measured).

## Splits

The corpus publishes no partition. Because there are only four speakers, whole
speakers are held out (`n_test_speakers`, `n_valid_speakers`, `split_seed` in
`dataset/config.yaml`): one speaker for test, one for valid and two for training.
Scores are therefore speaker-independent, but the sentences are shared with the
training set. Every class has the same count in each split.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 600 | 600 |
| valid | 300 | 300 |
| test | 300 | 300 (not filtered) |

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
