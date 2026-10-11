# EMOVO speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EMOVO alone and scores it on EMOVO's own test split, classifying each
utterance into one of seven emotions.

EMOVO is an Italian emotional speech corpus
(Costantini et al., "EMOVO Corpus: an Italian Emotional Speech Database"):
six professional actors (three female `f1`-`f3`,
three male `m1`-`m3`) each read the same 14 sentences in seven emotions,
`angry`, `disgust`, `fear`, `happy`, `neutral`, `sad` and `surprise`: 588
utterances, 84 per emotion, 0.51 hours. Sentences 8-10 are nonsense sentences.

The label comes from the file name, `<emotion>-<speaker>-<sentence>.wav` (e.g.
`pau-m2-l3.wav`: fear, male 2, sentence `l3`), through the three-letter Italian
emotion code. The recordings are 48 kHz stereo from two microphones, so
`create_dataset` downmixes them to mono and resamples them to 16 kHz with
ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOVO` — the extracted `emovo.zip`, or where to download it to
- `EMOVO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `emovo.zip` (249 MB) from Google Drive, the file
linked from the corpus's own page (voice.fub.it). That site is offline now and
its page survives only in the Internet Archive. Neither the archive nor the
paper it contains states a license.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers and sentences are not held out, so all three
splits contain all six speakers.

| split | utterances |
|---|---|
| train | 413 |
| valid | 56 |
| test | 119 |

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
