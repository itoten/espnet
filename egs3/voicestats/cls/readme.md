# Voice Statistics corpus speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on this corpus alone and scores it on its own test split, classifying each
utterance into one of three emotions.

The [Voice Statistics corpus](https://voice-statistics.github.io/) (声優統計コーパス,
by the doujin circle 日本声優統計学会) is Japanese speech by three professional
female voice actresses (`fujitou`, `tsuchiya`, `uemura`). Each reads the same 100
phoneme-balanced sentences in three styles, `normal`, `happy` and `angry`: 900
recordings, 300 per style, 2.28 hours. Each is 3.1 to 21.4 seconds long. The site
describes the styles as three emotion patterns; `normal` (通常) is mapped to
`neutral`. There is no `disgust`, `fear`, `sad` or `surprise`.

The label comes from the file name, `<speaker>_<emotion>_<sentence>.wav` (e.g.
`fujitou_angry_033.wav`). Each file also sits in a folder named
`<speaker>_<emotion>`, and `create_dataset` fails if the two disagree. The
recordings are 48 kHz mono; `create_dataset` resamples them to 16 kHz.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `VOICESTATS` — the extracted archives, or where to download them to
- `VOICESTATS_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads nine tar.gz archives (one per speaker and style,
about 420 MB in total) from the authors' GitHub repository, pinned to its latest
commit. The site offers the audio free of charge mainly for personal research
and analysis and forbids redistribution. No checksums are published, so the
download is checked only by the number of recordings (900).

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 630 |
| valid | 90 |
| test | 180 |

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
