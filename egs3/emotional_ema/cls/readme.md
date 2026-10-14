# Emotional EMA speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Emotional EMA alone and scores it on Emotional EMA's own test split, classifying each
utterance into one of five emotions.

[Emotional EMA](https://zenodo.org/records/19303746) is a Brazilian Portuguese
acted corpus recorded with electromagnetic articulography: 3 speakers (`jn`, `jr`,
`kf`) read sentences and passages in five emotions, `angry`, `fear`, `happy`,
`neutral` and `sad`. The wav files of the `EMA_5EMO_NSF` release number 953
(262 neutral, 178 happy, 171 each of angry, fear and sad), 1.63 hours in total,
1.4 to 63 seconds long. Some neutral clips are fast-rate variants or repeats
(`_rep1`).

The label is the intended emotion, taken from the file name
`ema_5emo_<speaker>_<emotion>_<rate>_<prompt>_utt<N>.wav`; `dataset/config.yaml` maps
`ang`, `fea`, `hap`, `neu` and `sad` to labels. The release also holds
listener-evaluation files, which this recipe does not use. Three zero-content
dummy files do not match the file name pattern and are skipped. The recordings
are 22.05 kHz mono, so `create_dataset` converts them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMOTIONAL_EMA` — the directory holding the extracted `EMA_5EMO_NSF/` wav files, or where to download them to
- `EMOTIONAL_EMA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `EMA_5EMO_NSF.zip` (548 MB, md5-checked) from
[Zenodo](https://zenodo.org/records/19303746) and extracts the wav files. Measured on one run: about 27 minutes to download (about 290 KB/s);
`create_dataset`, `remove_long_short` and `prepare_labels` together took about 72 seconds.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). With only three speakers and shared sentences, the
splits hold the same speakers and prompts, so scores are optimistic.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 668 | 647 |
| valid | 95 | 90 |
| test | 190 | 190 (not filtered) |

`remove_long_short` drops the 26 train and valid clips longer than 30 s.

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
