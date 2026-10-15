# DailyTalk speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on DailyTalk alone and scores it on DailyTalk's own test split, classifying each
utterance into one of seven emotions.

[DailyTalk](https://github.com/keonlee9420/DailyTalk) is an English conversational
speech corpus built for text-to-speech (Lee, Park and Kim, 2023,
[arXiv:2207.01063](https://arxiv.org/abs/2207.01063)): two speakers recorded
2,541 dialogues taken from the DailyDialog text corpus, 23,773 utterances and 21.7
hours in total, 0.5 to 17.8 seconds long. The recordings are 44.1 kHz; 23,720 are
mono and 53 are stereo.

The label is the per-utterance `emotion` field of `metadata.json`, inherited from
DailyDialog's text annotation, so it was not assigned by listening to the audio.
`dataset/config.yaml` maps it to `neutral` (`no emotion`), `happy`, `sad`, `angry`,
`fear`, `disgust` and `surprise`. The distribution is highly skewed:

| label | utterances |
|---|---|
| neutral | 18,966 |
| happy | 3,856 |
| surprise | 407 |
| sad | 301 |
| angry | 159 |
| disgust | 66 |
| fear | 18 |

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `DAILYTALK` — the directory holding the extracted `metadata.json` and `data/` (also found one level down, in `dailytalk/`), or where to download them to
- `DAILYTALK_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `dailytalk.zip` (5.3 GB, md5-checked) from the public
Google Drive folder linked in the GitHub README and extracts the wav files and
`metadata.json`. Measured with `gdown` on one run: about 3.3 minutes to download,
45 seconds to extract, and 6 seconds for the md5. `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 3.5 minutes (ffmpeg
resampling with 8 threads).

## Splits

The corpus publishes no partition. Both speakers (`0` and `1` in `metadata.json`)
take part in every dialogue, so speakers cannot be held out. Whole dialogues are
assigned at random to 70/10/20 (`split_seed` in `dataset/config.yaml`), so the
utterances of one conversation never fall in different splits. The splits share
both speakers, and the rarest classes are tiny in valid and test.

| split | dialogues | utterances | after `remove_long_short` |
|---|---|---|---|
| train | 1779 | 16,694 | 16,694 |
| valid | 254 | 2,396 | 2,396 |
| test | 508 | 4,683 | 4,683 (not filtered) |

Test labels: neutral 3715, happy 791, surprise 83, sad 52, angry 25, disgust 11,
fear 6.

## License and citation

The GitHub README states CC BY-SA 4.0. Cite Lee, Park and Kim,
"DailyTalk: Spoken Dialogue Dataset for Conversational Text-to-Speech",
arXiv:2207.01063.

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
