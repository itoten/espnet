# Emo-Emilia speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Emo-Emilia alone and scores it on a held-out part of Emo-Emilia,
classifying each clip into one of seven emotions.

[Emo-Emilia](https://huggingface.co/datasets/ASLP-lab/Emo-Emilia) was released
with the C<sup>2</sup>SER paper as a speech emotion recognition test set. Its
clips are a subset of Emilia, a large multilingual speech-generation corpus of
in-the-wild speech. According to the dataset card, the clips were labelled
automatically, then each label was checked by at least two experts; the set has
1,400 clips, 100 per emotion in each of two languages (English and Mandarin
Chinese, 700 each). The language is the prefix of the clip id (`EN_` or `ZH_`).

Labels come from the `emotion` field of `Emo-Emilia-ALL.jsonl`. All seven are
emotions and all are kept, renamed to the spellings of the other recipes:

| `emotion` field | label |
|---|---|
| angry | angry |
| disgusted | disgust |
| fearful | fear |
| happy | happy |
| neutral | neutral |
| sad | sad |
| surprised | surprise |

The `wav` field of the JSONL holds a path from the authors' layout
(`./Emo-Emilia/wav/<id>.wav`); the recipe uses only its file name and reads the
clip from `wav/` of the download.

`create_dataset` converts every clip to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EMO_EMILIA` — the directory holding `Emo-Emilia-ALL.jsonl` and `wav/`, or where to download them to
- `EMO_EMILIA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `wav/*.wav`, `Emo-Emilia-ALL.jsonl` and `README.md`
from Hugging Face at a pinned commit (`519da7a59f4a5a9b9bba5b5c318f1e2ee0eed97a`)
with `huggingface_hub.snapshot_download`; no login is needed. Files already
present are not fetched again.

## The data

TODO: filled after the first build (counts of the release).

## Splits

The release is meant as a test set and publishes no partition. To train on it
alone, clips are split 70/10/20 at random within each label (`train_ratio`,
`valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`). The split is
not stratified by language. Speakers are not identified in the release, so the
splits may share speakers. Because Emo-Emilia is published as an evaluation set,
scores from this recipe are not comparable to results that use the whole set
for testing.

## License and citation

CC BY-NC 4.0 (non-commercial). Emo-Emilia is a subset of Emilia; check the
Emilia terms as well before use. Please cite:

```
@misc{zhao2025steeringlanguagemodelstable,
  title         = {Steering Language Model to Stable Speech Emotion Recognition via Contextual Perception and Chain of Thought},
  author        = {Zhixian Zhao and Xinfa Zhu and Xinsheng Wang and Shuiyuan Wang and Xuelong Geng and Wenjie Tian and Lei Xie},
  year          = {2025},
  eprint        = {2502.18186},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  doi           = {10.1109/TASLPRO.2025.3648793},
  url           = {https://arxiv.org/abs/2502.18186},
}
```

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
