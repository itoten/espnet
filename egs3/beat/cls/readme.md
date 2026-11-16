# BEAT speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BEAT alone and scores it on a held-out-speaker test split, classifying
each speech segment into one of eight emotions: `neutral`, `happy`, `angry`,
`sad`, `contempt`, `surprise`, `fear` and `disgust`.

[BEAT](https://huggingface.co/datasets/haiyangliu0/BEAT) (Body-Expression-Audio-Text)
was recorded for conversational gesture synthesis: speakers talk while their
body and face are captured with motion capture. The Hugging Face repository
holds the English part, `beat_english_v0.2.1`, with 30 speakers (one folder per
speaker id). For each recording it has the audio (`.wav`), a word and phone
alignment (`.TextGrid`), a one-line emotion label (`.csv`), gesture annotations
(`.txt`), body motion (`.bvh`) and facial blendshapes (`.json`). This recipe
uses only the audio, the alignments and the labels.

Recordings are named `<speaker id>_<name>_<type>_<index>_<index>`. The type is
`0` for self-talk (a monologue of about one minute, performed in the emotion of
the label) and `1` for conversation (about ten minutes, nearly all labelled
neutral). The label is the first field of the `.csv`, for example
`01_happiness,0.0,62.0,62.0,1`; one label covers the whole recording. The
release spells neutral `00_netural`; `emotion_map` in `dataset/config.yaml`
maps the eight release labels to the names above. `contempt` is kept as its
own class.

## Segmentation

The recordings are far longer than an utterance, so `create_dataset` cuts each
one into segments along the `words` tier of its TextGrid, and every segment
gets the label of its recording:

1. consecutive words stay in one segment until the pause between two words is
   at least `pause_threshold` (0.3 s);
2. a segment longer than `max_segment` (15 s) is split at its longest internal
   pause, again and again, until every piece fits;
3. pieces shorter than `min_segment` (1 s, first word start to last word end)
   are dropped and counted in the log;
4. each cut is widened by `padding` (0.1 s) on both sides, but never past the
   middle of the pause to the neighbouring word.

Segment ids are `beat-<recording>_<NNN>`, NNN counting the kept segments of the
recording from 000 in time order, so they do not change between builds with the
same settings. Each segment is cut with ffmpeg and written as 16 kHz mono 16-bit
PCM. `data/segments.tsv` lists every segment with its recording, start, end,
label and words. All settings are in `dataset/config.yaml`, together with
`recording_types`, which can leave out the neutral-heavy conversations (type
`1`).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BEAT` — the Hugging Face snapshot (or its `beat_english_v0.2.1/beat_english_v0.2.1` folder), or where to download it to
- `BEAT_OUTPUT` — where the segments and manifests are written

## Download

`prepare_source` downloads the `.wav`, `.TextGrid` and `.csv` files of the
Hugging Face dataset `haiyangliu0/BEAT` at a pinned revision, with no login
(about 9 GB: audio 8.75 GB, alignments 0.2 GB). The motion capture (about 60 GB) and the facial
blendshapes (about 15 GB) are not downloaded.

## The data

TODO: filled after the first build (counts of the release).

## Splits

There is no published split for classification, so the speakers are split by
id (`split_speakers` in `dataset/config.yaml`): 24 for training, 3 for
validation (11, 27, 28) and 3 for test (21, 29, 30). No speaker is in two
splits. Ten speakers have about 118 self-talk recordings each and the other
twenty about 30; validation and test each take one speaker of the first group
and two of the second, the highest ids in each group. Every speaker has
self-talk recordings for all eight emotions, so all eight labels are in every
split. `remove_long_short` drops segments outside 0.2-30 s from train and
valid; with the default segmentation (1-15 s plus padding) it should drop
none.

## License and citation

The Hugging Face metadata says Apache-2.0. Check the terms of the BEAT project
before redistributing or using the data commercially. Please cite:

```
@inproceedings{liu2022beat,
  title     = {{BEAT}: A Large-Scale Semantic and Emotional Multi-Modal Dataset for Conversational Gestures Synthesis},
  author    = {Liu, Haiyang and Zhu, Zihao and Iwamoto, Naoya and Peng, Yichen and Li, Zhengqing and Zhou, You and Bozkurt, Elif and Zheng, Bo},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2022},
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
