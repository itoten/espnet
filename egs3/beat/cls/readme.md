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

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-18 against the Hugging Face dataset `haiyangliu0/BEAT` (repository revision `604f5eca9d8dc2e1b8c3ed21045f9e24a7b6d3ff`), its `beat_english_v0.2.1` folder: the `.wav`, `.TextGrid` and `.csv` files only (the motion, face and gesture files were not obtained). All counts cover every such file in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| recordings | speakers | total hours |
|---|---|---|
| 1,946 | 30 | 62.74 |

How these were counted: every `.wav` file; hours from the WAV headers (18.0 to 828.1 s). All are 16 kHz mono. Speakers are the folder names (1 to 30). The BEAT paper gives 76 hours for the whole multimodal recording; the released audio adds up to 62.74 hours.

### Files per label

| label in the `.csv` | recording type | recordings | hours |
|---|---|---|---|
| 00_netural | 0 (self-talk) | 971 | 18.732 |
| 00_netural | 1 (conversation) | 178 | 29.886 |
| 01_happiness | 0 | 117 | 1.877 |
| 01_happiness | 1 | 4 | 0.682 |
| 02_anger | 0 | 116 | 1.705 |
| 03_sadness | 0 | 96 | 1.771 |
| 04_contempt | 0 | 116 | 2.217 |
| 05_surprise | 0 | 116 | 1.841 |
| 06_fear | 0 | 116 | 2.036 |
| 07_disgust | 0 | 115 | 1.974 |

How these were counted: the first field of each recording's `.csv` (one line per file, one label per recording) and the recording type, the third field of the file name. The stray file below has no `.csv` and is not in the table. Hours per label are measured in the same way as the total hours above and summed over the recordings of that row.

### Issues in the release

- `21/1_wayne_0_103_103.wav` is a byte-identical copy of `1/1_wayne_0_103_103.wav` in another speaker's folder, without a `.TextGrid` or `.csv`. The recipe skips it.
- The alignments of `3_solomon_0_25_25` (one word) and `3_solomon_0_26_26` (no word) are nearly empty, although the recordings last 88 and 82 s. No segment is cut from them.
- Neutral is spelled `00_netural` in the label files.


## Splits

There is no published split for classification, so the speakers are split by
id (`split_speakers` in `dataset/config.yaml`): 24 for training, 3 for
validation (11, 27, 28) and 3 for test (21, 29, 30). No speaker is in two
splits. Ten speakers have about 118 self-talk recordings each and the other
twenty about 30; validation and test each take one speaker of the first group
and two of the second, the highest ids in each group. Every speaker has
self-talk recordings for all eight emotions, so all eight labels are in every
split. `remove_long_short` drops segments outside 0.2-30 s from train and
valid; with the default segmentation (segments of 1.2 to 15.2 s) it drops
none.

Segments per split with the default settings (both recording types; 39,431
segments, 51.78 hours, from 1,943 recordings):

| label | train | valid | test |
|---|---|---|---|
| neutral | 24,487 | 3,558 | 2,573 |
| happy | 817 | 120 | 629 |
| contempt | 1,097 | 138 | 134 |
| fear | 1,029 | 137 | 153 |
| sad | 994 | 115 | 112 |
| disgust | 979 | 115 | 123 |
| surprise | 913 | 123 | 118 |
| angry | 769 | 108 | 90 |
| all | 31,085 | 4,414 | 3,932 |

Neutral is 79% of the training segments, mostly from the conversations
(recording type 1). The four conversations labelled happiness belong to test
speakers 21 and 30, so 532 of the 629 happy test segments come from
conversations. Set `recording_types: ["0"]` to use the self-talk recordings
only.

## License and citation

Apache-2.0: the Hugging Face metadata says so, and the footer of the
[project page](https://pantomatrix.github.io/BEAT/) reads "Licensed under the
Apache-2.0 license." (checked 2026-11-21; the footer does not say whether it
covers the page or the data). Please cite:

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
