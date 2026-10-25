# MaSERC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MaSERC alone and scores it on MaSERC's own test split, classifying each
utterance into one of six emotions.

[MaSERC](https://data.mendeley.com/datasets/b9jrrpdxgb) is an acted Marathi corpus
recorded by students and faculty of MIT Art, Design and Technology University
(Pune). The archive holds 808 `.wav` files named
`<series> <number>[ name]-<Emotion>-<take>.wav`, one short sentence each, in six
emotions: `angry`, `fear`, `happy`, `neutral`, `sad` and `surprise`. The files are
44.1 kHz stereo (the two channels are nearly identical), so `create_dataset`
converts them to 16 kHz mono with ffmpeg. The archive contains no label table,
partition or speaker table; the description on the dataset page mentions
predefined splits, speaker metadata, a six-sentence monologue scenario and a
16 kHz mono release, none of which is in the archive.

The label is the emotion word in the file name; `dataset/config.yaml` maps it
(including three misspellings of `surprise`) to the manifest label. The numeric
suffix is not a consistent emotion code (`Sad-2` in `student` files, `Sad-5` in
`new` files), so it is not used.

`create_dataset` cleans the file list before splitting:

- 5 files (all of `student 71`) have zero samples and are skipped.
- Files whose audio is byte-identical to a file with a different name form
  identical sets. In each set the first file by name is kept (27 dropped). When the
  names in a set disagree on the emotion (4 files, e.g. `Sad-3` and `fear-3` with the
  same audio), all of them are dropped.

That leaves 772 recordings in 129 sessions, 0.80 hours, 1.4 to 12.3 seconds long:
132 angry, 129 surprise, 128 each of fear, happy and neutral, 127 sad.

Things to know when reading the scores:

- Seven pairs of recordings remain whose 16 kHz audio is identical although the
  files differ in bytes (one is 32-bit float, the other 16-bit PCM): the six
  emotions of `student 37` and `student 38`, and the angry clips of `student 33` and
  `student 34`. They are used as they are.
- Listening to a few clips, the paralinguistic emotion (prosody, voice quality)
  was hard to hear; whether this comes from the acting or from cultural
  differences is not known. The dataset page says three annotators verified the
  labels by majority vote (Cohen's kappa at least 0.78), but their individual
  judgements are not in the archive, so the label is the emotion each file was
  recorded as. Take the scores with some care.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MASERC` — the directory holding the extracted `Entire Marathi Dataset Corpus/` directory, or where to download it to
- `MASERC_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the zip (285 MB, SHA-256-checked) from
[Mendeley Data](https://data.mendeley.com/datasets/b9jrrpdxgb) and extracts the
wav files. Measured on one run: downloading and extracting took about 113 seconds;
`create_dataset`, `remove_long_short` and `prepare_labels` together took about
77 seconds.

## Splits

No partition is published. The session in the file name (series and number, e.g.
`student 91`) is taken as the grouping unit, and whole sessions are split
70/10/20 at random (`split_seed` in `dataset/config.yaml`), so no session spans two
splits. A session is probably one speaker, but the archive does not say so, and
the same speaker may appear in several sessions.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 537 | 537 |
| valid | 78 | 78 |
| test | 157 | 157 (not filtered) |

`remove_long_short` drops nothing.

## License and citation

The Mendeley Data record is licensed CC BY 4.0. Cite the dataset record:
MaSERC: A Dual-Scenario Marathi Emotional Speech Corpus for Speech Emotion
Recognition Research, Mendeley Data, V1, doi:10.17632/b9jrrpdxgb.1.

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
