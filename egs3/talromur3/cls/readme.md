# Talromur 3 speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Talromur 3 alone and scores it on Talromur 3's own test split, classifying
each utterance into one of six styles.

[Talromur 3](http://hdl.handle.net/20.500.12537/387) (v1.0) is an Icelandic
speech corpus for emotional speech synthesis, recorded in 2024 by Reykjavik
University and Grammatek in a soundproofed studio. Ten professional voice
talents (`F01`-`F05`, `M01`-`M05`) read 340 short sentences in each of six
styles: `neutral`, `happy`, `sad`, `angry`, `surprise` and `helpful`. `helpful`
is a speaking style for virtual assistants. What counts as an emotion is a
question for psychology and does not matter here: the aim is a general model of
paralinguistic information in speech, so `helpful` is a class like the others
(all six are used). Non-neutral styles were also read at five
intensity levels, which this recipe does not use.

The release holds 21,516 recordings (24.22 hours, 0.85 to 46.5 seconds, 48 kHz
24-bit mono FLAC). The README of the corpus and the CLARIN-IS page say 21,526 clips;
the 10 extra are probably the ten `index.tsv` files, since the archive holds
21,516 FLAC and 10 `index.tsv` files (21,526 files in all). This was not checked
further. Of these, 1,140 are an addendum of short readings of numbers and letters
(`addendum`, `addendum_style2` directories; the corpus README says v1.0 adds the
alphabet and the numbers 0 to 20). They are not one of the six styles, so the
recipe leaves them out; the log reports the count. To
include them, remove them from `drop_styles` in `dataset/config.yaml` and map them
in `emotion_map`. The remaining 20,376 recordings (23.78 hours) are used:
3,396 per style, except that speakers `M03` and `M04` skipped two sentences, so
their styles have 338 recordings each. The label is the style directory name
given in each speaker's `index.tsv`; `dataset/config.yaml` maps it to the
manifest label. One speaker's `surprised` directory is spelled `suprised`
(`M02`); both spellings map to `surprise`. `create_dataset` converts the audio
to 16 kHz mono 16-bit WAV with ffmpeg. The raw levels are not normalised.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `TALROMUR3` — the directory holding the extracted `recordings/` directory, or where to download it to
- `TALROMUR3_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `recordings.zip` (6.8 GB) from the CLARIN-IS
repository without login, tests the archive, and extracts the FLAC files and
`index.tsv` tables (the zip is then removed; extracted, the FLAC files take
6.4 GB). The repository publishes no checksum. On one run the download took
about 18 minutes (other downloads were running at the same time), the extraction
26 seconds, and `create_dataset`, `remove_long_short` and `prepare_labels`
together 2 minutes 55 seconds with 8 parallel ffmpeg processes.

## Splits

The corpus publishes no partition. Whole speakers are held out so that no voice
appears in two splits: `F04` and `M04` for test, `M05` for valid and the other
seven speakers for training (`test_speakers` and `valid_speakers` in
`dataset/config.yaml`). Scores are therefore speaker-independent. The sentences
are shared across speakers, so test sentences also appear in training. Every
style has the same count in each split.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 14,268 (2,378 per style) | 14,254 |
| valid | 2,040 (340 per style) | 2,040 |
| test | 4,068 (678 per style) | 4,068 (not filtered) |

`remove_long_short` drops 14 training clips longer than 30 seconds. The test
split is not filtered and holds 4 clips longer than 30 seconds (up to 35 s).

## License and citation

The CLARIN-IS item page states that the item is publicly available under CC BY
4.0, while the corpus README and the item description also call the corpus public
domain. This recipe follows the stricter CC BY 4.0, so attribute the authors. Authors: Gunnar Thor
Ornolfsson, Atli Thor Sigurgeirsson, Anna Bjork Nikulasdottir and Daniel
Schnell (Reykjavik University); issued 2025-08-25; handle
`http://hdl.handle.net/20.500.12537/387`. The corpus provides no citation text
of its own.

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
