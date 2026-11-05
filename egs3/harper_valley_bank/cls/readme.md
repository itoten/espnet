# Harper Valley Bank speech sentiment recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Harper Valley Bank alone and scores it on Harper Valley Bank's own test split,
classifying each utterance into one of three polarity classes: `negative`, `neutral`
and `positive`.

[Harper Valley Bank](https://github.com/cricketclub/gridspace-stanford-harper-valley)
(Gridspace and Stanford) is a corpus of 1,446 simulated English contact-center calls
between a bank agent and a caller, with 59 speakers across the agent and caller
roles. Each call is stored as two 8 kHz mono wav files, one per side. The
transcript of a call is a list of segments with a human transcript, a dialog act
and a softmax over `negative`, `neutral` and `positive`. This recipe makes one
utterance per segment that has spoken words: 19,456 of the 25,730 segments,
11.4 hours in total, 0.06 to 18.7 seconds long (the 47.3 hours of audio in the
repository count both sides of each call in full).

The label is the arg-max of the segment's softmax. The README of the corpus
describes it as the output of Gridspace's emotion model ("emotional valence"), so
the labels are model outputs, not human judgements, and they measure polarity,
not discrete emotions. Over the 19,456 segments: 10,960 neutral, 8,080 positive,
416 negative.

Segment boundaries come from Gridspace's automatic speech recognition (each
segment ends at the end of its last recognized word), and the label is assigned
to every segment, including ones with no speech. The 6,274 segments whose human
transcript is empty or only tags such as `[noise]` or `[music]` are therefore
excluded.

`start_ms` of a segment is its position in the per-side wav. `offset_ms`, which the
corpus README describes as the offset from the start of the recording, does not
match the audio (measured by scanning the energy of the audio around both
fields); this recipe uses `start_ms`. The wav is chosen by `channel_index`
(1 is the caller side, 2 the agent side), not by `speaker_role`: in 37 calls
the two disagree, and listening shows the voice is in the wav named by
`channel_index`. `create_dataset` cuts each segment out of its call and
converts it to 16 kHz mono with ffmpeg. 150 of the segments are shorter than
0.2 s; `remove_long_short` removes those in train and valid.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `HARPER_VALLEY_BANK` — the directory holding the extracted repository (`data/audio`, `data/transcript`), or where to download it to
- `HARPER_VALLEY_BANK_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Gridspace and Stanford Harper Valley Bank repository on GitHub (`cricketclub/gridspace-stanford-harper-valley`, files as downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from GitHub, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 25,730 | 59 | 12.02 |

How these were counted: every segment in the 1,446 transcript files `data/transcript/*.json`; hours from the segments' `duration_ms` (0.03 to 18.66 s); the samples are the utterance segments of the 1,446 calls. The audio is two 8 kHz mono wav files per call, one per side, 47.3 hours in all. Speakers are the 59 distinct speakers named in the metadata (agent role 58, caller role 55, 54 in both roles); the release has no gender, age or language information.

### Files per label

| label (arg-max of a segment's `emotion` scores) | samples | hours |
|---|---|---|
| neutral | 17,149 | 5.813 |
| positive | 8,148 | 5.919 |
| negative | 433 | 0.292 |

How these were counted: the largest of the three scores (`negative`, `neutral`, `positive`) in each segment's `emotion` field, over all 25,730 segments. The corpus README describes the scores as the output of Gridspace's emotion model; the release has no human emotion judgement. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- 6,274 segments have a human transcript that is empty or only tags such as `[noise]` or `[music]`; their audio is background sound, yet every segment carries emotion scores.
- The segment boundaries come from automatic speech recognition (each segment ends at the end of its last recognised word).
- `offset_ms` does not match the position of the segment in the audio (checked by scanning the energy of the audio); `start_ms` does. `duration_ms` is the length of the segment, not of the call as the README says. The number of utterances and words per call differs from the paper's table.
- In 37 calls (596 segments) `speaker_role` and `channel_index` name opposite sides; the voice of the segment is in the wav that `channel_index` names.
- Observation by the recipe author (2026-10-19 at the latest): segments cut from the two candidate wav files of the 37 calls were listened to; the voice was in the file chosen by `channel_index`. Segments from the empty-transcript set sounded like background sound.


## Download

`prepare_source` downloads the repository tarball (676 MB compressed, 2.6 GB
extracted) from GitHub without a login and extracts it. Measured on one run:
about 3 minutes to download; `create_dataset` took 81 seconds, `remove_long_short`
14 seconds and `prepare_labels` under a second (8 ffmpeg workers).

## Splits

The corpus publishes no partition. Whole calls are assigned at random
(`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in `dataset/config.yaml`),
so no call spans two splits. Speakers are not held out: the same agents and
callers appear in several calls and therefore in several splits.

| split | utterances | after `remove_long_short` | negative / neutral / positive (filtered) |
|---|---|---|---|
| train | 13,599 | 13,488 | 273 / 7,586 / 5,629 |
| valid | 1,950 | 1,937 | 51 / 1,090 / 796 |
| test | 3,907 | 3,907 (not filtered) | 92 / 2,160 / 1,655 (unfiltered) |

The label is skewed to neutral and `negative` is rare, so `prepare_labels` writes class
weights.

## License and citation

The repository's `LICENSE` file is the Creative Commons Attribution 4.0
International license. The corpus README asks for this citation:

```
@misc{wu2020harpervalleybank,
  title         = {HarperValleyBank: A Domain-Specific Spoken Dialog Corpus},
  author        = {Mike Wu and Jonathan Nafziger and Anthony Scodary and Andrew Maas},
  year          = {2020},
  eprint        = {2010.13929},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
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
