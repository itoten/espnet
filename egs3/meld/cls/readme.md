# meld speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on meld alone, and scores it on meld's own test split (7 classes:
`angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/`surprise`, spelled to match
the other corpora here; MELD's own labels are `anger`/`disgust`/`fear`/`joy`/
`neutral`/`sadness`/`surprise`).

[MELD](https://affective-meld.github.io/) ships audio as MP4; `create_dataset`
downloads the corpus if not already present, extracts it, converts every clip
to 16 kHz mono WAV, and writes one manifest per split. `MELD` points at an
existing MELD source tree, or where to download it to.

`remove_long_short` filters the train and valid splits to 0.2-30 s; the test
split is left untouched so the reported scores cover the whole set.

## Data

MELD (Poria et al., ACL 2019) extends EmotionLines: utterances of the TV series
*Friends*, cut out from the episodes by the subtitle times, with audio, video and
text. The CSV tables hold 13,708 utterances (train 9,989, valid 1,109, test 2,610) from
about 300 speakers (character names, concentrated on the six main characters). The
emotion is the majority vote of three annotators who also saw the video
(Cohen's kappa 0.43), so the label does not rest on the audio alone. The
audio inside the MP4 files is mostly 6-channel (5.1) AAC and is mixed down to mono.
After the exclusions below, 13,704 utterances (about 11.9 hours) are used, and
`remove_long_short` then drops 116 train and 16 valid utterances outside 0.2-30 s.

Four utterances listed in `excluded_utterance_ids` (`dataset/config.yaml`) are left
out because they cannot be used: the MP4 of train `Ross-dia125-utt3` is corrupt, the MP4
of valid `Phoebe-dia110-utt7` is missing, and the CSV time spans of test
`Student-dia38-utt4` and `Phoebe-dia220-utt0` are about 305 s and 235 s, so each clip
is minutes long for one label.

### Duplicates

Of the 13,704 utterances, 43 groups (91 files, 48 surplus) have the same or nearly
the same sound. 24 of the groups carry different emotion labels, and 3 groups span
the official train and test splits. 24 groups are neighbouring utterances of one dialogue whose
subtitle times overlap, so the same stretch was cut twice (inferred). They are kept
for now so that the official split stays as it is, and are to be removed at a later
point.

## License and citation

The GitHub repository `declare-lab/MELD` lists GPL-3.0. The clips come from a
television series; the paper states that a legal office was consulted and that using
and distributing very short videos falls under fair use. Cite Poria, Hazarika,
Majumder, Naik, Cambria and Mihalcea, "MELD: A Multimodal Multi-Party Dataset for
Emotion Recognition in Conversations", ACL 2019.

## Quick start

```bash
# 1) Write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`MELD_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.

**Duplicates.** Many recordings have the same sound as another one (identical after decoding,
or a waveform correlation of 0.998 or more). 19 recordings are same-label copies; `exclude_duplicates: true` in
`dataset/config.yaml` drops them (`duplicate_ids`, the first of each group by id is kept).
53 recordings sit in same-sound groups whose members carry different labels;
`exclude_conflicting: true` drops all of them (`conflicting_ids`). Both default to `false`, so
nothing is removed unless chosen. The four utterances in `excluded_utterance_ids` are always
removed.
