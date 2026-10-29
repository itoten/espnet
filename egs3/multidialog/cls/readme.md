# MultiDialog speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MultiDialog alone and scores it on MultiDialog's own test splits, classifying
each utterance into one of seven labels.

[MultiDialog](https://multidialog.github.io) (Park et al., "Let's Go Real Talk: Spoken
Dialogue Model for Face-to-Face Conversation", ACL 2024, [arXiv:2406.07867](https://arxiv.org/abs/2406.07867))
is English two-person dialogue acted by 12 people (six women, six men, aged 20 to 30, from six
countries). The scripts come from the text dialogue dataset Topical-Chat, in which every
utterance already carries one of eight emotion labels; the actors read the script and acted
the label shown for each utterance (paper Section 3.2). The audio release used here (Hugging
Face `IVLLab/MultiDialog`) holds one WAV file per utterance, 16 kHz mono.

## Labels

The label is the emotion given to the utterance in the Topical-Chat script, which the actor
was asked to act; it is not a judgement of the recording. The metadata has eight labels:
`Neutral`, `Happy`, `Surprised`, `Sad`, `Disgusted`, `Fearful`, `Angry` and
`Curious to dive deeper`. The recipe uses seven (lower case: `neutral`, `happy`, `surprise`,
`sad`, `disgust`, `fear`, `angry`) because **`Curious to dive deeper` is merged into
`neutral`**: the paper (Section 3.2) says that for the recordings these two labels were
combined into one (Neutral) because there is no visually apparent difference between them,
so the actors acted both the same way. The merged-in utterances are 81,045 of the 187,733
used (43%). `merge_labels` in `dataset/config.yaml` controls this; set it to `{}` to keep
`curious` as a separate eighth label.

The actor is the last letter of the file name (`a` to `l`). The paper's Table 6 gives each
actor's accuracy in a listening study (below) and the release notes name a "gold emotion
dialogue subset": the dialogues of actors a, b, c, e, f, g, i, j and k, those with accuracy
above 40%. The Hugging Face dataset card names the same nine actors (a, b, c, e, f, g, i,
j and k) as the gold emotion dialogue subset. Among the utterances the recipe uses, the file
name's last letter puts 143,816 utterances (245.4 hours) in the gold actors and 43,917 (79.6
hours) in the others (d, h and l). The recipe uses all actors.

## Listening study

Paper, Appendix A.3: 25 participants each predicted the emotion of utterances randomly
sampled from the dataset (70 utterances in total). The paper does not say whether they saw
the video or only heard the audio; it compares the result with human audio-visual emotion
recognition. Per-actor accuracies (Table 6):
a 69.3, b 63.6, c 59.3, d 33.8, e 41.5, f 43.8, g 50.0, h 37.0, i 60.0, j 44.7, k 44.0,
l 21.2 (%). Table 7 is the confusion matrix over the actors above 40%: neutral is
recognised 0.88 of the time, happy 0.75, angry 0.76, disgusted 0.83, surprised 0.68,
sad 0.59 and fearful 0.39. The paper says fearful and sad had lower accuracy. The individual
answers are not in the download. Table 7 has no `curious` row.

## Data

Measured on the metadata and the extracted audio (all 187,854 audio files listed in
`metadata.jsonl` are present, all readable, 16 kHz mono):

- 187,854 utterances, 8,723 dialogues, 325.45 hours, 0.021 to 298.4 s long. The dataset
  page states 187,859 utterances, 8,733 dialogues and 339.71 hours; the counts here are those
  of the files.
- 2 utterances are not used: they have the same sound (waveform correlation 0.9992) but
  different labels (`conflicting_ids` in `dataset/config.yaml`).
- Other than that pair, no byte-identical files and no near-duplicates (correlation 0.98 or
  more) were found in the whole corpus; one pair of 0.04 s and 0.06 s clips has correlation
  0.94 (not looked into).

| label | train | valid | test |
|---|---|---|---|
| neutral (includes curious) | 98,623 | 11,418 | 12,207 |
| surprise | 24,473 | 3,192 | 2,627 |
| happy | 23,769 | 2,882 | 2,796 |
| sad | 2,058 | 233 | 195 |
| disgust | 1,121 | 148 | 166 |
| fear | 818 | 54 | 51 |
| angry | 699 | 121 | 82 |

(Numbers after `remove_long_short`; see Splits.)

## Environment variables

Both are optional.

- `MULTIDIALOG` - a directory holding `metadata.jsonl` and the audio, or where to download
  them (default `download/` under the recipe)
- `MULTIDIALOG_OUTPUT` - where the converted audio and manifests are written (default `data/`)

## Download

`prepare_source` downloads `metadata.jsonl` and the 18 `data/<split>/*.tar.gz` archives
(about 33 GB in total, 31 GB measured on disk) of the Hugging Face dataset, which needs no
sign-in, and extracts them to `audio/<split>/` (about 36 GB). Measured: the first 1.8 GB
archive took about 1.5 minutes; `create_dataset` (converting 187,854 files with 16
parallel ffmpeg jobs) took about 8 minutes and about 36 GB more disk; `remove_long_short` and
`prepare_labels` ran together with the stages around them; `collect_stats` took about 5
minutes.

## Splits

The published partition is used: `train`; `valid_freq` and `valid_rare` together as valid;
`test_freq` and `test_rare` together as test.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 151,642 | 151,561 |
| valid | 18,069 | 18,048 |
| test | 18,141 | 18,124 |

`remove_long_short` drops utterances shorter than 0.2 s or longer than 30 s. Unlike most
recipes, it is applied to test too: the longest test utterance is 298 s, which may not fit in
GPU memory at inference (not tried).
The test utterances removed are 14 under 0.2 s and 3 over 30 s.

## License

The Hugging Face dataset card gives `license: cc` and no license type; the paper and the
card state no further terms. The video recordings are distributed separately (Google Drive)
and are not used here.

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
