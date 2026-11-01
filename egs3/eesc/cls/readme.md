# EESC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on the Estonian Emotional Speech Corpus (EESC) alone and scores it on EESC's own
test split, classifying each utterance into one of four emotions.

[EESC](https://doi.org/10.6084/m9.figshare.25249744) (Institute of the Estonian
Language) is read-out, non-acted Estonian speech from one female speaker. The
release holds 1164 sentences, 1.08 hours in total, 0.45 to 14.0 seconds long:
338 neutral, 306 anger, 271 joy and 249 sadness. (The figshare description says 1234
sentences; the archive holds 1164 `.wav` files, each with a `.TextGrid`.)

**The label is the emotion that listeners perceived, not the emotion the speaker
intended.** It is the value in the last tier (`emotion`) of each `<id>.TextGrid`.
`dataset/config.yaml` maps `anger`, `joy`, `sadness` and `neutral` to `angry`,
`happy`, `sad` and `neutral`. The recordings are 44.1 kHz 16-bit mono, so
`create_dataset` converts them to 16 kHz with ffmpeg.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the figshare record 25249744 (Estonian Emotional Speech Corpus; `.wav` and `.TextGrid` files, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from figshare, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,164 | 1 | 1.08 |

How these were counted: every `.wav` in the archive (44.1 kHz mono 16-bit, 0.45 to 14.0 s, each with a `.TextGrid`); hours from the file lengths. All sentences are read by one female speaker.

### Files per label

| label (the `emotion` tier of the TextGrid) | samples |
|---|---|
| neutral | 338 |
| anger | 306 |
| joy | 271 |
| sadness | 249 |

How these were counted: the value in the last tier (`emotion`) of each `<id>.TextGrid`.

### Issues in the release

- The figshare description says 1,234 sentences; the archive holds 1,164 `.wav` files.
- The paper's numbers of identified (908) and unidentified (326) sentences do not match the per-label counts of the files; how sentences without a majority label were labelled is not stated in the documents read.


## How the emotions arose and how the labels were made

This section summarises what the corpus authors report, so that the labels are not
read as more than they are. Sources (read on 2026-10-24): Altrov & Pajupuu (2012),
"Estonian Emotional Speech Corpus: theoretical base and implementation", ES3
workshop, pp. 50-53 ([Academia.edu](https://www.academia.edu/7142292/Estonian_Emotional_Speech_Corpus_Theoretical_base_and_implementation));
Altrov & Pajupuu (2015), JEFUL 6(3),
[doi:10.12697/jeful.2015.6.3.01](https://ojs.utlib.ee/index.php/jeful/article/download/jeful.2015.6.3.01/10143/14873);
the corpus site <http://peeter.eki.ee:5000/>, which we could not open.

- **How the speaker's emotion arose.** One female reader, chosen for clear
  articulation, a pleasant voice and empathy (Baron-Cohen & Wheelwright's empathy
  test), read 130 journalistic passages aloud. She was not told which emotion to
  express and was not asked to act. The authors assumed that the content of each
  passage would evoke an emotion in her and that it would show in her voice,
  which avoids the exaggeration of acted speech. The passages had been chosen
  because readers in a separate test agreed that they carried joy, anger or
  sadness. The passages were cut into sentences. The speaker was given no
  instruction about how to speak, so how clearly a given sentence expresses its
  emotion in the voice is known only through the listening test below.
- **How the label was decided.** Each sentence was played on its own (without
  the text) to at least 30 Estonian listeners, aged over 30 and chosen on the
  grounds that vocal emotion recognition is learned within a culture, who chose
  joy, anger, sadness or neutral. A sentence counts as identified when more than
  50% of the listeners chose the same label. 908 of the 1,234 sentences were
  identified (joy 232, anger 277, sadness 191, neutral 208); for 326 the
  listeners did not agree.
- **Whether listeners judged by the wording or by the voice.** The listeners
  were native speakers who understood the words, so the words may have affected their
  choice, since text that evokes joy, anger or sadness contains words that
  express them. The authors raise this and run a second test: the same sentences
  were judged from the written text alone, without audio, by at least 10
  other people (14 reading tests). If the readers' label differs from the
  listeners' label, the authors say the content did not affect the identification;
  if it agrees, the content might have affected it. By this rule, for the 908 identified sentences: content might
  have affected the label for 515 (joy 163, anger 177, sadness 88, neutral 87),
  and did not for 393 (joy 69, anger 100, sadness 103, neutral 121). The authors
  note that the two groups differ in measured intensity for joy, anger and
  neutral, and that this may mean the text also affects how an emotion is
  expressed in the voice. This test shows where the text-only label disagrees
  with the listeners' label; it cannot show that the listeners did not use the
  words.
- **What the release does not contain.** The figshare archive has only the final
  label per sentence. It has neither the listeners' agreement rates nor the
  reading-test flag, which the corpus site shows per sentence, so the sentences
  where the label may rest on the wording cannot be told apart here. Also, the
  paper's counts above (908 identified sentences, 326 not identified) do not
  match the 1164 released sentences and their label counts; how the labels of
  sentences without a majority were set in the release is not stated in the
  documents we read.

For models: because the speaker's emotion came from the text, the wording of many
sentences carries the label, so a model could partly rely on it, and how clearly each
clip expresses its emotion in the voice is unknown. The recipe uses only the audio.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `EESC` — the directory holding the extracted `*.wav` and `*.TextGrid` files, or where to download them to
- `EESC_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `ekorpus.zip` (277 MB, md5-checked) from
[figshare](https://doi.org/10.6084/m9.figshare.25249744) and extracts the `.wav` and
`.TextGrid` files. Measured on one run: `prepare_source` took about 3 minutes
(a plain `curl` of the zip alone took 93 seconds; another download was running at
the same time); `create_dataset`, `remove_long_short` and `prepare_labels`
together took about 85 seconds.

## Splits

The release publishes no partition and has a single speaker. Utterances are split
70/10/20 at random within each label (`train_ratio`, `valid_ratio`, `test_ratio`,
`split_seed` in `dataset/config.yaml`). The splits share the speaker, and the
1164 recordings hold only 1083 distinct sentence texts (80 texts occur more than once),
so the same sentence can fall in different splits and scores are optimistic.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 814 | 814 |
| valid | 117 | 117 |
| test | 233 | 233 (not filtered) |

`remove_long_short` drops nothing.

## License and citation

The figshare record is licensed CC BY 4.0. It cites: Altrov, R. and Pajupuu, H.
2012. Estonian Emotional Speech Corpus: Theoretical base and implementation. In:
4th International Workshop on Corpora for Research on Emotion Sentiment & Social
Signals (ES3), 50-53. Istanbul.

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
