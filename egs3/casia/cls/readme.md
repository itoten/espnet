# CASIA speech emotion recognition (single-corpus baseline)

> **Warning: do not use this recipe's data source.** The audio it downloads is an
> unofficial copy of a corpus that is licensed for a fee, and no permission to
> redistribute it was found. The recipe is kept for reference only. Use it only
> with a copy you have licensed yourself.

In short (details in [Provenance and citation notes](#provenance-and-citation-notes),
checked on 2026-10-23):

- The corpus is distributed for a fee by the Chinese Linguistic Data Consortium
  (ChineseLDC). The freely available 1,200-clip copies trace back to a re-upload
  by an individual user, not to the corpus owners.
- No paper by the corpus authors describing it was found. The paper most often
  cited for it describes a different corpus (one female speaker, recorded for
  speech synthesis).

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on CASIA alone and scores it on CASIA's own test split, classifying each
utterance into one of six emotions.

CASIA is a Mandarin Chinese acted corpus from the Institute of Automation,
Chinese Academy of Sciences. The copy used here has 4 speakers (`liuchanhg`,
`ZhaoZuoxiang`, `zhaoquanyin`, `wangzhe`) reading 50 sentences in six emotions:
`angry`, `fear`, `happy`, `neutral`, `sad` and `surprise`. That gives 1,200
recordings, 200 per emotion (50 per speaker), 0.64
hours in total, 0.96 to 4.93 seconds long. They are 16 kHz mono as distributed.

The label is the emotion in the file name `<sentence>-<emotion>-<speaker>.wav`;
`dataset/config.yaml` maps it to the manifest label. The sentence texts are shared
across speakers and emotions.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the 1,200-clip copy of CASIA obtained from the Hugging Face mirror used by this recipe (downloaded around 2026-10-15); this is not the official ChineseLDC release. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from the Hugging Face mirror, no sign-in. The official corpus is sold by the Chinese Linguistic Data Consortium; the permission to redistribute the copy was not found.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,200 | 4 | 0.64 |

How these were counted: every `.wav` in the copy (16 kHz mono, 0.96 to 4.93 s); hours from the file lengths. Speakers are the 4 distinct names in the file names `<sentence>-<emotion>-<speaker>.wav` (`liuchanhg`, `ZhaoZuoxiang`, `zhaoquanyin`, `wangzhe`; two women and two men according to the documentation).

### Files per label

| label (emotion in the file name) | samples |
|---|---|
| angry | 200 |
| fear | 200 |
| happy | 200 |
| neutral | 200 |
| sad | 200 |
| surprise | 200 |

How these were counted: the emotion field of each file name (50 sentences per speaker and emotion; the texts are shared across speakers and emotions).

### Issues in the release

- The copy has 1,200 clips; the sizes given for the official corpus differ by version (9,600; 12,000; 10,000), and which part of which version this copy is, is not known.
- The copy traces back to a re-upload by an individual user, not to the corpus owners. The paper most often cited for the corpus describes a different corpus (one female speaker, recorded for speech synthesis, no emotions).


## Provenance and citation notes

This section records what was found on 2026-10-23 when tracing where the corpus
comes from, so that others can check it. It is not a judgement on anyone: the
corpus has no dedicated paper that we could find, so every user has had to choose
something to cite and somewhere to get the data from, and the copies and
citations below are the ones that are easy to find. We may have missed a document
that settles these questions; corrections are welcome.

### The corpus and its distributor

- **ChineseLDC catalogue, CLDC-SPC-2005-010 "CASIA汉语情感语料库"** (CASIA Chinese
  emotional speech corpus). The resource page
  <http://www.chineseldc.org/resource_info.php?rid=76> was not reachable from our
  network; the archived copy
  [web.archive.org/web/20200501144342/…rid=76](http://web.archive.org/web/20200501144342/http://www.chineseldc.org/resource_info.php?rid=76)
  gives: developed by the Institute of Automation, Chinese Academy of Sciences;
  4 speakers, 6 emotions, 9,600 utterances (300 sentences with the same text in
  every emotion plus 100 with different text); added 2008-06-11, updated
  2010-10-09; licence fees of 5,000 RMB (research in China), 15,000 RMB
  (commercial in China), 35,000 RMB (research outside China) and 105,000 RMB
  (commercial outside China). We could not check whether these terms still apply.
- The 2005 version of the same catalogue entry describes the corpus differently:
  [intro.htm](http://web.archive.org/web/20051214160150/http://www.chineseldc.org:80/doc/CLDC-SPC-2005-010/intro.htm)
  says 6 emotions and 12,000 utterances (300 same-text and 200 different-text
  sentences), while
  [label.htm](http://web.archive.org/web/20051214151014/http://www.chineseldc.org:80/doc/CLDC-SPC-2005-010/label.htm)
  and
  [report.htm](http://web.archive.org/web/20051214172403/http://www.chineseldc.org:80/doc/CLDC-SPC-2005-010/report.htm)
  say 5 emotions (高兴 happy, 悲哀 sad, 生气 angry, 惊吓 frightened, 中性 neutral),
  two female and two male speakers (folders `w1`, `w2`, `m1`, `m2`), 500 sentences
  per emotion and 10,000 utterances, 16 kHz 16-bit PCM, about 35 dB SNR. The
  different-text sentences are written so that their meaning matches the emotion.
- The ChineseLDC home page
  ([archived 2024-11-26](http://web.archive.org/web/20241126183355/http://www.chineseldc.org/))
  gives the Institute of Automation as copyright holder and its address as
  contact, and the
  [resource list archived 2024-06-15](http://web.archive.org/web/20240615040043/http://www.chineseldc.org/resource_list.php?begin=0&count=20)
  still lists the corpus. We found no statement from the corpus authors
  themselves naming ChineseLDC as the distributor, nor any statement allowing free
  redistribution.

### The 1,200-clip version

- [EmoBox](https://github.com/emo-box/EmoBox) (Ma et al.,
  [arXiv 2406.07162](https://arxiv.org/abs/2406.07162)) lists CASIA with 4
  speakers, 6 emotions and 1,200 utterances, and its
  [README](https://github.com/emo-box/EmoBox/blob/39c306b37302e6696d4622e23da2d32418a30491/README.md)
  links to Baidu AI Studio dataset
  [209512](https://aistudio.baidu.com/datasetdetail/209512).
- That AI Studio page, "CASIA汉语情感音频语料库-1200音频版本", was published on
  2023-04-16 by an individual user; its licence field reads "其他" (other), and it
  names the Institute of Automation and links the ChineseLDC page above as the
  source. It holds `CASIA.zip` (54.89 MB) with files dated 2005.
- This recipe downloads the Hugging Face mirror
  [`Bill13579/casia-mirror`](https://huggingface.co/datasets/Bill13579/casia-mirror)
  (created 2025-06-29, licence field `other`, empty `LICENSE` file). The 20 files
  listed on the AI Studio page match our copy in size and modification time to
  the second, and our extracted `CASIA/` folder is dated 2023-04-16 13:55, three
  minutes before the AI Studio upload, so the mirror appears to be the same zip.
  We did not compare checksums.
- The 1,200 clips use sentence numbers 201 to 250 and speaker names
  (`liuchanhg`, `ZhaoZuoxiang`, `zhaoquanyin`, `wangzhe`) rather than the
  `w1`/`m1` folders of the 2005 description, and have 6 emotions. For sentences
  201 to 205 (transcribed with Whisper) the text is the same in every emotion and
  for every speaker. Which part or version of the full corpus they were cut from
  is not documented.

### The paper usually cited

- **J. Tao, F. Liu, M. Zhang, H. Jia, "Design of Speech Corpus for Mandarin Text to
  Speech"**, Blizzard Challenge 2008 workshop. Full text from the Institute's site
  ([archived PDF](http://web.archive.org/web/20240503152623/https://nlpr.ia.ac.cn/2008papers/gjhy/gh166.pdf)).
  It describes the "CASIA Mandarin corpus" for speech synthesis: one professional
  female speaker (aged 30, chosen from eight candidates), 5,000 sentences, about
  7 hours, delivered as the common Mandarin corpus of Blizzard Challenge 2008. It
  does not mention emotion.
- The Blizzard Challenge 2008 overview
  ([Karaiskos et al.](https://www.isca-archive.org/blizzard_2008/karaiskos08_blizzard.pdf))
  likewise describes the Mandarin data as a 6.5-hour single female speaker
  database released by the Institute of Automation. The same group's system paper
  ([Tao et al., "The WISTON Text to Speech System for Blizzard 2008"](https://www.isca-archive.org/blizzard_2008/tao08_blizzard.pdf))
  points to a different licence page for that database. The "Design of Speech
  Corpus" paper itself is not in the
  [festvox list of Blizzard 2008 papers](http://www.festvox.org/blizzard/blizzard2008.html)
  or the ISCA Archive.
- The Institute of Automation has published many corpora under the "CASIA" name
  (the ChineseLDC catalogue lists, for example, a weather-forecast broadcast
  corpus, northern- and southern-accent corpora and a question-intonation
  corpus), which may be why the speech-synthesis paper became the usual
  reference.

### How some papers describe and cite the corpus

These are the papers we read; there are many more, and we did not survey them.

| Paper | Description of CASIA | Reference given |
|---|---|---|
| Ma et al., "EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark", [arXiv 2406.07162](https://arxiv.org/abs/2406.07162) | acted, Mandarin, 6 emotions, 4 speakers, 1,200 utterances, 0.6 h | [10] J. Tao, F. Liu, M. Zhang, H. Jia, "Design of speech corpus for Mandarin text to speech", The Blizzard Challenge Workshop, 2008 |
| Mao, Ching, Lee, "Enhancing Segment-Based Speech Emotion Recognition by Deep Self-Learning", [arXiv 2103.16456](https://arxiv.org/abs/2103.16456) | 9,600 utterances, 4 speakers (2 male, 2 female), 6 emotions; 7,200 utterances of 300 sentences used | [26] the same paper, given as "Proc. the 4th Workshop on Blizzard Challenge, 2005" |
| Zhao et al., "Layer-Adapted Implicit Distribution Alignment Networks for Cross-Corpus Speech Emotion Recognition", [arXiv 2310.03992](https://arxiv.org/abs/2310.03992) | 4 speakers (2 male, 2 female), 1,200 samples, 300 sentences per speaker | [27] the same paper, given as "J. Zhang and H. Jia, … The blizzard challenge 2008 workshop, 2008" |
| Wang, "Speech Emotion Recognition Based on Multi-feature and Multi-lingual Fusion", [arXiv 2001.05908](https://arxiv.org/abs/2001.05908) | recorded by the Institute of Automation; 4 professional speakers, six emotions (five are listed), 9,600 utterances | none |
| "跨语言语料库的语音情感识别对比研究", Journal of Nanjing University (Natural Science) 55(5), 2019, [doi:10.13232/j.cnki.jnju.2019.05.008](https://jns.nju.edu.cn/article/2019/0469-5097/0469-5097-2019-55-5-765.shtml) | Institute of Automation; 4 speakers, 6 emotions, 50 sentences per emotion with the same text | [25] S. Pan, J. Tao, Y. Li, "The CASIA audio emotion recognition method for audio/visual emotion challenge 2011", ACII 2011 (a method paper; we did not read it) |

The Blizzard Challenge 2008 was the fourth Blizzard Challenge, so "4th Workshop …
2005" in the second row appears to mix two details. The numbers in the table
differ partly because papers describe the full corpus (9,600), a subset they used
(7,200), or the 1,200-clip version.

If you use the corpus, the ChineseLDC catalogue entry (CLDC-SPC-2005-010)
describes it most directly. None of this changes how the recipe itself works.

## Provenance

See the warning above. The recipe uses an
**unofficial Hugging Face mirror** (`Bill13579/casia-mirror`, pinned to commit
`a6699530434a32a5ce8f49e037016ba58e934abd`, `CASIA.zip`, SHA-256 checked). The
mirror's licence field is `other` and states no terms; check the terms of the
original corpus before redistributing or using it commercially.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CASIA` — the directory holding the extracted `CASIA/` wav files, or where to download them to
- `CASIA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `CASIA.zip` from the mirror above, checks its SHA-256
and extracts the wav files. `create_dataset`, `remove_long_short` and
`prepare_labels` together took about 77 seconds on one run (the archive was
already downloaded; its download time was not measured).

## Splits

The corpus publishes no partition. Because there are only four speakers, whole
speakers are held out (`n_test_speakers`, `n_valid_speakers`, `split_seed` in
`dataset/config.yaml`): one speaker for test, one for valid and two for training.
Scores are therefore speaker-independent, but the sentences are shared with the
training set. Every class has the same count in each split.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 600 | 600 |
| valid | 300 | 300 |
| test | 300 | 300 (not filtered) |

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
