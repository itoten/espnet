# Multi-corpus speech emotion recognition (full20)

Trains `wavlmasp` (fine-tuned [WavLM
Large](https://github.com/microsoft/unilm/tree/master/wavlm) + attentive
statistics pooling + a fully connected head, the [Interspeech 2025 SER in
Naturalistic Conditions challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.html)
baseline) on the union of ten corpora — CREMA-D, EMNS, EmoDB, Emozionalmente,
eNTERFACE'05, JL-Corpus, MELD, Quechua Collao, SUBESCO and Thorsten — and
scores it on each corpus's own test split plus MSP-Podcast Test1/Test2, which
this recipe never trains on.

This is the **full20** sibling of `egs3/multicorpus_core7/wavlmasp`. Instead
of restricting labels to the 7 classes shared with the IS2025 baseline, every
label a corpus carries becomes its own class — 20 in total — with only
MSP-Podcast's `contempt` dropped, since none of the ten training corpora ever
produce it. The two recipes are meant to be compared: does keeping minor,
corpus-specific labels (`excited`, `boredom`, `calm`, `sleepy`, `amused`,
`drunk`, `whisper`, `sarcastic`, JL-Corpus's five secondary emotions) help or
hurt, especially on the unseen MSP-Podcast domain? See
`espnet_work/multicorpus_ser_plan.md` §1.1 for the full class list and counts,
and `espnet_work/label_inventory.md` for the per-corpus breakdown.

Everything else — how the merge works, environment variables, quick start —
is identical to `egs3/multicorpus_core7/wavlmasp/readme.md`; only
`MULTICORPUS_FULL20_OUTPUT` (in place of `MULTICORPUS_CORE7_OUTPUT`) and
`model.num_classes: 20` differ.

## Notes

- Class imbalance is far more severe than in `core7`: `sarcastic` contributes
  112 training utterances against `neutral`'s 9,480, roughly an 85x spread.
  `class_weights` (inverse label frequency) is applied as-is for a first run;
  whether that is enough to train stably is left to be checked from the
  actual run rather than assumed up front.
- When scoring on MSP-Podcast, this model can predict any of its 20 classes,
  including ones MSP-Podcast's own labels never use (e.g. `excited`,
  `boredom`). That is not a bug to work around — it is the comparison this
  recipe exists to run: whether a wider label vocabulary learned from other
  corpora leaks into errors on an unseen domain.
