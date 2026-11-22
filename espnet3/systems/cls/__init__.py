"""Utterance classification over pre-trained or from-scratch speech encoders.

``cls`` trains its own model (``src/model.py``: an encoder frontend,
attentive statistics pooling and a classification head) and runs it with
``src/inference_model.py``. ``esp2_cls`` trains ESPnet2's classification
task instead. The two share the data stages, the metrics and the data
helpers, which live in ``esp2_cls``.
"""
