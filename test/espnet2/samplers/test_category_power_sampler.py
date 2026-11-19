import pytest

from espnet2.samplers.category_power_sampler import (
    CategoryDatasetPowerSampler,
    CategoryPowerNumElementsSampler,
    CategoryPowerSampler,
)


@pytest.fixture()
def category2utt_file(tmp_path):
    p = tmp_path / "category2utt"
    with p.open("w") as f:
        f.write("c1 utt1 utt2 utt3\n")
        f.write("c2 utt4 utt5 utt6\n")
        f.write("c3 utt7 utt8 utt9")

    return str(p)


@pytest.fixture()
def dataset2utt_file(tmp_path):
    p = tmp_path / "dataset2utt"
    with p.open("w") as f:
        f.write("d1 utt1 utt2 utt3\n")
        f.write("d2 utt4 utt5 utt6 utt7 utt8\n")
        f.write("d3 utt9")

    return str(p)


@pytest.fixture()
def utt2dataset_file(tmp_path):
    p = tmp_path / "utt2dataset"
    with p.open("w") as f:
        f.write("utt1 d1\n")
        f.write("utt2 d1\n")
        f.write("utt3 d1\n")
        f.write("utt4 d2\n")
        f.write("utt5 d2\n")
        f.write("utt6 d2\n")
        f.write("utt7 d2\n")
        f.write("utt8 d2\n")
        f.write("utt9 d3")

    return str(p)


@pytest.fixture()
def shape_file(tmp_path):
    p = tmp_path / "speech_shape"
    with p.open("w") as f:
        f.write("utt1 500\n")
        f.write("utt2 600\n")
        f.write("utt3 400\n")
        f.write("utt4 500\n")
        f.write("utt5 500\n")
        f.write("utt6 550\n")
        f.write("utt7 700\n")
        f.write("utt8 800\n")
        f.write("utt9 900")

    return str(p)


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryPowerSampler(category2utt_file, shape_file, drop_last):
    sampler = CategoryPowerSampler(
        batch_bins=1500,
        shape_files=[shape_file],
        min_batch_size=1,
        max_batch_size=None,
        upsampling_factor=0.5,
        dataset_scaling_factor=1.2,
        drop_last=drop_last,
        category2utt_file=category2utt_file,
        epoch=1,
    )

    assert hasattr(sampler, "__iter__")

    batches = list(sampler)
    assert len(batches) > 0


@pytest.mark.parametrize("drop_last", [True, False])
def test_CategoryDatasetPowerSampler(
    category2utt_file, dataset2utt_file, utt2dataset_file, shape_file, drop_last
):
    sampler = CategoryDatasetPowerSampler(
        batch_bins=1500,
        shape_files=[shape_file],
        min_batch_size=1,
        max_batch_size=None,
        category_upsampling_factor=0.5,
        dataset_upsampling_factor=0.5,
        dataset_scaling_factor=1.2,
        drop_last=drop_last,
        category2utt_file=category2utt_file,
        dataset2utt_file=dataset2utt_file,
        utt2dataset_file=utt2dataset_file,
        epoch=1,
    )

    assert hasattr(sampler, "__iter__")

    batches = list(sampler)
    assert len(batches) > 0


@pytest.fixture()
def uneven_corpora(tmp_path):
    """A big corpus of 60 short utterances and two small ones of 5 and 3."""
    sizes, groups = {}, {}
    for name, count, size in [("big", 60, 400), ("mid", 5, 900), ("small", 3, 1500)]:
        keys = [f"{name}{i}" for i in range(count)]
        groups[name] = keys
        for i, key in enumerate(keys):
            sizes[key] = size + 10 * i
    shape = tmp_path / "speech_shape"
    shape.write_text("".join(f"{k} {v}\n" for k, v in sizes.items()))
    c2u = tmp_path / "category2utt"
    c2u.write_text("".join(f"{c} {' '.join(k)}\n" for c, k in groups.items()))
    return str(shape), str(c2u), sizes, groups


def _numel_sampler(uneven_corpora, **kwargs):
    shape, c2u, _, _ = uneven_corpora
    return CategoryPowerNumElementsSampler(
        batch_bins=kwargs.pop("batch_bins", 5000),
        shape_files=[shape],
        category2utt_file=c2u,
        **kwargs,
    )


def test_CategoryPowerNumElementsSampler_beta_one_is_the_dataset(uneven_corpora):
    _, _, sizes, _ = uneven_corpora
    sampler = _numel_sampler(uneven_corpora, upsampling_factor=1.0)
    drawn = sorted(u for batch in sampler for u in batch)
    assert drawn == sorted(sizes)


def test_CategoryPowerNumElementsSampler_gives_equal_time_at_beta_zero(
    uneven_corpora,
):
    _, _, sizes, groups = uneven_corpora
    sampler = _numel_sampler(uneven_corpora, upsampling_factor=0.0)
    seconds = {c: 0 for c in groups}
    owner = {u: c for c, keys in groups.items() for u in keys}
    for batch in sampler:
        for utt in batch:
            seconds[owner[utt]] += sizes[utt]
    total = sum(seconds.values())
    longest = max(sizes.values())
    for c in groups:
        # one third each, within one utterance
        assert abs(seconds[c] - total / 3) <= longest


def test_CategoryPowerNumElementsSampler_batches_are_length_sorted(
    uneven_corpora,
):
    _, _, sizes, _ = uneven_corpora
    # length_bucket=1 sorts strictly, as NumElementsBatchSampler does
    sampler = _numel_sampler(
        uneven_corpora, upsampling_factor=0.5, batch_bins=4000, length_bucket=1
    )
    for batch in sampler:
        lengths = [sizes[u] for u in batch]
        assert lengths == sorted(lengths, reverse=True)
        if len(batch) > 1:
            # a batch closes at the utterance that exceeds batch_bins: without
            # that longest one, the rest fit
            assert (len(batch) - 1) * lengths[1] <= 4000


def test_CategoryPowerNumElementsSampler_redraws_each_epoch(uneven_corpora):
    first = _numel_sampler(uneven_corpora, upsampling_factor=0.5, epoch=1)
    again = _numel_sampler(uneven_corpora, upsampling_factor=0.5, epoch=1)
    second = _numel_sampler(uneven_corpora, upsampling_factor=0.5, epoch=2)
    assert list(first) == list(again)
    assert list(first) != list(second)
