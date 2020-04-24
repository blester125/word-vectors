from word_vectors.utils import _find_max


def test_max():
    words = ["A" * i for i in range(10)]
    gold_len = 9
    max_len = _find_max(words)
    assert max_len == gold_len


def test_max_bytes():
    words = ["a", "🙄", "c"]
    str_max = len(max(words, key=lambda x: len(x)))
    gold = 4
    res = _find_max(words)
    assert res == gold
    assert res > str_max
