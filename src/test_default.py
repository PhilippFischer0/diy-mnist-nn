import src
import numpy as np


def test_foo():
    assert True

def test_forward():
    x = np.random.uniform(0, 10, 20)

    FF = src.FeedForward(20, 1024, 5)
    out = FF(x)

    assert len(out) == 5
    assert sum(np.round(out, 9)) == 1

def test_2dim_forward():
    x = np.random.uniform(0, 10, (3, 20))

    FF = src.FeedForward(20, 1024, 5)
    out = FF(x)

    assert out.shape == (3, 5)
    assert sum(sum(np.round(out, 9))) == 1