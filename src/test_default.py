import src
import numpy as np


def test_forward():
    x = np.random.uniform(0, 10, 20)

    ff = src.FeedForward(20, 1024, 5)
    out = ff(x)

    assert out.shape == (1, 5)
    assert np.isclose(sum(out[0, :]), 1)


def test_2dim_forward():
    x = np.random.uniform(0, 10, (3, 20))

    ff = src.FeedForward(20, 1024, 5)
    out = ff(x)

    print(out)
    print(sum(out[0, :]), sum(out[1, :]), sum(out[2, :]))
    assert out.shape == (3, 5)
    assert np.isclose(sum(sum(out)), 3)
