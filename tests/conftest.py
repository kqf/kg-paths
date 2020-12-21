import pytest
import pandas as pd

from models.data import flatten


@pytest.fixture
def data(size=320):
    return pd.DataFrame({
        "text": ["1 2 3 4 5", ] * size
    })


@pytest.fixture
def flat_data(data):
    return flatten(data["text"])


@pytest.fixture
def oov(size=320):
    return pd.DataFrame({
        "text": ["6 7 8 9 10", ] * size
    })


@pytest.fixture
def flat_oov(oov):
    return flatten(oov["text"])
