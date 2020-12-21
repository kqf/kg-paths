import click
import pathlib

import datetime
import numpy as np
import pandas as pd


def read_data(raw):
    path = pathlib.Path(raw)
    return (
        pd.read_csv(path / 'train.txt', names=["text"]),
        pd.read_csv(path / 'valid.txt', names=["text"]),
    )


def read_file(path, filename, frac=None):
    df = pd.read_csv(
        pathlib.Path(path) / filename,
        sep=',',
        header=None,
        names=['session_id', 'time', 'item_id'],
        usecols=[0, 1, 2],
        dtype={0: np.int32, 1: str, 2: np.int64},
    )
    df["time"] = pd.to_datetime(df["time"])
    # The original data preprocessing:
    df = remove_short(df, "session_id")
    df = remove_short(df, "item_id", 5)
    df = remove_short(df, "session_id")

    # Ensure the data is in the right order
    df = df.sort_values(["session_id", "time"]).reset_index()
    return df


def remove_short(data, col="session_id", min_size=1):
    lengths = data.groupby(col).size()
    return data[np.in1d(data[col], lengths[lengths > min_size].index)]


def sample(data, frac=None, col="session_id"):
    if frac is None:
        return data

    n_samples = int(len(data) * frac)
    train = data.index > data.index.max() - n_samples

    old = data.loc[~train, col].unique()
    recent = data.loc[train, col].unique()

    # Take only the most recent data
    return data[np.in1d(data[col], list(set(recent) - set(old)))]


def build_sessions(
    df,
    session_col="session_id",
    item_col="item_id",
    min_len=1,
):
    df[item_col] = df[item_col].astype(str)
    sess = df.groupby(session_col)[item_col].apply(list)
    return sess


def dump(sessions, path):
    cleaned = sessions.apply(" ".join)
    cleaned.to_csv(path, index=False, header=None)


@click.command()
@click.option("--raw", type=click.Path(exists=False))
@click.option("--out", type=click.Path(exists=False))
@click.option("--train", default='yoochoose-clicks.dat')
def main(raw, out, train):
    train = read_file(raw, train)

    # Take the last day for validation
    split_day = train["time"].max() - datetime.timedelta(days=1)
    valid = train[train["time"] >= split_day]
    train = train[train["time"] < split_day]

    train = sample(train, 1. / 64)

    # Make sure validation has the same nodes
    valid = valid[np.in1d(valid["item_id"], train["item_id"])]
    valid = remove_short(valid, "session_id")

    train_sessions = build_sessions(train)
    valid_sessions = build_sessions(valid)

    opath = pathlib.Path(out)
    opath.mkdir(parents=True, exist_ok=False)

    dump(train_sessions, opath / "train.txt")
    dump(valid_sessions, opath / "valid.txt")


if __name__ == '__main__':
    main()
