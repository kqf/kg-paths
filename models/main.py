import click
import numpy as np
import pandas as pd

from models.data import read_data, flatten, evaluate
from models.model import build_model


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, valid = read_data(path)
    X = flatten(train["text"])
    X_val = flatten(valid["text"])

    model = build_model(X_val=X_val, max_epochs=3)
    model.fit(X)

    # Check the language model
    evaluate(model, X, "train")
    evaluate(model, X_val, "valid")

    # Prepare the dataset for qualitative validation
    splitted = valid["text"].str.split()

    # Take the first and the last token of the real sequence as inputs
    pairs = np.stack([splitted.str[0].values, splitted.str[-1].values]).T

    # Keep the goold column to compare them with the original results
    evaluation = pd.DataFrame({
        "text": pairs.tolist(),
        "gold": splitted.values.tolist()
    }).sample(frac=0.001)
    evaluation["preds"] = model.transform(evaluation)

    # Just compare by eyes
    print(evaluation[evaluation["preds"].str.len() > 0])


if __name__ == '__main__':
    main()
