import click
import numpy as np
import pandas as pd
import networkx as nx

from models.data import read_data
from sklearn.feature_extraction.text import CountVectorizer


class WeightedShortestPath:

    def __init__(self, min_df=1):
        self.vec = CountVectorizer(min_df=1)

        # Initialize the graph with Nones
        self.graph = None

    def fit(self, X, y=None):
        # freqs [n_users, n_items]
        freqs = self.vec.fit_transform(X)

        # co-occurrence matrix [n_items, n_items]
        coo = (freqs.T @ freqs)

        # remove the self-interaction
        coo.setdiag(0)

        # this is a heavy operation for large graphs
        coo = coo.todense()

        # memorize zero elements
        is_zero = coo == 0

        # convert to floats
        coo = coo.astype(float)

        # regularize the missing values
        coo[is_zero] = 1e-6

        # probability to go from P(B| A)
        probas = coo / coo.sum(axis=-1)

        # this is the heaviest operation
        self.graph = nx.DiGraph(probas)
        self.stoi = np.vectorize(self.vec.vocabulary_.__getitem__)
        self.inverse_vocab = {v: k for k, v in self.vec.vocabulary_.items()}
        self.itos = np.vectorize(self.inverse_vocab.__getitem__)
        return self

    def transform(self, X):
        outputs = []
        for pair in X["text"]:
            try:
                start, stop = self.stoi(pair)
            except KeyError:
                outputs.append([])
                continue
            seq = nx.shortest_path(self.graph, start, stop)
            outputs.append(self.itos(seq))
        return outputs


@click.command()
@click.option(
    "--path", type=click.Path(exists=True), default="data/processed/")
def main(path):
    train, valid = read_data(path)
    model = WeightedShortestPath(min_df=2)
    model.fit(train["text"].sample(frac=0.001).values)

    # Prepare the dataset for qualitative validation
    splitted = valid["text"].str.split()

    # Take the first and the last token of the real sequence as inputs
    pairs = np.stack([splitted.str[0].values, splitted.str[-1].values]).T

    # Keep the goold column to compare them with the original results
    evaluation = pd.DataFrame({
        "text": pairs.tolist(),
        "gold": splitted.values.tolist()
    })
    evaluation["preds"] = model.transform(evaluation)

    # Just compare by eyes
    print(evaluation[evaluation["preds"].str.len() > 0])


if __name__ == '__main__':
    main()
