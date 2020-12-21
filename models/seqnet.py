import torch
import skorch
import numpy as np

from tqdm import tqdm


def decode(module, start, stop):
    # This variable will be is an update variable
    path_ = start.unsqueeze(-1).unsqueeze(-1)

    # If start and stop are the same token
    if start == stop:
        return torch.cat([path_, stop.unsqueeze(-1).unsqueeze(-1)], axis=-1)

    # Convert stop token to a vector
    with torch.no_grad():
        stopv = module._emb(stop).unsqueeze(-1)

    # Stop when path_ contains all the nodes
    while path_.shape[-1] < module._emb.weight.T.shape[-1]:
        with torch.no_grad():
            hidden = module.encode(path_)
            # The main part is here:
            # module._emb.weight.T -- is the [emb_dim, vocab_size] matrix
            # we want next token to be similar both to hidden vector of the
            # given path and to the stop token.
            # This heuristic should drive us closer to the stop token
            search_idx = module._emb.weight.T - stopv

            # normalize the resulting vectors to unit length
            norm = search_idx / search_idx.norm(p=2, dim=-1, keepdim=True)
            scores = hidden @ norm

        # Remove the scores that are
        scores[:, path_] = -float('inf')
        most_similar = scores.argmax(dim=-1, keepdim=True)

        # Always update path_
        path_ = torch.cat([path_, most_similar], axis=-1)
        if torch.squeeze(most_similar) == stop:
            break

    # Convert the matrix to a vector
    return torch.squeeze(path_)


class SeqNet(skorch.NeuralNet):
    def predict(self, X):
        # Now predict_proba returns top k indexes
        indexes = self.predict_proba(X)
        return np.take(X.fields["text"].vocab.itos, indexes)

    def transform(self, X):
        outputs = []
        batches = self.iterator_valid(X, self.batch_size,
                                      device=self.device, shuffle=False)
        with tqdm(total=len(X)) as progress_bar:
            for batch, _ in batches:
                for start, stop in batch:
                    seq = decode(self.module_, start, stop).cpu().numpy()
                    outputs.append(np.take(X.fields["text"].vocab.itos, seq))
                    progress_bar.update()
        return outputs
