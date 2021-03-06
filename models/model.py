import torch
import skorch
import random
import numpy as np

from sklearn.pipeline import make_pipeline
from functools import partial

# from models.legacy import SessionGraph
from models.layers import AttentiveModel
from models.dataset import SequenceIterator, build_preprocessor, train_split
from models.seqnet import SeqNet

from irmetrics.topk import recall, rr

SEED = 137
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class DynamicVariablesSetter(skorch.callbacks.Callback):
    def on_train_begin(self, net, X, y):
        vocab = X.fields["text"].vocab
        net.set_params(module__vocab_size=len(vocab))
        # net.set_params(module__pad_idx=vocab["<pad>"])
        net.set_params(criterion__ignore_index=vocab["<pad>"])

        n_pars = self.count_parameters(net.module_)
        print(f'The model has {n_pars:,} trainable parameters')
        print(f'There number of unique items is {len(vocab)}')

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scoring(model, X, y, k, func):
    return func(y, model.predict_proba(X), k=k).mean()


def ppx(model, X, y, entry="train_loss"):
    loss = model.history[-1, entry]
    return np.exp(-loss.item())


def inference(logits, k, device):
    probas = torch.softmax(logits.to(device), dim=-1)
    # Return only indices
    return torch.topk(probas, k=k, dim=-1)[-1].clone().detach()


def build_model(X_val=None, max_epochs=5, k=20):
    preprocessor = build_preprocessor(min_freq=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SeqNet(
        module=AttentiveModel,
        module__vocab_size=30000,
        optimizer=torch.optim.Adam,
        optimizer__lr=0.001,
        optimizer__weight_decay=1e-5,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=max_epochs,
        batch_size=100,
        iterator_train=SequenceIterator,
        iterator_train__shuffle=True,
        iterator_train__sort=False,
        iterator_valid=SequenceIterator,
        iterator_valid__shuffle=False,
        iterator_valid__sort=False,
        train_split=partial(train_split, prep=preprocessor, X_val=X_val),
        device=device,
        predict_nonlinearity=partial(inference, k=k, device=device),
        callbacks=[
            DynamicVariablesSetter(),
            skorch.callbacks.EpochScoring(
                partial(ppx, entry="valid_loss"),
                name="perplexity",
                use_caching=False,
                lower_is_better=False,
            ),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k, func=recall),
                name="recall@20",
                on_train=False,
                lower_is_better=False,
                use_caching=True
            ),
            skorch.callbacks.BatchScoring(
                partial(scoring, k=k, func=rr),
                name="mrr@20",
                on_train=False,
                lower_is_better=False,
                use_caching=True
            ),
            skorch.callbacks.ProgressBar(),
        ]
    )

    full = make_pipeline(
        preprocessor,
        model,
    )
    return full
