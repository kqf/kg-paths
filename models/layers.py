import torch


class AdditiveAttention(torch.nn.Module):
    def __init__(self, k_dim, q_dim, v_dim):
        super().__init__()
        self._fck = torch.nn.Linear(k_dim, v_dim, bias=False)
        self._fcq = torch.nn.Linear(k_dim, v_dim, bias=True)
        self._fcv = torch.nn.Linear(v_dim, 1)
        self._sig = torch.nn.Sigmoid()

    def forward(self, k, q, v, mask=None):
        energy = self._fcv(self._sig(self._fck(k) + self._fcq(q)))

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e9)

        p_atten = torch.softmax(energy, dim=1)
        return torch.sum(p_atten * v, dim=1, keepdim=True), p_atten


class AttentiveModel(torch.nn.Module):
    def __init__(self, vocab_size, emb_dim=100, pad_idx=0):
        super().__init__()
        self._emb = torch.nn.Embedding(
            vocab_size,
            emb_dim,
            padding_idx=pad_idx,
            max_norm=1,  # This should help with search
        )
        self._att = AdditiveAttention(emb_dim, emb_dim, emb_dim)
        self._out = torch.nn.Linear(2 * emb_dim, emb_dim)
        self.pad_idx = pad_idx

    def encode(self, inputs):
        mask = self.mask(inputs).unsqueeze(-1)
        embedded = self._emb(inputs) * mask

        sl = embedded[:, [-1], :]
        sg, _ = self._att(embedded, sl, embedded, mask)
        hidden = self._out(torch.cat([sl, sg], dim=-1).squeeze(1))
        return hidden

    def forward(self, inputs):
        return self.encode(inputs) @ self._emb.weight.T

    def mask(self, x):
        return x != self.pad_idx
