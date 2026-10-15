import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self, c, shape):
        super().__init__()
        if len(shape) != 3:
            raise ValueError('RAPF expects x with shape N,modalities,channels,time')
        m, channels, _ = shape
        self.c = c
        self.encoders = nn.ModuleList([nn.Sequential(nn.Conv1d(channels, c['width'], 5, padding=2), nn.ReLU(),
                                                   nn.Conv1d(c['width'], c['width'], 3, padding=1), nn.ReLU(),
                                                   nn.AdaptiveAvgPool1d(1), nn.Flatten()) for _ in range(m)])
        self.quality_heads = nn.ModuleList([nn.Sequential(nn.Linear(c['width'] + 4, 32), nn.ReLU(), nn.Linear(32, 1)) for _ in range(m)])
        self.tokens = nn.Parameter(torch.zeros(m, c['width']))
        self.fusion = nn.Linear(2 * c['width'], c['width'])
        self.classifier = nn.Linear(c['width'], c['classes'])
        self.prototypes = nn.Parameter(torch.randn(c['classes'], c['width']))
        if c['variant'] not in ['full', 'uniform', 'no-tokens', 'no-prototype', 'early', 'late']:
            raise ValueError('Unknown variant')
        self.early = nn.Linear(m * c['width'], c['classes'])
        self.late = nn.ModuleList([nn.Linear(c['width'], c['classes']) for _ in range(m)])

    def forward(self, batch):
        x = batch['x'].float()
        available = batch.get('available', torch.ones(x.shape[:2], device=x.device)).bool()
        if not available.any(1).all():
            raise ValueError('Every window must contain at least one available modality')
        z = torch.stack([encoder(x[:, i]) for i, encoder in enumerate(self.encoders)], 1)
        completed = torch.where(available[..., None], z, self.tokens[None])
        quality = torch.stack([x.var(dim=(-1, -2), unbiased=False),
                               batch.get('sample_missing', torch.zeros_like(available, dtype=x.dtype)),
                               (x.abs() > self.c['clip_threshold']).float().mean((-1, -2))], -1)
        quality = quality * available[..., None]
        scores = torch.cat([head(torch.cat([completed[:, i], quality[:, i], available[:, i:i + 1].float()], 1))
                            for i, head in enumerate(self.quality_heads)], 1)
        if self.c['variant'] == 'uniform':
            scores = torch.zeros_like(scores)
        weights = (scores / self.c['temperature']).masked_fill(~available, -torch.inf).softmax(1)
        content = (weights[..., None] * completed).sum(1)
        context = ((~available)[..., None] * self.tokens[None]).mean(1)
        if self.c['variant'] == 'no-tokens':
            context = torch.zeros_like(context)
        representation = F.relu(self.fusion(torch.cat([content, context], 1)))
        logits = self.classifier(representation)
        if self.c['variant'] == 'early':
            logits = self.early((z * available[..., None]).flatten(1))
        elif self.c['variant'] == 'late':
            local = torch.stack([head(z[:, i]) for i, head in enumerate(self.late)], 1)
            logits = (local * available[..., None]).sum(1) / available.sum(1, keepdim=True)
        distances = torch.cdist(F.normalize(representation, dim=1), F.normalize(self.prototypes, dim=1)).square()
        return dict(logits=logits, prototype=-distances / self.c['prototype_temperature'], weights=weights,
                    representation=representation, available=available)

    def loss(self, batch, c, epoch):
        x = batch['x']
        original = batch.get('available', torch.ones(x.shape[:2], device=x.device)).bool()
        sampled = (torch.rand(x.shape[:2], device=x.device) >= c['missing_rate']) & original
        empty = ~sampled.any(1)
        sampled[empty] = original[empty]
        masked = dict(batch, available=sampled)
        full, partial = self(batch), self(masked)
        loss = .5 * (F.cross_entropy(full['logits'], batch['y']) + F.cross_entropy(partial['logits'], batch['y']))
        if c['variant'] not in ['no-prototype', 'early', 'late']:
            loss += c['prototype_weight'] * .5 * (F.cross_entropy(full['prototype'], batch['y']) + F.cross_entropy(partial['prototype'], batch['y']))
        return loss


def extra_metrics(d, c):
    weights = d['weights']
    return dict(unavailable_weight=float(np.max(np.where(d['available'], 0, weights))),
                fusion_entropy=float(-(weights * np.log(weights.clip(1e-12))).sum(1).mean()))
