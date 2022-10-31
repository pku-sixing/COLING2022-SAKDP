import torch
from torch.nn import functional as F


def compute_nlg_loss(logits, labels, pad, unk, unk_learning='none', reduction="mean"):
    if unk_learning == 'none':
        loss = F.nll_loss(logits, labels, ignore_index=pad, reduction=reduction)
    elif unk_learning == 'penalize':
        loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
        is_unk = labels == unk
        loss = torch.where(is_unk, loss / is_unk.sum(), loss).sum() / (labels != pad).sum()
    elif unk_learning == 'skip':
        loss = F.nll_loss(logits, labels, ignore_index=pad, reduction="none")
        is_unk = labels == unk
        loss = torch.where(is_unk, loss * 1e-20, loss).sum() / ((labels != pad).sum() - is_unk.sum())
    else:
        raise NotImplementedError()
    return loss