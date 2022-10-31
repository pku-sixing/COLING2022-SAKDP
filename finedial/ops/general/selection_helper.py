import torch


def batch_index_select_2d(candidates, index):
    """
        Args:
            candidates: [batch, candidates]
            index:[index] or [batch, index]
    """
    batch_size, candidate_num = candidates.size()

    index_offset = torch.arange(0, batch_size, dtype=torch.long) * candidate_num
    index_offset = index_offset.to(index.device)

    if index.dim() == 2:
        index_offset = index_offset.unsqueeze(1)
    return candidates.take(index + index_offset)


def batch_index_select_3d(candidates, index):
    """
        Args:
            candidates: [batch, candidates, dims]
            index:[index]
    """
    batch_size, candidate_num, dims = candidates.size()
    index_offset = torch.arange(0, batch_size, dtype=torch.long, device=index.device) * candidate_num
    candidates = candidates.view(batch_size * candidate_num, dims)
    if index.dim() == 2:
        index_offset = index_offset.unsqueeze(1)
    index = index + index_offset
    index = index.view(-1)
    return candidates.index_select(index=index, dim=0)

