import torch

from finedial.ops.general import selection_helper


def sample_tokens_from_dist(word_sample_param, next_step, tgt, current_token_dist, diverse_rate=0.0):
    word_sample_mode = word_sample_param['decoder_mode']
    if word_sample_mode == 'forward':
        assert diverse_rate == 0
        next_token = tgt[next_step]
        next_token_score = torch.zeros_like(next_token)
    elif word_sample_mode == 'greedy':
        assert diverse_rate == 0
        top_1 = current_token_dist.max(-1)
        next_token = top_1.indices
        next_token_score = top_1.values
    elif word_sample_mode == 'beam_top_k':
        top_k = word_sample_param['beam_width']
        top_k_candidates = current_token_dist.topk(top_k)
        candidates = top_k_candidates.indices
        scores = top_k_candidates.values
        next_token = candidates
        next_token_score = scores
        if diverse_rate > 0:
            offsets = torch.arange(0, top_k, 1, device=current_token_dist.device, dtype=torch.float32) * - diverse_rate
            offsets = offsets.unsqueeze(0)
            batch_size = current_token_dist.size()[0]
            offsets = torch.repeat_interleave(offsets, batch_size, 0)
            next_token_score += offsets
    elif word_sample_mode == 'sample_k':
        assert diverse_rate == 0
        top_k = word_sample_param['top_k']
        top_k_candidates = current_token_dist.topk(top_k)
        candidates = top_k_candidates.indices
        scores = top_k_candidates.values
        # 这里是inclusive的
        random_index = torch.randint(0, top_k, [candidates.size()[0]])
        next_token = selection_helper.batch_index_select_2d(candidates, random_index)
        next_token_score = selection_helper.batch_index_select_2d(scores, random_index)
    else:
        raise NotImplementedError()
    return next_token, next_token_score
