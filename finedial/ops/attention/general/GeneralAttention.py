"""
实现最常见的Attention操作
Memory Bank Time First：

 Partially Copy from OpenNMT

"""
import torch
from torch import nn

VALID_ATTENTION_TYPES = {"dot", "general", "mlp"}



def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    # print('max_len', max_len)
    # print('lengths', lengths)
    # print('lengths.max()', lengths.max())
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class GeneralAttention(nn.Module):

    def __init__(self, attention_type, query_dim, key_dim, value_dim, hidden_dim):
        """
            * Luong Attention (dot, general):
                * dot: :math:`\text{score}(H_j,q) = H_j^T q`
                * general: :math:`\text{score}(H_j, q) = H_j^T W_a q`

            * Bahdanau Attention (mlp):
                * :math:`\text{score}(H_j, q) = v_a^T \text{tanh}(W_a q + U_a h_j)`
        """
        super(GeneralAttention, self).__init__()

        self.attention_type = attention_type
        assert self.attention_type in VALID_ATTENTION_TYPES, "Invalid Attention Type: %s" % self.attention_type

        self.hidden_dim = hidden_dim
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim

        if self.attention_type == 'general':
            self.general_matrix = nn.Linear(key_dim, query_dim, bias=False)
        elif self.attention_type == "mlp":
            self.query_in = nn.Linear(query_dim, hidden_dim, bias=True)
            self.key_in = nn.Linear(key_dim, hidden_dim, bias=True)
            self.v_out = nn.Linear(hidden_dim, 1, bias=False)
        elif self.attention_type == 'dot':
            assert query_dim == key_dim

    def _score(self, keys, queries):
        """
        Args:
            keys: [batch, bank_len, key_dim]
            queries: [batch, query_num, query_dim]
        Out:
            => query_len, bank_len
            scores: [batch, query_num, bank_len]
        """

        batch_size, bank_len, key_dim = keys.size()
        batch_size, query_num, query_dim = queries.size()

        if self.attention_type == 'general':
            # Out: [batch, query_dim, bank_len]
            attn_in = self.general_matrix(keys).transpose(2, 1)
            # Out: [batch, query_num, bank_len]
            scores = torch.bmm(queries, attn_in)
            return scores
        elif self.attention_type == 'dot':
            # Out: [batch, query_dim, bank_len]
            attn_in = keys.transpose(2, 1)
            # Out: [batch, query_num, bank_len]
            try:
                scores = torch.bmm(queries, attn_in)
            except:
                print(1)
            return scores
        elif self.attention_type == 'mlp':
            # Out: [batch, query_num, hidden_dim]
            query_in = self.query_in(queries)
            query_in = query_in.view(batch_size, query_num, 1, self.hidden_dim)
            query_in = query_in.expand(batch_size, query_num, bank_len, self.hidden_dim)

            # Out:  [batch, bank_len, key_dim]
            key_in = self.key_in(keys)
            key_in = key_in.contiguous().view(-1, self.hidden_dim)
            key_in = key_in.view(batch_size, 1, bank_len, self.hidden_dim)
            key_in = key_in.expand(batch_size, query_num, bank_len, self.hidden_dim)

            # Out:  [batch, query_num, bank_len, key_dim]
            sum_in = torch.tanh(query_in + key_in)

            sum_in = sum_in.view(-1, self.hidden_dim)
            scores = self.v_out(sum_in).view(batch_size, query_num, bank_len)
            return scores

    def forward(self, queries, keys, values, bank_length, valid_bank_start_index=0):
        """
            Batch First
            Args:
                queries: [batch, query_num, query_dim] or [batch, query_dim]
                keys: [batch, bank_len, key_dim]
                values: [batch, query_num, value_dim]
                bank_length: [batch]
                valid_bank_start_index : int, 这些Attention，从什么位置开始是有效的，例如1开始可以屏蔽第一个符号的Attention值。

            Returns:
                attention_vector: [batch, query_num, dim] or [batch, dim]
                align_distributions: [batch, query_num, bank_len] or [batch, bank_len]

        """
        if queries.dim() == 2:
            one_step_mode = True
            queries = queries.unsqueeze(1)
        else:
            one_step_mode = False

        # [batch, query_num, bank_len]
        align_scores = self._score(keys, queries)

        # 将不符合的长度进行屏蔽
        if bank_length is not None:
            mask = sequence_mask(bank_length, max_len=align_scores.size(-1))
            mask = mask.unsqueeze(1)
            batch, query_num, bank_len = align_scores.size()
            mask = mask.expand(batch, query_num, bank_len)
            align_scores.masked_fill_(~mask, -float('inf'))


        if valid_bank_start_index > 0:
            for idx in range(0, valid_bank_start_index):
                align_scores[:, :, 0] = -float('inf')

        align_distributions = torch.softmax(align_scores, -1)
        # In: [batch, query_num, bank_len], [batch, bank_len, dim]
        # Out： [batch, query_num, dim]
        attention_vector = torch.bmm(align_distributions, values)

        if self.attention_type in ["general", "dot"]:
            attention_vector = torch.tanh(attention_vector)

        if one_step_mode:
            attention_vector = attention_vector.squeeze(1)
            align_distributions = align_distributions.squeeze(1)

        return attention_vector, align_distributions








