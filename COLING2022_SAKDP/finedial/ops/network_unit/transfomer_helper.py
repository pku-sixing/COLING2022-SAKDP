import torch

from finedial.ops.general import model_helper


def create_transformer_encoder_network_from_params(params):
    if 'dim_feedforward' in params:
        dim_feedforward = params.dim_feedforward
    else:
        dim_feedforward = params.hidden_size * 2
    transformer_layer = torch.nn.TransformerEncoderLayer(d_model=params.hidden_size,
                                                         dropout=params.get('dropout', 0.10),
                                                         dim_feedforward=dim_feedforward,
                                                         nhead=params.n_heads)
    transformer_encoder = torch.nn.TransformerEncoder(transformer_layer, num_layers=params.n_layers)
    return transformer_encoder

#
# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask
#
#
# def create_mask(src, tgt):
#     src_seq_len = src.shape[0]
#     tgt_seq_len = tgt.shape[0]
#
#     tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
#     src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)
#
#     src_padding_mask = (src == PAD_IDX).transpose(0, 1)
#     tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
#     return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_encoding_mask(lengths, src_seq_len):
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=lengths.device).type(torch.bool)
    # True是需要被忽略的
    src_padding_mask = ~model_helper.sequence_mask_fn(lengths, maxlen=src_seq_len)
    return src_mask, src_padding_mask