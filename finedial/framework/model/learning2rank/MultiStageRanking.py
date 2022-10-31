import random

from torch import nn
import torch
from transformers import BertModel

from finedial.modules.encoder.sequential.SequentialEncoder import SequentialEncoder
from finedial.ops.network_unit import projection_helper
from finedial.utils.data_utils.param_helper import ParamDict
from torch.nn.modules.loss import MarginRankingLoss


class MultiStageRanking(nn.Module):

    def __init__(self, params, token_to_id):
        """
            数据说明
            SRC : 正样本 Positive Samples
            TGT : 负样本 Negative Samples
            TGT : 负样本 Negative Samples
        """
        super(MultiStageRanking, self).__init__()
        assert params.model == 'MultiStageRanking'
        assert params.dataset.pair_wise_input_mode, 'this config should be True for ensuring the input'

        self.token_to_id = token_to_id
        self.sos = token_to_id.sos
        self.eos = token_to_id.eos
        self.unk = token_to_id.unk
        self.pad = token_to_id.pad

        self.token_embeddings = []
        self.trans_embeddings = []

        # Pretrained 加载Pretrain的模型
        self.pretrained_encode_mode = params.learning2rank.get('pretrained_encode_mode', None)
        self.padding_len = params.learning2rank.get('padding_len', )
        self.pretrained_config = False
        pretrained_models = dict()
        if self.pretrained_encode_mode is not None:
            for key, config in params.learning2rank.pretrained_lm_configs.items():
                if key not in ['query']:
                    continue
                plm_model_name = config.model_name
                model = BertModel.from_pretrained(plm_model_name)
                if config.fixed_param:
                    for param in model.parameters():
                        param.requires_grad = False
                pretrained_models[key] = model
            self.pretrained_models = nn.ModuleDict(pretrained_models)
        else:
            raise NotImplementedError()

        self.query_embedding = pretrained_models['query']
        self.query_encoder = SequentialEncoder(params.learning2rank.query_encoder)

        self.sample_num_in_train_batch = params.learning2rank.get('sample_num_in_batch', 9999)

        bert_dim = self.query_encoder.output_size
        output_dim = params.learning2rank.output_dim
        self.mid_representation = params.learning2rank.mid_representation
        # 分别创建两个网络，用于创建语义表示
        self.query_projection = projection_helper.create_projection_layer(
            in_dim=bert_dim,
            out_dim=output_dim,
            bias=False,
            activation='none',
        )

        self.multi_stage_discriminator = nn.Sequential(
            nn.Dropout(params.learning2rank.get('dropout', 0.0)),
            nn.Linear(output_dim, output_dim // 2),
            nn.Tanh(),
            nn.Linear(output_dim // 2, 1, bias=False),
            nn.Sigmoid()
        )

        self.loss_fn = MarginRankingLoss(margin=params.learning2rank.get('margin'))
        self.loss_terms = params.training['loss_terms']

    def get_token_embeddings(self):
        """
            得到需要初始化的Embedding，（field_name, embedding）
        """
        return self.token_embeddings

    def get_trans_embeddings(self):
        """
            得到需要初始化的Embedding，（field_name, embedding）
        """
        return self.trans_embeddings

    def get_inputs_from_batch(self, batch, infer_mode):
        """
        获得数据的输入
        """
        # print(batch.dataset)
        src = batch.src[0]
        src_len = batch.src[1]
        batch_size = src.size(1)
        tgt = batch.tgt[0]
        tgt_len = batch.tgt[1]
        if not infer_mode:
            res = {
                'src': src,
                'tgt': tgt,
                'src_len': src_len,
                'tgt_len': tgt_len,
                'batch_size': batch_size,
                # 'dynamic_vocab_projections' : dynamic_vocab_projections,
            }

            neg_neg_sample = batch.__dict__.get('neg_neg_sample', None)
            if neg_neg_sample is not None:
                res['neg2'] = neg_neg_sample[0]
                res['neg2_len'] = neg_neg_sample[1]
            return ParamDict(res)
        else:
            res = {
                'src': src,
                'tgt': tgt,
                'src_len': src_len,
                'tgt_len': tgt_len,
                'batch_size': batch_size,
                # 'dynamic_vocab_projections': dynamic_vocab_projections,
            }
            neg_neg_sample = batch.__dict__.get('neg_neg_sample', None)
            if neg_neg_sample is not None:
                res['neg2'] = neg_neg_sample[0]
                res['neg2_len'] = neg_neg_sample[1]
            return ParamDict(res)

    def batch_level_train(self, batch_dict, epoch_num):
        """
           绝对不能复用已经计算过的结果，不然会出错
            Return:
                states: [layers, batch, dim]
                summaries: [batch, dim]
                memories: [batch, seq_len, dim]
        """
        my_inputs = self.get_inputs_from_batch(batch_dict, infer_mode=False)

        # to batch_first
        pos_input = my_inputs.src.transpose(0, 1)
        neg_input = my_inputs.tgt.transpose(0, 1)

        batch_len, seq_len = pos_input.size()
        if self.sample_num_in_train_batch < batch_len and self.training:
            random_ids = random.sample(list(range(batch_len)), self.sample_num_in_train_batch)
            random_ids = torch.LongTensor(random_ids).to(pos_input.device)
            pos_input = torch.index_select(pos_input, dim=0, index=random_ids)
            neg_input = torch.index_select(neg_input, dim=0, index=random_ids)

        batch_size, _ = pos_input.size()

        def SEP_encoder(embed_input, token_input, CSK_POS=3):
            batch_len, seq_len, dim = embed_input.size()
            batch_2, seq_len = token_input.size()
            token_mask = token_input == CSK_POS
            arange_index = torch.arange(0, seq_len, device=token_mask.device).unsqueeze(0).repeat(batch_len, 1)
            pos_index = (token_mask * arange_index).sum(-1)
            # print(pos_index)
            pos_offset = torch.arange(0, batch_len, device=embed_input.device) * seq_len
            pos_index = pos_index + pos_offset
            flatten_embed = embed_input.view(batch_len * seq_len, -1)
            embed = torch.index_select(flatten_embed, 0, pos_index)
            return embed

        # Must Padding
        def padding(x):
            batch_size, seq_len = x.size()
            assert seq_len < self.padding_len
            paadings = torch.zeros([batch_size, self.padding_len - seq_len], device=x.device, dtype=torch.long)
            return torch.cat([x, paadings], -1)

        pos_input = padding(pos_input)
        neg_input = padding(neg_input)

        pos_embed = self.query_embedding(pos_input)[0]
        neg_embed = self.query_embedding(neg_input)[0]
        pos_embed = SEP_encoder(pos_embed, pos_input)
        neg_embed = SEP_encoder(neg_embed, neg_input)

        if self.mid_representation:
            pos_embed = self.query_projection(pos_embed)
            neg_embed = self.query_projection(neg_embed)

        pos_scores = self.multi_stage_discriminator(pos_embed).view(batch_size)
        neg_scores = self.multi_stage_discriminator(neg_embed).view(batch_size)

        pos_neg = self.loss_fn(pos_scores, neg_scores, torch.ones_like(pos_scores))
        scores = pos_neg
        valid_score_dict = {
            'pos_neg': pos_neg.mean()
        }
        return scores.mean(), valid_score_dict, 1

    def infer_batch(self, batch_dict):
        """
            Return:
                states: [layers, batch, dim]
                summaries: [batch, dim]
                memories: [batch, seq_len, dim]
        """
        self.eval()
        my_inputs = self.get_inputs_from_batch(batch_dict, infer_mode=False)

        # to batch_first
        pos_input = my_inputs.src.transpose(0, 1)
        batch_size, _ = pos_input.size()

        def SEP_encoder(embed_input, token_input, CSK_POS=3):
            batch_len, seq_len, dim = embed_input.size()
            batch_2, seq_len = token_input.size()
            token_mask = token_input == CSK_POS
            arange_index = torch.arange(0, seq_len, device=token_mask.device).unsqueeze(0).repeat(batch_len, 1)
            pos_index = (token_mask * arange_index).sum(-1)
            # print(pos_index)
            pos_offset = torch.arange(0, batch_len, device=embed_input.device) * seq_len
            pos_index = pos_index + pos_offset
            flatten_embed = embed_input.view(batch_len * seq_len, -1)
            embed = torch.index_select(flatten_embed, 0, pos_index)
            return embed

        # 必须要Padding，否则失效
        def padding(x):
            batch_size, seq_len = x.size()
            assert seq_len < self.padding_len
            paadings = torch.zeros([batch_size, self.padding_len - seq_len], device=x.device, dtype=torch.long)
            return torch.cat([x, paadings], -1)

        pos_input = padding(pos_input)

        pos_embed = self.query_embedding(pos_input)[0]
        pos_embed = SEP_encoder(pos_embed, pos_input)

        if self.mid_representation:
            pos_embed = self.query_projection(pos_embed)

        pos_scores = self.multi_stage_discriminator(pos_embed).view(batch_size)
        return pos_scores, None
