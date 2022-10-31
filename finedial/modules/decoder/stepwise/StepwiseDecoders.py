"""
用于处理Sequential 数据的解码
Time First
"""
from finedial.modules.decoder.stepwise.StepwiseDFARNNDecoder import StepwiseDFARNNDecoder
from finedial.modules.decoder.stepwise.StepwiseDecoupledRNNDecoder import StepwiseDecoupledRNNDecoder
from finedial.modules.decoder.stepwise.StepwiseRNNDecoder import StepwiseRNNDecoder
from finedial.modules.decoder.stepwise.StepwiseRNNDecoderSingleChannel import StepwiseRNNDecoderSingleChannel
from finedial.modules.decoder.stepwise.StepwiseReferenceDecoderWrapper import StepwiseReferenceDecoder


def create_stepwise_decoder_from_params(params, memory_agents, reference_agents=None):
    """
       需要给出类似 params.stepwise_decoder
    """
    if params.stepwise_decoder_type == 'stepwise_rnn_decoder':
        decoder = StepwiseRNNDecoder(params, memory_agents)
    elif params.stepwise_decoder_type == 'stepwise_decoupled_rnn_decoder':
        decoder = StepwiseDecoupledRNNDecoder(params, memory_agents)
    elif params.stepwise_decoder_type == 'stepwise_dfa_rnn_decoder':
        decoder = StepwiseDFARNNDecoder(params, memory_agents)
    elif params.stepwise_decoder_type == 'stepwise_rnn_decoder_single_channel':
        decoder = StepwiseRNNDecoderSingleChannel(params, memory_agents)
    elif params.stepwise_decoder_type == 'stepwise_reference_decoder':
        decoder = StepwiseReferenceDecoder(params, memory_agents, reference_agents)
    else:
        raise NotImplementedError()
    return decoder

