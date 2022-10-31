class StepwiseDecoderState:
    def __init__(self, state=None, input=None, memory_dict=None, dyanmic_projections=None):
        self.state = state
        self.input = input
        self.memory_dict = memory_dict
        self.dynamic_vocab_projections = dyanmic_projections
        self.token_prob_dist = None
        self.token_mode_order = None
        self.channel_token_prob_dist = None
        self.global_relevance_gates = None
