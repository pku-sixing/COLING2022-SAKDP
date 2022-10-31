from finedial.framework.model.learning2rank.MultiStageRanking import MultiStageRanking


def create_l2r_model(params, vocab):
    if params.model == "MultiStageRanking":
        return MultiStageRanking(params, vocab)
    else:
        raise NotImplementedError()