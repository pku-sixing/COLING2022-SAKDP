from transformers import BertTokenizer
import transformers

def get_finedial_sp_tokens():
    res = []
    # 必须要按照顺序
    res += ['<CSK>', '</CSK>', '<HR>', '<RT>', '#NH', '#NF', '#NT']
    return res

def check_version(pretrained_config):
    assert pretrained_config.version == transformers.__version__, '%s-%s' % (
        pretrained_config.pretrained_config.version, transformers.__version__
    )


def get_tokenizer_and_vocab(pretrained_config):
    check_version(pretrained_config)
    plm_model_name = pretrained_config.model_name
    tokenizer = BertTokenizer.from_pretrained(plm_model_name)
    pretrained_vocab = tokenizer.vocab.keys()
    return tokenizer, pretrained_vocab
