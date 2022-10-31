from collections import Counter, OrderedDict, defaultdict
from finedial.utils.logging.logger_helper import logger
from torchtext.vocab import Vocab

def load_vocab(vocab_path, special_tokens=[], placeholder_tokens=0, enable_duplication=False, reset_unk=None):
    """
     创建词表，
        Args:
            vocab_path: 需要给定的是已经排序好，确定好词表的文件，这里仅增加特殊符号
            special_tokens: 哪些特殊符号
            placeholder_tokens: 哪些需要创建的特殊符号，以及对应的最大数量。
            enable_duplication: 多个词表之间是否允许重复
    """
    counter = Counter()
    assert vocab_path is not None
    if isinstance(vocab_path, str):
        with open(vocab_path, 'r+', encoding='utf-8') as fin:
            vocabs = [x.strip('\r\n') for x in fin.readlines()]
            assert len(vocabs) == len(set(vocabs))
    elif isinstance(vocab_path, list):
        vocabs = []
        vocab_set = set()
        for sub_vocab_path in vocab_path:
            with open(sub_vocab_path, 'r+', encoding='utf-8') as fin:
                sub_vocabs = [x.strip('\r\n') for x in fin.readlines()]
                if not enable_duplication:
                    assert len(vocab_set - set(sub_vocabs)) == 0
                    assert len(set(sub_vocabs)) == len(sub_vocabs)
                vocabs += sub_vocabs
    else:
        raise NotImplementedError()

    vocabs += ['<%s_%d>' % ('dmc', x) for x in range(placeholder_tokens)]
    vocab_size = len(vocabs)
    for idx, token in enumerate(vocabs):
        counter[token] = vocab_size - idx

    specials = list(OrderedDict.fromkeys(tok for tok in special_tokens if tok is not None))
    vocab = Vocab(counter, specials=specials)
    if reset_unk is None:
        return vocab
    # 增加
    unk_index = vocab.stoi[reset_unk]
    new_stoi = defaultdict(lambda :unk_index)
    for k, v in vocab.stoi.items():
        new_stoi[k] = v
    vocab.stoi = new_stoi
    return vocab


def load_vocab_from_list(vocabs, sp_tokens=None):
    """
     创建词表，
        Args:
            vocab: 已经确定好的词表
    """
    counter = Counter()
    vocab_size = len(vocabs)
    vocabs = list(vocabs)
    if sp_tokens is not None:
        vocab_set = set(vocabs)
        vocab_dict = dict()
        for word in vocabs:
            vocab_dict[word] = len(vocab_dict)
        unused_id = 0
        res = ''
        for sp_token in sp_tokens:
            if sp_token not in vocab_set:
                unused_id += 1
                uid = '[unused%d]' % unused_id
                assert uid in vocab_dict, 'unused id is not enough : %s, vocablist=%s' % (uid, str(vocab_dict))
                vocabs[vocab_dict[uid]] = sp_token
                vocab_dict[sp_token] = unused_id
                res += '%s->%s ' % (uid, sp_token)
                vocab_set.add(sp_token)
        logger.info('[SP_TOKEN MAP] %s' % res)
    for idx, token in enumerate(vocabs):
        counter[token] = vocab_size - idx
    return Vocab(counter, specials=[])

