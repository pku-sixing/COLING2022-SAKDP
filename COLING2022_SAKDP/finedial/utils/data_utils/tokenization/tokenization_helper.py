import jieba


def default_fn(text):
    return text.strip('\r').split()


def sub_text_fn(text):
    res = []
    for x in text.strip('\r').split():
        res.append(list(x))
    return res


def get_tokenizer(method='default'):
    if method == 'default':
        return default_fn
    elif method == 'sub_text':
        return sub_text_fn
    elif method == 'jieba':
        return lambda x: jieba.cut(x)
