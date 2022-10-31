from collections import Counter

import torch
import six
from torchtext.data import RawField, Pipeline
from torchtext.data.utils import dtype_to_attr, is_tokenizer_serializable, get_tokenizer
from torchtext.vocab import Vocab


class SubTextField(RawField):
    """Defines a datatype together with instructions for converting to Tensor.
    提供      # [sub_seq, seq, batch]的
    Field class models common text processing datatypes that can be represented
    by tensors.  It holds a Vocab object that defines the set of possible values
    for elements of the field and their corresponding numerical representations.
    The Field object also holds other parameters relating to how a datatype
    should be numericalized, such as a tokenization method and the kind of
    Tensor that should be produced.

    If a Field is shared between two columns in a dataset (e.g., question and
    answer in a QA dataset), then they will have a shared vocabulary.

    Attributes:
        sequential: Whether the datatype represents sequential data. If False,
            no tokenization is applied. Default: True.
        use_vocab: Whether to use a Vocab object. If False, the data in this
            field should already be numerical. Default: True.
        init_token: A token that will be prepended to every example using this
            field, or None for no initial token. Default: None.
        eos_token: A token that will be appended to every example using this
            field, or None for no end-of-sentence token. Default: None.
        fix_length: A fixed length that all examples using this field will be
            padded to, or None for flexible sequence lengths. Default: None.
        dtype: The torch.dtype class that represents a batch of examples
            of this kind of data. Default: torch.long.
        preprocessing: The Pipeline that will be applied to examples
            using this field after tokenizing but before numericalizing. Many
            Datasets replace this attribute with a custom preprocessor.
            Default: None.
        postprocessing: A Pipeline that will be applied to examples using
            this field after numericalizing but before the numbers are turned
            into a Tensor. The pipeline function takes the batch as a list, and
            the field's Vocab.
            Default: None.
        lower: Whether to lowercase the text in this field. Default: False.
        tokenize: The function used to tokenize strings using this field into
            sequential examples. If "spacy", the SpaCy tokenizer is
            used. If a non-serializable function is passed as an argument,
            the field will not be able to be serialized. Default: string.split.
        tokenizer_language: The language of the tokenizer to be constructed.
            Various languages currently supported only in SpaCy.
        include_lengths: Whether to return a tuple of a padded minibatch and
            a list containing the lengths of each examples, or just a padded
            minibatch. Default: False.
        batch_first: Whether to produce tensors with the batch dimension first.
            Default: False.
        pad_token: The string token used as padding. Default: "<pad>".
        unk_token: The string token used to represent OOV words. Default: "<unk>".
        pad_first: Do the padding of the sequence at the beginning. Default: False.
        truncate_first: Do the truncating of the sequence at the beginning. Default: False
        stop_words: Tokens to discard during the preprocessing step. Default: None
        is_target: Whether this field is a target variable.
            Affects iteration over batches. Default: False

    """

    vocab_cls = Vocab
    # Dictionary mapping PyTorch tensor dtypes to the appropriate Python
    # numeric type.
    dtypes = {
        torch.float32: float,
        torch.float: float,
        torch.float64: float,
        torch.double: float,
        torch.float16: float,
        torch.half: float,

        torch.uint8: int,
        torch.int8: int,
        torch.int16: int,
        torch.short: int,
        torch.int32: int,
        torch.int: int,
        torch.int64: int,
        torch.long: int,
    }

    ignore = ['dtype', 'tokenize']

    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 context_split_notation=' ', utterance_split_notation='_', include_lengths=False,
                 batch_first=True, pad_token="<pad>", unk_token="<unk>", truncate_first=False, stop_words=None,
                 is_target=False, flat_to_dmc=False, max_len=-1, dynamic_vocab=False):
        assert batch_first is False and sequential, 'provide       # [sub_seq, seq, batch]'
        # assert utterance_split_notation != '_' and context_split_notation != '_', '_ is not allowed to be a notation'
        self.flat_to_dmc =flat_to_dmc
        self.max_total_len = max_len
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = [init_token]
        self.eos_token = [eos_token]
        self.unk_token = unk_token
        self.sub_eos_token = '</w>'
        self.sub_init_token = '<w>'
        # self.sub_eos_token = '</w>'
        self.sub_unk_token = unk_token
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.context_split_notation = context_split_notation
        self.utterance_split_notation = utterance_split_notation
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = [pad_token] if self.sequential else None
        self.sub_pad_token = '<p>'
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target
        self.dynamic_vocab = dynamic_vocab

    def __getstate__(self):
        str_type = dtype_to_attr(self.dtype)
        if is_tokenizer_serializable(*self.tokenizer_args):
            tokenize = self.tokenize
        else:
            # signal to restore in `__setstate__`
            tokenize = None
        attrs = {k: v for k, v in self.__dict__.items() if k not in self.ignore}
        attrs['dtype'] = str_type
        attrs['tokenize'] = tokenize

        return attrs

    def __setstate__(self, state):
        state['dtype'] = getattr(torch, state['dtype'])
        if not state['tokenize']:
            state['tokenize'] = get_tokenizer(*state['tokenizer_args'])
        self.__dict__.update(state)

    def __hash__(self):
        # we don't expect this to be called often
        return 42

    def __eq__(self, other):
        if not isinstance(other, RawField):
            return False

        return self.__dict__ == other.__dict__

    def preprocess(self, x):
        """Load a single example using this field, tokenizing if necessary.

        If the input is a Python 2 `str`, it will be converted to Unicode
        first. If `sequential=True`, it will be tokenized. Then the input
        will be optionally lowercased and passed to the user-provided
        `preprocessing` Pipeline."""
        if (six.PY2 and isinstance(x, six.string_types)
                and not isinstance(x, six.text_type)):
            x = Pipeline(lambda s: six.text_type(s, encoding='utf-8'))(x)
        if self.sequential and isinstance(x, six.text_type):
            x = [y.split(self.utterance_split_notation) for y in x.strip('\r\n').split(self.context_split_notation)]
        if self.lower:
            x = Pipeline(six.text_type.lower)(x)
        if self.sequential and self.use_vocab and self.stop_words is not None:
            x = [w for w in x if w not in self.stop_words]
        if self.preprocessing is not None:
            return self.preprocessing(x)
        else:
            return x

    def process(self, batch, device=None):
        """ Process a list of examples to create a torch.Tensor.

        Pad, numericalize, and postprocess a batch and create a tensor.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            torch.autograd.Variable: Processed object given the input
            and custom postprocessing Pipeline.
        """
        # 先做分割
        padded = self.pad(batch)
        uni_token_vocab_t2i = None
        uni_token_vocab_i2t_list = None
        if self.dynamic_vocab:
            uni_token_vocab_t2i = dict()
            sp_tokens = [self.pad_token[0], self.unk_token, self.init_token[0], self.eos_token[0],
                         self.sub_eos_token, self.sub_unk_token, self.sub_init_token, self.sub_pad_token]
            for sp_token in sp_tokens:
                if sp_token is not None and sp_token not in uni_token_vocab_t2i:
                    uni_token_vocab_t2i[sp_token] = len(uni_token_vocab_t2i)
            token_set = set()
            for batch in padded[0] if self.include_lengths else padded: #把Batch里的所有Token放一起
                for line in batch:
                    token_set = token_set | set(line)
            for token in token_set:
                if token not in uni_token_vocab_t2i:
                    uni_token_vocab_t2i[token] = len(uni_token_vocab_t2i)
            # Padding
            uni_token_vocab_i2t_list = []
            for key in uni_token_vocab_t2i.keys():
                uni_token_vocab_i2t_list.append(self.vocab.stoi[key])
            uni_token_vocab_i2t_list = torch.tensor(uni_token_vocab_i2t_list, dtype=self.dtype, device=device)

        if self.include_lengths:
            my_vocab = self.vocab.stoi if not self.dynamic_vocab else uni_token_vocab_t2i
            var, lengths, sub_lengths = self.numericalize(padded, my_vocab, device=device)
            if self.dynamic_vocab:
                return var, (lengths, sub_lengths), (uni_token_vocab_t2i, uni_token_vocab_i2t_list)
            else:
                return var,  (lengths, sub_lengths),
        else:
            my_vocab = self.vocab.stoi if not self.dynamic_vocab else uni_token_vocab_t2i
            var = self.numericalize(padded, my_vocab, device=device)
            if self.dynamic_vocab:
                return var, (uni_token_vocab_t2i, uni_token_vocab_i2t_list)
            else:
                return var

    def pad(self, minibatch):
        """Pad a batch of examples using this field.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True` and `self.sequential` is `True`, else just
        returns the padded list. If `self.sequential` is `False`, no padding is applied.
        """
        minibatch = list(minibatch)
        if not self.sequential:
            raise NotImplementedError()
        max_len = max(len(x) for x in minibatch)
        max_sub_len = 1
        for x in minibatch:
            for y in x:
                max_sub_len = max(max_sub_len, len(y))
        assert (max_len + 2) * (max_sub_len + 2) < self.max_total_len, \
            '%s-%s-%s-%s' % (max_len, max_sub_len, (max_len + 2) * (max_sub_len + 2), self.max_total_len)

        # 首先Sequence 级别Padding
        padded, lengths = [], []
        for x in minibatch:
            padded.append(
                ([] if self.init_token is None else [self.init_token])
                + list(x[-max_len:] if self.truncate_first else x[:max_len])
                + ([] if self.eos_token is None else [self.eos_token])
                + [self.pad_token] * max(0, max_len - len(x)))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))

        # 然后进行SubLevel的Padding

        spec_tokens = set()
        if self.init_token is not None: spec_tokens.add(self.init_token[0])
        if self.eos_token is not None: spec_tokens.add(self.eos_token[0])
        if self.pad_token is not None: spec_tokens.add(self.pad_token[0])

        padded_subs = []
        sub_lengths = []
        for seq in padded:
            padded_x_seq = []
            padded_x_len = []
            for x in list(seq):
                padded_x_seq.append(
                    ([] if self.sub_init_token is None else [self.sub_init_token])
                    + list(x[-max_sub_len:] if self.truncate_first else x[:max_sub_len])
                    + ([] if self.sub_eos_token is None else [self.sub_eos_token])
                    + [self.sub_pad_token] * max(0, max_sub_len - len(x)))
                x_len = len(padded_x_seq[-1]) - max(0, max_sub_len - len(x))
                assert x_len > 0
                padded_x_len.append(x_len)
            padded_subs.append(padded_x_seq)
            sub_lengths.append(padded_x_len)


        if self.include_lengths:
            return (padded_subs, lengths, sub_lengths)
        return padded_subs


    def numericalize(self, arr, vocab, device=None):
        """Turn a batch of examples that use this field into a Variable.

        If the field has include_lengths=True, a tensor of lengths will be
        included in the return value.

        Arguments:
            arr (List[List[str]], or tuple of (List[List[str]], List[int])):
                List of tokenized and padded examples, or tuple of List of
                tokenized and padded examples and List of lengths of each
                example if self.include_lengths is True.
            device (str or torch.device): A string or instance of `torch.device`
                specifying which device the Variables are going to be created on.
                If left as default, the tensors will be created on cpu. Default: None.
        """
        if self.include_lengths and not isinstance(arr, tuple):
            raise ValueError("Field has include_lengths set to True, but "
                             "input data is not a tuple of "
                             "(data batch, batch lengths).")
        if isinstance(arr, tuple):
            arr, lengths, sub_lengths = arr
            lengths = torch.tensor(lengths, dtype=self.dtype, device=device)
            sub_lengths = torch.tensor(sub_lengths, dtype=self.dtype, device=device).transpose(0, 1)

        for bid in range(len(arr)):
            for sid in range(len(arr[bid])):
                for wid in range(len(arr[bid][sid])):
                    arr[bid][sid][wid] = vocab[arr[bid][sid][wid]]

        if self.postprocessing is not None:
            arr = self.postprocessing(arr, self.vocab)

        arr = torch.tensor(arr, dtype=self.dtype, device=device)
        if self.sequential and not self.batch_first:
            # [sub_seq, seq, batch]
            if self.flat_to_dmc:
                # [batch, seq, sub_seq] => [ seq* sub_seq, batch]
                batch_len, seq_len, sub_seq_len = arr.size()
                res = arr.view(batch_len, seq_len * sub_seq_len).transpose(0, 1)
            else:
                #  [batch, seq, sub_seq]  = >[sub_seq, seq, batch]
                res = arr.permute(2, 1, 0)
        else:
            res = arr
        if self.sequential:
            res = res.contiguous()
        if self.include_lengths:
            return res, lengths, sub_lengths
        return res

