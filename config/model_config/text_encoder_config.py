from constant import PAD_ID
from util.better_abc import ABCMeta, abstract_attribute


class TextEncoderConfig(metaclass=ABCMeta):
    pad_index = abstract_attribute()
    embed_size = abstract_attribute()
    hidden_size = abstract_attribute()
    num_directions = abstract_attribute()
    num_layers = abstract_attribute()
    dropout = abstract_attribute()
    text_feat_size = abstract_attribute()
    vocab_size = abstract_attribute()
    embed_init = abstract_attribute()


class ContextTextEncoderConfig(TextEncoderConfig):
    pad_index = PAD_ID
    embed_size = 300
    hidden_size = 256
    num_directions = 2
    num_layers = 1
    dropout = 0
    text_feat_size = hidden_size * num_layers * num_directions

    def __init__(self, vocab_size, embed_init=None):
        super(ContextTextEncoderConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init


class ProductTextEncoderConfig(TextEncoderConfig):
    pad_index = PAD_ID
    embed_size = 300
    hidden_size = 256
    num_directions = 2
    num_layers = 1
    dropout = 0
    text_feat_size = hidden_size * num_layers * num_directions

    def __init__(self, vocab_size, embed_init=None):
        super(ProductTextEncoderConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init
