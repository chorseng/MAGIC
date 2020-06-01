from config.model_config import ContextTextEncoderConfig, ContextEncoderConfig
from constant import PAD_ID
from util.better_abc import ABCMeta, abstract_attribute
from config import DatasetConfig


class TextDecoderConfig(metaclass=ABCMeta):
    text_length = abstract_attribute()
    pad_index = abstract_attribute()
    word_embed_size = abstract_attribute()
    embed_size = abstract_attribute()
    hidden_size = abstract_attribute()
    num_directions = abstract_attribute()
    num_layers = abstract_attribute()
    dropout = abstract_attribute()
    to_hidden_fc_sizes = abstract_attribute()
    vocab_size = abstract_attribute()
    embed_init = abstract_attribute()
    utter_type_size = abstract_attribute()
    intention_embed_size = abstract_attribute()


class SimpleTextDecoderConfig(TextDecoderConfig):
    text_length = 30
    pad_index = PAD_ID
    word_embed_size = 300
    embed_size = 300
    hidden_size = 512
    num_directions = 2
    num_layers = 1
    dropout = 0
    utter_type_size = DatasetConfig.utterance_type_size
    intention_embed_size = 256
    to_hidden_fc_sizes = [ContextEncoderConfig.output_size, hidden_size]

    def __init__(self, vocab_size, embed_init=None):
        super(SimpleTextDecoderConfig, self).__init__()
        self.vocab_size = vocab_size
        self.embed_init = embed_init


class KnowledgeTextDecoderConfig(TextDecoderConfig):
    text_length = 30
    pad_index = PAD_ID
    word_embed_size = 300
    num_directions = 2
    num_layers = 1
    dropout = 0
    utter_type_size = DatasetConfig.utterance_type_size
    intention_embed_size = 256
    embed_size = 512

    def __init__(self, vocab_size, memory_size: int, output_size: int,
                 embed_init=None):
        super(KnowledgeTextDecoderConfig, self).__init__()
        self.fc_in_size = self.word_embed_size + output_size + \
                          ContextTextEncoderConfig.text_feat_size
        self.hidden_size = memory_size
        self.to_hidden_fc_sizes = [ContextEncoderConfig.output_size,
                                   self.hidden_size]
        self.vocab_size = vocab_size
        self.embed_init = embed_init
