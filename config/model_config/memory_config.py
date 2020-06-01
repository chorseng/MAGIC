from config.model_config import GraphEncoderConfig
from util.better_abc import ABCMeta, abstract_attribute


class MemoryConfig(metaclass=ABCMeta):
    knowledge_size = abstract_attribute()
    entry_size = abstract_attribute()
    memory_size = 512
    output_size = 512


class StyletipMemoryConfig(MemoryConfig):
    entry_size = GraphEncoderConfig.embed_size * 2

    def __init__(self, knowledge_size):
        self.knowledge_size = knowledge_size


class CelebrityMemoryConfig(MemoryConfig):
    memory_size = 512
    output_size = 512

    def __init__(self, knowledge_size, entry_size):
        self.knowledge_size = knowledge_size
        self.entry_size = entry_size
