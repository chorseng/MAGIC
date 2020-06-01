from util.better_abc import ABCMeta, abstract_attribute


class ValidConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_batches = abstract_attribute()
    num_data_loader_workers = abstract_attribute()


class IntentionValidConfig(ValidConfig):
    batch_size = 64
    num_batches = 100
    num_data_loader_workers = 5


class TextValidConfig(ValidConfig):
    batch_size = 64
    num_batches = 100
    num_data_loader_workers = 5


class RecommendValidConfig(ValidConfig):
    batch_size = 64
    num_batches = 100
    num_data_loader_workers = 5


class KnowledgeStyletipValidConfig(ValidConfig):
    batch_size = 64
    num_batches = 100
    num_data_loader_workers = 5


class KnowledgeAttributeValidConfig(ValidConfig):
    batch_size = 64
    num_batches = 100
    num_data_loader_workers = 5


class KnowledgeCelebrityValidConfig(ValidConfig):
    batch_size = 64
    num_batches = 100
    num_data_loader_workers = 5
