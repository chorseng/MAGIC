from util.better_abc import ABCMeta, abstract_attribute


class TestConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_data_loader_workers = abstract_attribute()


class IntentionTestConfig(TestConfig):
    batch_size = 64
    num_data_loader_workers = 5


class TextTestConfig(TestConfig):
    batch_size = 64
    num_data_loader_workers = 5


class RecommendTestConfig(TestConfig):
    batch_size = 64
    num_data_loader_workers = 5


class KnowledgeStyletipTestConfig(TestConfig):
    batch_size = 64
    num_data_loader_workers = 5


class KnowledgeAttributeTestConfig(TestConfig):
    batch_size = 64
    num_data_loader_workers = 5


class KnowledgeCelebrityTestConfig(TestConfig):
    batch_size = 64
    num_data_loader_workers = 5
