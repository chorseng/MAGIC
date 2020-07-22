from util.better_abc import ABCMeta, abstract_attribute


class TrainConfig(metaclass=ABCMeta):
    batch_size = abstract_attribute()
    num_iterations = abstract_attribute()
    learning_rate = abstract_attribute()
    print_freq = abstract_attribute()
    valid_freq = abstract_attribute()
    patience = abstract_attribute()
    num_data_loader_workers = abstract_attribute()


class IntentionTrainConfig(TrainConfig):
    batch_size = 64
    num_iterations = 10000
    learning_rate = 0.0001
    print_freq = 100
    valid_freq = 100
    patience = 5
    num_data_loader_workers = 5


class TextTrainConfig(TrainConfig):
    batch_size = 64
    num_iterations = 10000
    learning_rate = 0.0001
    print_freq = 10
    valid_freq = 10
    patience = 20
    num_data_loader_workers = 5


class RecommendTrainConfig(TrainConfig):
    batch_size = 64
    num_iterations = 10000
    learning_rate = 0.0001
    print_freq = 100
    valid_freq = 100
    patience = 5
    num_data_loader_workers = 5


class KnowledgeStyletipTrainConfig(TrainConfig):
    batch_size = 64
    num_iterations = 10000
    learning_rate = 0.00004
    print_freq = 100
    valid_freq = 100
    patience = 20
    num_data_loader_workers = 5


class KnowledgeAttributeTrainConfig(TrainConfig):
    batch_size = 64
    num_iterations = 10000
    learning_rate = 0.0004
    print_freq = 100
    valid_freq = 100
    patience = 20
    num_data_loader_workers = 5


class KnowledgeCelebrityTrainConfig(TrainConfig):
    batch_size = 64
    num_iterations = 10000
    learning_rate = 0.0004
    print_freq = 100
    valid_freq = 100
    patience = 20
    num_data_loader_workers = 5
