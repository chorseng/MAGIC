"""Configurations."""
from config.dataset_config import DatasetConfig
from config.global_config import GlobalConfig

from config.train_config import TrainConfig
from config.train_config import (IntentionTrainConfig,
                                 TextTrainConfig,
                                 RecommendTrainConfig,
                                 KnowledgeStyletipTrainConfig,
                                 KnowledgeAttributeTrainConfig,
                                 KnowledgeCelebrityTrainConfig)

from config.valid_config import ValidConfig
from config.valid_config import (IntentionValidConfig,
                                 TextValidConfig,
                                 RecommendValidConfig,
                                 KnowledgeStyletipValidConfig,
                                 KnowledgeAttributeValidConfig,
                                 KnowledgeCelebrityValidConfig)

from config.test_config import TestConfig
from config.test_config import (IntentionTestConfig,
                                TextTestConfig,
                                RecommendTestConfig,
                                KnowledgeStyletipTestConfig,
                                KnowledgeAttributeTestConfig,
                                KnowledgeCelebrityTestConfig)

from config.beam_search_config import BeamSearchConfig
