from config import DatasetConfig
from config.model_config import ContextEncoderConfig


class IntentionConfig:
    in_size = ContextEncoderConfig.output_size
    out_size = DatasetConfig.utterance_type_size
    fc_sizes = [in_size, 512, out_size]
