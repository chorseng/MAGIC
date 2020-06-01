import torch
from torch import nn
from config import GlobalConfig
from config.model_config import AttributeKVMemoryConfig
from util import get_mask, masked_softmax


class KVMemory(nn.Module):
    def __init__(self, config: AttributeKVMemoryConfig):
        super(KVMemory, self).__init__()

        self.key_embedding = nn.Embedding(config.num_keys,
                                          config.key_size)
        self.key_embedding = self.key_embedding.to(GlobalConfig.device)

        self.value_embedding = nn.Embedding(config.num_values,
                                            config.value_size)
        self.value_embedding = self.value_embedding.to(GlobalConfig.device)

    def forward(self, keys, values, pair_length, query, id=None):
        """Forward.

        Args:
            keys: Keys (batch_size, num_keys).
            values: Values (batch_size, num_keys).
            pair_length: Pair length (batch_size, ).
            query: Query (batch_size, hidden_size).

        Returns:
            (batch_size, value_size)

        """

        num_keys = keys.size(1)

        keys = self.key_embedding(keys)
        # (batch_size, num_keys, key_size)
        values = self.value_embedding(values)
        # (batch_size, num_keys, value_size)

        probability = torch.matmul(query.unsqueeze(1), keys.transpose(1, 2))
        probability = probability.squeeze(1)
        # (batch_size, num_keys)

        mask = get_mask(num_keys, pair_length)
        # (batch_size, num_keys)

        probability = masked_softmax(probability, mask, 1)
        # (batch_size, num_keys)

        knowledge = torch.matmul(probability.unsqueeze(1), values)
        knowledge = knowledge.squeeze(1)
        # (batch_size, value_size)

        return knowledge
