import torch
from torch import nn

from config import GlobalConfig
from config.model_config import MemoryConfig


class Memory(nn.Module):
    def __init__(self, config: MemoryConfig):
        super(Memory, self).__init__()
        self.memory_embedding = nn.Linear(config.entry_size, config.memory_size,
                                          bias=False).to(GlobalConfig.device)
        self.output_embedding = nn.Linear(config.entry_size, config.output_size,
                                          bias=False).to(GlobalConfig.device)
        self.softmax = nn.Softmax(dim=1).to(GlobalConfig.device)

    def forward(self, entry, query, id=None):
        """Forward.

        Args:
            entry: Knowledge entry (knowledge_size, entry_size).
            query: Query (batch_size, hidden_size).

        Returns:
            knowledge: Knowledge (batch_size, output_size).

        """

        memory = self.memory_embedding(entry)
        # (knowledge_size, memory_size)

        output = self.output_embedding(entry)
        # (knowledge_size, output_size)

        probability = torch.mm(query, memory.transpose(0, 1))
        probability = self.softmax(probability)
        # (batch_size, knowledge_size)

        knowledge = torch.mm(probability, output)
        # (batch_size, output_size)

        return knowledge
