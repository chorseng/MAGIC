from torch import nn

from config import GlobalConfig
from config.model_config import GraphEncoderConfig


class GraphEncoder(nn.Module):
    def __init__(self, config: GraphEncoderConfig):
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.embedding = self.embedding.to(GlobalConfig.device)

    def forward(self, graph):
        """Forward.

        Args:
            graph: Edges of a graph (Number of edges, 2).

        Returns:
            output: (Number of edges, 2 * embed_size)

        """
        output = self.embedding(graph)
        # (Number of edges, 2, embed_size)
        output = output.view(output.size(0), -1)
        # (Number of edges, 2 * embed_size)
        return output
