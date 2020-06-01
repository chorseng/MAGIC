class GraphEncoderConfig:
    embed_size = 300

    def __init__(self, vocab_size: int):
        self.vocab_size: int = vocab_size
