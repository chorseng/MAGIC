class AttributeKVMemoryConfig:
    key_size = 512
    value_size = 512

    def __init__(self, num_keys: int, num_values: int):
        self.num_keys = num_keys
        self.num_values = num_values
