import torch


class GlobalConfig:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
