from dataclasses import dataclass

@dataclass
class Config:
    # Dataset parameters
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    max_tokens: int = 20

    # Model parameters
    vocab_size: int = 0          # filled later from data_pkg
    pad_id: int = 0              # filled later from data_pkg
    dim_embed: int = 64          # embedding / attention dimension
    num_classes: int = 0         # filled later from data_pkg
    use_multihead: bool = False  # single-head baseline

    # Training parameters
    lr: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda"       
    log_interval: int = 100
