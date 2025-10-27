from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42

    # Dataset parameters
    batch_size: int = 128
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    max_tokens: int = 20

    # Model parameters
    vocab_size: int = 0       
    pad_id: int = 0              
    dim_embed: int = 2         
    num_classes: int = 0        
    use_multihead: bool = False  

    # Training parameters
    lr: float = 1e-3
    num_epochs: int = 10
    device: str = "cuda"       
    log_interval: int = 100
