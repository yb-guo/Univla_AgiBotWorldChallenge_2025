from lightning.pytorch.cli import LightningCLI
from genie.dataset_a2d import LightningA2D
from genie.model import DINO_LAM


cli = LightningCLI(
    DINO_LAM,
    LightningA2D,
    seed_everything_default=42,
)