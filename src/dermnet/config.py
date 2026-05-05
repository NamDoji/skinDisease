"""
Config loader using OmegaConf.
Provides load_config() and override_config() for CLI argument merging.
"""
from pathlib import Path
from omegaconf import OmegaConf, DictConfig


def load_config(config_path: str = "configs/default.yaml") -> DictConfig:
    """Load YAML config file and return an OmegaConf DictConfig."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return OmegaConf.load(path)


def override_config(cfg: DictConfig, args) -> DictConfig:
    """
    Apply CLI argument overrides onto an existing config.
    Only overrides keys that are explicitly provided (not None).

    Args:
        cfg:  base OmegaConf config
        args: argparse.Namespace — keys matching config fields are merged in

    Returns:
        Updated DictConfig
    """
    overrides = {}
    if getattr(args, "backbone", None) is not None:
        overrides["model.backbone"] = args.backbone
    if getattr(args, "batch_size", None) is not None:
        overrides["training.batch_size"] = args.batch_size
    if getattr(args, "epochs", None) is not None:
        overrides["training.epochs"] = args.epochs
    if getattr(args, "lr", None) is not None:
        overrides["training.learning_rate"] = args.lr

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(
            [f"{k}={v}" for k, v in overrides.items()]
        ))
    return cfg
