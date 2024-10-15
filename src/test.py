from pathlib import Path

import hydra
import lightning as L
import torch
from fire import Fire
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf


def test(ckpt: str) -> None:
    hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(Path(ckpt).parent.parent / ".hydra"),
    )
    cfg = hydra.compose(config_name="config")
    cfg.ckpt = hydra.utils.to_absolute_path(ckpt)

    L.seed_everything(cfg.seed, workers=True)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    wandb_logger = WandbLogger(project=cfg.project_name)
    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    model = hydra.utils.instantiate(cfg.litmodule)

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    trainer: L.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=wandb_logger,
    )

    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt)


if __name__ == "__main__":
    Fire(test)
