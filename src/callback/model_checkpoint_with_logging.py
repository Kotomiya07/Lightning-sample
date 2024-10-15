import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class ModelCheckpointWithLogging(ModelCheckpoint):
    def _save_none_monitor_checkpoint(
        self, trainer: pl.Trainer, monitor_candidates: dict[str, torch.Tensor]
    ) -> None:
        super()._save_none_monitor_checkpoint(trainer, monitor_candidates)
        self._log_ckpt_path(trainer)

    def _update_best_and_save(
        self,
        current: torch.Tensor,
        trainer: pl.Trainer,
        monitor_candidates: dict[str, torch.Tensor],
    ) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        self._log_ckpt_path(trainer)

    def _log_ckpt_path(self, trainer: pl.Trainer) -> None:
        trainer.logger.experiment.summary["ckpt"] = self.best_model_path
