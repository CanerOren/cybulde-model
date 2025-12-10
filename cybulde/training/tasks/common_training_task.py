from typing import Union, TYPE_CHECKING

from lightning.pytorch import Trainer
from cybulde.configs_schemas.config_schema import Config
from cybulde.data_modules.data_modules import DataModule, PartialDataModule
from cybulde.training.lightning_modules.bases import TrainingLightningModule
from cybulde.training.tasks.bases import TrainingTask
from cybulde.utils.mlflow_utils import activate_mlflow, log_artifacts_for_reproducibility
from cybulde.utils.io_utils import is_file

if TYPE_CHECKING:
    from cybulde.configs_schemas.config_schema import Config
    from cybulde.configs_schemas.training.training_task_schema import TrainingTaskConfig

class CommonTrainingTask(TrainingTask):
    def __init__(
        self,
        name: str,
        data_module: Union[DataModule, PartialDataModule],
        lightning_module:TrainingLightningModule,
        trainer: Trainer,
        best_training_checkpoint: str,
        last_training_checkpoint: str,
    ) -> None:
        super().__init__(
            name=name,
            data_module=data_module,
            lightning_module=lightning_module,
            trainer=trainer,
            best_training_checkpoint=best_training_checkpoint,
            last_training_checkpoint=last_training_checkpoint
        )

    def run(self, config: "Config", task_config: "TrainingTaskConfig") -> None:
        experiment_name = config.infrastructure.mlflow.experiment_name
        run_id = config.infrastructure.mlflow.run_id
        run_name = config.infrastructure.mlflow.run_name
        with activate_mlflow(experiment_name=experiment_name, run_id=run_id, run_name=run_name) as _:
            if self.trainer.is_global_zero:
                log_artifacts_for_reproducibility()
                # log_training_hparams(config)

            assert isinstance(self.data_module, DataModule)
            if is_file(self.last_training_checkpoint):
                self.logger.info(f"Found checkpoint here: {self.last_training_checkpoint}. Resuming training")
                self.trainer.fit(model=self.lightning_module, datamodule=self.data_module, ckpt_path=self.last_training_checkpoint)
            else:
                self.trainer.fit(model=self.lightning_module, datamodule=self.data_module)
            self.logger.info("Training finished...")