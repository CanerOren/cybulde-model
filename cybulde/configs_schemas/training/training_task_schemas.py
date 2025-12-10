from dataclasses import dataclass, field

from omegaconf import SI
from hydra.core.config_store import ConfigStore 

from cybulde.configs_schemas.base_schemas import TaskConfig
from cybulde.configs_schemas import data_module_schemas
from cybulde.configs_schemas.training import training_lightning_module_schemas
from cybulde.configs_schemas.trainer import trainer_schemas


@dataclass
class TrainingTaskConfig(TaskConfig):
    best_training_checkpoint: str = SI("${infrastructure.mlflow.artifact_uri}/best-checkpoints/")
    last_training_checkpoint: str = SI("${infrastructure.mlflow.artifact_uri}/last-checkpoints/")


@dataclass
class CommonTrainingTaskConfig(TrainingTaskConfig):
    _target_: str = "cybulde.training.tasks.common_training_task.CommonTrainingTask"

@dataclass
class DefaultCommonTrainingTaskConfig(CommonTrainingTaskConfig):
    name: str = "binary_text_classification_task"
    data_module: data_module_schemas.DataModuleConfig = field(default_factory=data_module_schemas.ScrappedDataTextClassificationDataModuleConfig)
    lightning_module: training_lightning_module_schemas.TrainingLightningModuleConfig = field(default_factory=training_lightning_module_schemas.CybuldeBinaryTextClassificationTrainingLightningModuleConfig)
    trainer: trainer_schemas.TrainerConfig = field(default_factory=trainer_schemas.GPUDev)

def setup_config() -> None:
    data_module_schemas.setup_config()
    training_lightning_module_schemas.setup_config()
    trainer_schemas.setup_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="common_training_task_schema",
        group="tasks",
        node=CommonTrainingTaskConfig
    )