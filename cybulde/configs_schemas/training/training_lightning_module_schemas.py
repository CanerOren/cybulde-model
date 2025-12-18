from dataclasses import dataclass, field
from typing import Optional

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore

from cybulde.configs_schemas.base_schemas import LightningModuleConfig
from cybulde.configs_schemas.models.model_schemas import BertTinyBinaryTextClassificationModelConfig ,ModelConfig
from cybulde.configs_schemas.training import loss_schemas, optimizer_schemas, scheduler_schemas
from cybulde.utils.mixins import LoggableParamsMixin

@dataclass
class TrainingLightningModuleConfig(LightningModuleConfig, LoggableParamsMixin):
    _target_: str = MISSING
    model: ModelConfig = MISSING
    loss: loss_schemas.LossFunctionConfig = MISSING
    optimizer: optimizer_schemas.OptimizerConfig = MISSING
    scheduler: Optional[scheduler_schemas.LightningSchedulerConfig] = None

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class BinaryTextClassificationTrainingLightningModuleConfig(TrainingLightningModuleConfig):
    _target_: str = "cybulde.training.lightning_modules.binary_text_classification.BinaryTextClassificationTrainingLightningModule"

@dataclass
class CybuldeBinaryTextClassificationTrainingLightningModuleConfig(BinaryTextClassificationTrainingLightningModuleConfig):
    model: ModelConfig = field(default_factory=BertTinyBinaryTextClassificationModelConfig)
    loss: loss_schemas.LossFunctionConfig = field(default_factory=loss_schemas.BCEWithLogitsLossConfig)
    optimizer: optimizer_schemas.OptimizerConfig = field(default_factory=optimizer_schemas.AdamWOptimizerConfig)
    scheduler: Optional[scheduler_schemas.LightningSchedulerConfig] = field(default_factory=scheduler_schemas.ReduceLROnPlateauLightningSchedulerConfig) 


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_training_lightning_module_schema",
        group="tasks/lightning_module",
        node=BinaryTextClassificationTrainingLightningModuleConfig
    )