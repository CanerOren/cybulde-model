from typing import Optional
from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from cybulde.configs_schemas.models import backbone_schemas, adapter_schemas, head_schemas
from cybulde.utils.mixins import LoggableParamsMixin


@dataclass
class ModelConfig(LoggableParamsMixin):
    _target_: str = MISSING

    def loggable_params(self) -> list[str]:
        return ["_target_"]


@dataclass
class BinaryTextClassificationModelConfig(ModelConfig):
    _target_: str = "cybulde.models.models.BinaryTextClassificationModel"
    backbone: backbone_schemas.BackboneConfig = MISSING
    adapter: Optional[adapter_schemas.AdapterConfig] = None
    head: head_schemas.HeadConfig = MISSING

@dataclass
class BertTinyBinaryTextClassificationModelConfig(BinaryTextClassificationModelConfig):
    backbone: backbone_schemas.BackboneConfig = field(default_factory=backbone_schemas.BertTinyHuggingFaceBackboneConfig)
    adapter: Optional[adapter_schemas.AdapterConfig] = field(default_factory=adapter_schemas.PoolerOutputAdapterConfig)
    head: head_schemas.HeadConfig = field(default_factory=head_schemas.BinaryClassificationSigmoidHead)

def setup_config() -> None:
    backbone_schemas.setup_config()
    adapter_schemas.setup_config()
    head_schemas.setup_config()
    
    cs = ConfigStore.instance()
    cs.store(
        name="binary_text_classification_model_schema",
        group="tasks/lightning_module/model",
        node=BinaryTextClassificationModelConfig
    )
    
    cs.store(
        name="test_model",
        node=BertTinyBinaryTextClassificationModelConfig
    )