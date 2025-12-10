from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from omegaconf import MISSING

from cybulde.configs_schemas.transformation_schemas import CustomHuggingFaceTokenizationTransformationConfig ,TransformationConfig


@dataclass
class BackboneConfig:
    _target_: str = MISSING
    transformation: TransformationConfig = MISSING


@dataclass
class HuggingFaceBackboneConfig(BackboneConfig):
    _target_: str = "cybulde.models.backbones.HuggingFaceBackbone"
    pretrained_model_name_or_path: str = MISSING
    pretrained: bool = False


@dataclass
class BertTinyHuggingFaceBackboneConfig(HuggingFaceBackboneConfig):
    pretrained_model_name_or_path: str = "prajjwal1/bert-tiny"
    transformation: TransformationConfig = field(default_factory=CustomHuggingFaceTokenizationTransformationConfig)

def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="hugging_face_backbone_schema",
        group="tasks/lightning_module/model/backbone",
        node=HuggingFaceBackboneConfig,
    )

    cs.store(
        name="test_backbone_config",
        node=BertTinyHuggingFaceBackboneConfig
    )