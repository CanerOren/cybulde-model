from dataclasses import dataclass

from omegaconf import MISSING
from hydra.core.config_store import ConfigStore


@dataclass
class TransformationConfig:
    _target_: str = MISSING

@dataclass
class HuggingFaceTokenizationTransformationConfig:
    _target_: str = "cybulde.data_modules.transformations.HuggingFaceTokenizationTransformation"
    pretrained_tokenizer_name_or_path: str = MISSING
    max_sequence_length: int = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        name="huggingface_tokenization_transformation_schema",
        group="tasks/data_module/transformation",
        node=HuggingFaceTokenizationTransformationConfig
    )