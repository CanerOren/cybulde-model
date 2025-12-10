from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cybulde.configs_schemas.base_schemas import TaskConfig
from cybulde.configs_schemas.config_schema import Config
from cybulde.configs_schemas.training.training_task_schemas import DefaultCommonTrainingTaskConfig

@dataclass
class LocalBertExperiment(Config):
    tasks: dict[str, TaskConfig] = field(default_factory=lambda: {
        "binary_text_classification_task": DefaultCommonTrainingTaskConfig(),
    }
    )


FinalLocalBertExperiment = OmegaConf.merge(
    LocalBertExperiment,
    OmegaConf.from_dotlist([
        #"tasks.binary_text_classification_task.data_module.batch_size=128"
    ])
)

cs = ConfigStore.instance()
cs.store(
    name="local_bert",
    group="experiment/bert",
    node=FinalLocalBertExperiment,
    package="_global_"
)