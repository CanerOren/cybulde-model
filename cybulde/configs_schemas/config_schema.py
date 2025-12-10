from dataclasses import field

from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass
from omegaconf import MISSING

from cybulde.configs_schemas import base_schemas
from cybulde.configs_schemas.infrastructure import infrastructure_schema
from cybulde.configs_schemas.training import training_task_schemas


@dataclass 
class Config:
    infrastructure: infrastructure_schema.InfrastructureConfig = field(default_factory=infrastructure_schema.InfrastructureConfig)
    save_last_checkpoint_every_n_train_steps: int = 500
    seed: int = 123
    tasks: dict[str, base_schemas.TaskConfig] = MISSING

def setup_config() -> None:
    infrastructure_schema.setup_config()
    training_task_schemas.setup_config()

    cs = ConfigStore.instance()
    cs.store(name="config_schema", node=Config)