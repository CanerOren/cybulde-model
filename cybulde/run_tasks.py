from lightning.pytorch import seed_everything

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cybulde.configs_schemas.config_schema import Config
from cybulde.utils.config_utils import get_config
from cybulde.utils.utils import get_logger
from cybulde.utils.torch_utils import get_local_rank

@get_config(config_path="../configs/automatically_generated", config_name="config", to_object=False, return_dict_config=True)
def entrypoint(config: Config) -> None:

    print(60 * "#")
    print(OmegaConf.to_yaml(config, resolve= True))
    print(60 * "#")

    return(0)
    logger = get_logger(__file__)
    assert config.infrastructure.mlflow.run_id is not None, "Run id has to be set for running tasks"

    backend = "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{get_local_rank()}")
        backend = "nccl"

    torch.distributed.init_process_group(backend=backend)

    seed_everything(seed=config.seed, workers=True)

    for task_name, task_config in config.tasks.items():
        logger.info(f"Running: {task_name}")
        task = instantiate(task_config)
        task.run(config=config, task_config=task_config)

if __name__ == "__main__":
    entrypoint() # type: ignore