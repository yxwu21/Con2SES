import wandb
import torch

from dataclasses import dataclass, field, asdict
from src.model import MultiScaleConvMlsesModel
from src.utils import seed_everything
from src.config import RunConfig, WandbConfig
from src.slurm import SlurmConfig, slurm_launcher
from src.trainer import EPBSurfaceTrainer, EPBSurfaceTrainerConfig


@dataclass
class ExperimentConfig:
    basic: RunConfig = field(default_factory=RunConfig)

    model: MultiScaleConvMlsesModel = field(default_factory=MultiScaleConvMlsesModel)

    trainer: EPBSurfaceTrainerConfig = field(default_factory=EPBSurfaceTrainerConfig)

    wandb: WandbConfig = field(default_factory=WandbConfig)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # trainer should be resumed
    resume_trainer: bool = False

    # trainer checkpoint path
    trainer_ckpt_path: str = ""


@slurm_launcher(ExperimentConfig)
def main(args: ExperimentConfig):
    seed_everything(args.basic.seed)

    wandb.login(key=args.wandb.api_key)
    wandb.init(
        project=args.wandb.project,
        entity=args.wandb.entity,
        config=asdict(args),
        name=args.wandb.name,
    )

    model = MultiScaleConvMlsesModel(args.model.kernel_sizes, args.model.model_type)
    trainer = EPBSurfaceTrainer(
        model, args.basic.seed, args=args.trainer, has_wandb_writer=True
    )
    if args.resume_trainer:
        state_pt = torch.load(args.trainer_ckpt_path, map_location=trainer.device)
        trainer.load_state(state_pt)

    trainer.train()


if __name__ == "__main__":
    main()
