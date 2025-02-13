import wandb
import torch

from dataclasses import dataclass, field, asdict
from nntool.slurm import SlurmArgs, slurm_launcher
from src.model import MultiScaleConv3dMlsesModel
from src.utils import seed_everything
from src.config import RunConfig, WandbConfig
from src.trainer import EPB3dSurfaceTrainer, EPBSurfaceTrainerConfig


@dataclass
class ExperimentConfig:
    basic: RunConfig = field(default_factory=RunConfig)

    model: MultiScaleConv3dMlsesModel = field(
        default_factory=MultiScaleConv3dMlsesModel
    )

    trainer: EPBSurfaceTrainerConfig = field(default_factory=EPBSurfaceTrainerConfig)

    wandb: WandbConfig = field(default_factory=WandbConfig)

    slurm: SlurmArgs = field(default_factory=SlurmArgs)

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

    model = MultiScaleConv3dMlsesModel(args.model.kernel_sizes, args.model.model_type)
    trainer = EPB3dSurfaceTrainer(
        model, args.basic.seed, args=args.trainer, has_wandb_writer=True
    )
    if args.resume_trainer:
        state_pt = torch.load(args.trainer_ckpt_path, map_location=trainer.device)
        trainer.load_state(state_pt)

    trainer.train()


if __name__ == "__main__":
    main()
