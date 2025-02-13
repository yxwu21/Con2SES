import torch

from tqdm import tqdm
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from src.model import MultiScaleConvMlsesModel, MLSESModel
from src.utils import seed_everything
from src.config import RunConfig
from src.slurm import SlurmConfig, slurm_launcher
from src.trainer import EPBSurfaceTrainer, GENIUSESTrainer, EPBSurfaceTrainerConfig
from src.utils import visualize_level_set_image, visualize_joint_level_set_image


@dataclass
class ExperimentConfig:
    # trainer checkpoint path
    trainer_ckpt_path: str

    geniuses_model_ckpt_path: str = ""

    chunk_size: int = 5

    model: MultiScaleConvMlsesModel = field(default_factory=MultiScaleConvMlsesModel)

    basic: RunConfig = field(default_factory=RunConfig)

    trainer: EPBSurfaceTrainerConfig = field(default_factory=EPBSurfaceTrainerConfig)

    slurm: SlurmConfig = field(default_factory=SlurmConfig)


@slurm_launcher(ExperimentConfig)
def main(args: ExperimentConfig):
    seed_everything(args.basic.seed)

    # eval geniuses
    model = MLSESModel(96, 64, 32)
    model.load_state_dict(torch.load(args.geniuses_model_ckpt_path))
    gen_trainer = GENIUSESTrainer(
        model, args.basic.seed, args=args.trainer, has_wandb_writer=False
    )

    _, gen_outputs = gen_trainer.eval(gen_trainer.test_dl)

    # eval epb
    model = MultiScaleConvMlsesModel(args.model.kernel_sizes)
    trainer = EPBSurfaceTrainer(
        model, args.basic.seed, args=args.trainer, has_wandb_writer=False
    )
    state_pt = torch.load(args.trainer_ckpt_path, map_location=trainer.device)
    trainer.load_state(state_pt)

    _, epb_outputs = trainer.eval(trainer.test_dl)

    num_sample = len(epb_outputs["preds"])
    lst = list(range(num_sample))
    chunk_size = args.chunk_size
    chunk_lst = [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]

    print("plotting level-set")
    for i, chunk in tqdm(enumerate(chunk_lst), total=len(chunk_lst)):
        # fig = visualize_level_set_image(
        #     outputs["preds"],
        #     outputs["labels"],
        #     outputs["masks"],
        #     chunk,
        #     trainer.label_transformer.transform(torch.zeros(1)).item(),
        #     trainer.label_transformer.transform(
        #         torch.tensor(args.trainer.probe_radius_lowerbound)
        #     ).item(),
        #     trainer.label_transformer.transform(
        #         torch.tensor(args.trainer.probe_radius_upperbound)
        #     ).item(),
        # )
        fig = visualize_joint_level_set_image(
            gen_outputs["preds"],
            epb_outputs["preds"],
            epb_outputs["labels"],
            epb_outputs["masks"],
            chunk,
            trainer.label_transformer.transform(torch.zeros(1)).item(),
            trainer.label_transformer.transform(
                torch.tensor(args.trainer.probe_radius_lowerbound)
            ).item(),
            trainer.label_transformer.transform(
                torch.tensor(args.trainer.probe_radius_upperbound)
            ).item(),
        )
        fig.savefig(trainer.output_folder / f"level_set_{i}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    main()
