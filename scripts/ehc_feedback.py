"""Entorhinal Hippocampal Circuit training script with hybrid feedback learning."""

import matplotlib.pyplot as plt
import torch
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ehc_sn.core.datamodule import BaseDataModule, DataModuleParams
from ehc_sn.core.trainer import TrainerParams
from ehc_sn.data.obstacle_maps import DataGenerator, DataParams
from ehc_sn.figures.decoder_montage import DecoderMontageFigure
from ehc_sn.figures.decoder_montage import DecoderMontageParams as Figure3Params
from ehc_sn.figures.reconstruction_map import ReconstructionMapFigure
from ehc_sn.figures.reconstruction_map import ReconstructionMapParams as Figure1Params
from ehc_sn.figures.sparsity import SparsityFigure
from ehc_sn.figures.sparsity import SparsityParams as Figure2Params
from ehc_sn.models.ann.ehc import EHC, ModelParams
from ehc_sn.trainers.feed_forward import FeedbackTainer


# -------------------------------------------------------------------------------------------
class Experiment(BaseSettings):
    model_config = SettingsConfigDict(extra="forbid", cli_parse_args=True)

    data: DataParams = Field(default_factory=DataParams, description="Data generation parameters")
    datamodule: DataModuleParams = Field(default_factory=DataModuleParams, description="Data module parameters")
    model: ModelParams = Field(default_factory=ModelParams, description="Autoencoder parameters")

    figure_1: Figure1Params = Field(default_factory=Figure1Params, description="Reconstruction figure parameters")
    figure_2: Figure2Params = Field(default_factory=Figure2Params, description="Sparsity figure parameters")
    figure_3: Figure3Params = Field(default_factory=Figure3Params, description="Decoder montage figure parameters")

    trainer: TrainerParams = Field(
        default_factory=lambda: TrainerParams(experiment_name="ehc"),
        description="Trainer parameters",
    )


# -------------------------------------------------------------------------------------------
def main(experiment: Experiment) -> None:
    # Generate data and initialize components
    data_gen = DataGenerator(experiment.data)
    datamodule = BaseDataModule(data_gen, experiment.datamodule)
    trainer = FeedbackTainer(experiment.trainer)
    model = EHC(experiment.model, trainer)

    # Train till end of training or keyboard interup
    try:
        trainer.fit(model, datamodule)
    except KeyboardInterrupt:
        print("Training interrupted by user. Generating figures...")
    finally:
        gen_figures(model, datamodule, experiment)


# -------------------------------------------------------------------------------------------
def gen_figures(model: EHC, datamodule: BaseDataModule, experiment: Experiment) -> None:
    """Generate reconstruction and sparsity figures from model outputs."""
    model.eval()
    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()

    try:
        inputs, _ = next(iter(test_dataloader))
    except StopIteration:
        print("No test data available for plotting.")
        return

    with torch.inference_mode():
        outputs, activations = model(inputs)

    fig_reconstruction = ReconstructionMapFigure(experiment.figure_1)
    _ = fig_reconstruction.plot(inputs, outputs)
    plt.show()

    sparsity_figure = SparsityFigure(experiment.figure_2)
    _ = sparsity_figure.plot(activations)
    plt.show()

    latent_dim = experiment.model.dg_units  # Number of latent units
    latents = torch.eye(latent_dim)[:18]  # One-hot encoding for each unit

    # Generate decoder outputs for one-hot latents
    reconstructions = model.decode(latents)

    # Figure 3: Decoder montage showing individual latent unit reconstructions
    decoder_montage_figure = DecoderMontageFigure(experiment.figure_3)
    _ = decoder_montage_figure.plot(latents, reconstructions)
    plt.show()


if __name__ == "__main__":
    """Main entry point for DFA autoencoder experiment."""
    experiment = Experiment()
    main(experiment)
