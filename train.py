import torch
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
import wandb


class MySaveConfigCallback(SaveConfigCallback):

    def save_config(self, trainer, pl_module, stage: str) -> None:

        if isinstance(trainer.logger, WandbLogger):
            config = self.parser.dump(
                self.config, skip_none=False
            )  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})

            artifact = wandb.Artifact("config", type="config")

            # Serialize config to a string first
            config_str = self.parser.dump(
                self.config, format="yaml"
            )  # Adjust 'format' as needed

            # Manually write to file within the artifact
            with artifact.new_file("config.yaml", mode="w") as f:
                f.write(config_str)  # Write the serialized string to file

            wandb.log_artifact(artifact)


def main():
    torch.set_printoptions(precision=4, linewidth=200, sci_mode=False)

    LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_callback=MySaveConfigCallback,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
