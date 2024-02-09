from lightning.pytorch.cli import LightningCLI


def main():
    LightningCLI(
        parser_kwargs={"parser_mode": "omegaconf"},
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
