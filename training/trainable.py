# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import ray.tune

from models.reference_architectures import get_ref_arch
from training.train import Trainer


class MPTrainable(ray.tune.Trainable):
    def _setup(self, config={}):
        """Decimation is for speedup during unit testing only."""
        if config.get("base_config", False) and "model_config" not in config:
            # Use the requested 'base config', updating it with any other
            #  requested options.
            config["model_config"] = get_ref_arch(config["base_config"])
            print(f"Using base config: {config['base_config']}")

        self.trainer = trainer = Trainer(**config)
        self.trainer.init_data()
        self.trainer.init_train()

    def _train(self):
        epoch_record = self.trainer.train_one_epoch()
        d = epoch_record.to_dict()
        d["mean_loss"] = d["val_loss"]
        d["mean_accuracy"] = d["val_acc"]
        return d

    def _save(self, checkpoint_dir):
        """ Saves/checkpoints model state and training state to disk. """

        return self.trainer._save(checkpoint_dir)

    def _restore(self, checkpoint_dir):
        """ Restores model state and training state from disk. """
        return self.trainer._restore(checkpoint_dir)
