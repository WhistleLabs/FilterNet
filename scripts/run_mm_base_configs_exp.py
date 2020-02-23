# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

"""
Executes a series of benchmarking runs to be used in plots & tables in the paper:
* Runs for 'multimodal sensor fusion analysis'

"""

import sys

sys.path.insert(0, ".")

from models.reference_architectures import ref_archs
from training.evalmodel import *
from training.ensemble_train import EnsembleTrainer

NAME = "mm_base_configs_2"  # unique name for this particular run
MAX_EPOCHS = 100  # 0
NUM_REPEATS = 5  # 20
saved_model_glob = (
    f"saved_models/{NAME}*/evalmodel.pth"  # helps NB's to load these models
)

# Iterate through these architectures
ref_archs = [
    # 'deepconvlstm',  # slow
    "base_cnn",
    "base_lstm",
    "cnn_lstm",
    "multi_scale_cnn",
    "multi_scale_cnn_lstm",
]

# For each architecture, train/eval on these sensor subsets
sensor_subsets = [
    "accels",
    "gyros",
    "accels+gyros",
    "accels+gyros+magnetic",
    "opportunity",
]


def do_run():
    for i in range(1, NUM_REPEATS):
        # Do the FilterNet reference architectures
        for ref_arch in ref_archs:
            for sensor_subset in sensor_subsets:
                name = f"{NAME}_{ref_arch}_{sensor_subset}_{i}"

                config = {}
                config["base_config"] = ref_arch
                config["name"] = name
                config["sensor_subset"] = sensor_subset

                trainer = Trainer(**config)
                trainer.init_data()
                trainer.init_train()

                trainer.train(max_epochs=MAX_EPOCHS)

                em = EvalModel(trainer=trainer)
                em._save()

                # Load fresh for consistency
                em = load_eval_model_from_dir(em.model_path)

                em.run_test_set()
                em.calc_metrics()
                em.calc_ward_metrics()
                print(em.classification_report_df)
                print(f"Weighted F1: {em.f1:.4f}")
                print(f"Event F1: {em.event_f1:.4f}")
                print(f"Nonull F1: {em.nonull_f1:.4f}")
                em._save()

        # And also do the 4-fold ensemble, which requires slightly different code
        for sensor_subset in sensor_subsets:
            num_folds = 4

            name = f"{NAME}_{num_folds}_folds_{sensor_subset}_{i}"

            config = {"base_config": "multi_scale_cnn_lstm", "model_config": {}}
            config["sensor_subset"] = sensor_subset

            trainer = EnsembleTrainer(n_folds=num_folds, name=name, config=config)
            trainer.init_data()

            trainer.train(max_epochs=MAX_EPOCHS)
            trainer._save()

            em = EvalModel(trainer=trainer)
            em._save()

            # Load fresh for consistency
            em = load_eval_model_from_dir(em.model_path)

            # annotate extra field, to make post-analysis easier.
            em.extra["exp_name"] = NAME
            em.extra["num_folds"] = num_folds
            em.extra["i_repeat"] = i

            em.run_test_set()
            em.calc_metrics()
            em.calc_ward_metrics()
            print(em.classification_report_df)
            em._save()


if __name__ == "__main__":
    do_run()
