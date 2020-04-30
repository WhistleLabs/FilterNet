# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

"""
Executes a series of benchmarking runs to be used in plots & tables in the paper:
* Ensembled MS-C/L models with various #'s of folds / submodels
"""

import sys

sys.path.insert(0, ".")

from filternet.training.evalmodel import *
from filternet.training.ensemble_train import EnsembleTrainer

NAME = "ensembles_3"  # unique name for this particular run
MAX_EPOCHS = 100
NUM_REPEATS = 10
NUM_FOLDS = [2, 3, 4, 5]
saved_model_glob = (
    f"saved_models/{NAME}*/evalmodel.pth"  # helps NB's to load these models
)


def do_run():
    for i in range(0, NUM_REPEATS):
        # Do the FilterNet reference architectures
        for num_folds in NUM_FOLDS:
            name = f"{NAME}_{num_folds}_folds_{i}"

            config = {"base_config": "multi_scale_cnn_lstm", "model_config": {}}

            trainer = EnsembleTrainer(n_folds=num_folds, name=name)
            trainer.init_data()

            trainer.train(max_epochs=MAX_EPOCHS)
            trainer._save()

            em = EvalModel(trainer=trainer)

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
