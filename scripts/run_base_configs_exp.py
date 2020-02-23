# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

"""
Executes a series of benchmarking runs to be used in plots & tables in the paper:
* FilterNet reference configurations
* DeepConvLSTM reimpplementation
* .5x ms-c/l
# 2X ms-c/l

"""

import sys

sys.path.insert(0, ".")

from models.reference_architectures import ref_archs
from training.evalmodel import *

NAME = "base_configs_7"  # unique name for this particular run
MAX_EPOCHS = 100
NUM_REPEATS = 10
saved_model_glob = (
    f"saved_models/{NAME}*/evalmodel.pth"  # helps NB's to load these models
)


def do_run():
    for i in range(1, NUM_REPEATS):
        # Do the FilterNet reference architectures
        for ref_arch in ref_archs.keys():
            name = f"{NAME}_{ref_arch}_{i}"

            config = {}
            config["base_config"] = ref_arch
            config["name"] = f"{NAME}_{ref_arch}_{i}"

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
            em._save()

        # Also do a matching number of DeepConvLSTMs
        name = f"{NAME}_deepconvlstm_{i}"

        config = {
            "win_len": 24,
            "batch_size": 100,
            "model_class": "DeepConvLSTM",
            "model_config": {"scale": 1.0},
        }
        # config["base_config"] = ref_arch
        # config["model_config"] = {} #get_ref_arch("multi_scale_cnn_lstm")
        config["name"] = name

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
        em._save()

        # Also do a matching number of .5x scale models.
        ref_arch = "multi_scale_cnn_lstm"

        name = f"{NAME}_mscl_p5x_{i}"

        config = {}
        config["base_config"] = ref_arch
        config["model_config"] = {"scale": 0.5}
        config["name"] = name

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
        em._save()

        # Also do a matching number of 2x scale models.
        ref_arch = "multi_scale_cnn_lstm"

        name = f"{NAME}_mscl_2x_{i}"

        config = {}
        config["base_config"] = ref_arch
        config["model_config"] = {"scale": 2}
        config["name"] = name

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
        em._save()

        # Also do a matching number of .5x scale models.
        ref_arch = "multi_scale_cnn_lstm"

        name = f"{NAME}_mscl_p5x_{i}"

        config = {}
        config["base_config"] = ref_arch
        config["model_config"] = {"scale": 0.5}
        config["name"] = name

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
        em._save()


if __name__ == "__main__":
    do_run()
