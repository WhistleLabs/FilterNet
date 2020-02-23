# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import modeling_paper.datasets


def test_datasets_dir():
    assert "datasets" in modeling_paper.datasets.datasets_dir
    print(modeling_paper.datasets.datasets_dir)
