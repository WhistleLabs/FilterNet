# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import filternet.datasets


def test_datasets_dir():
    assert "datasets" in filternet.datasets.datasets_dir
    print(filternet.datasets.datasets_dir)
