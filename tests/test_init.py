# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import filternet


def test_base_dir():
    assert "filternet" in filternet.base_dir
    print(filternet.base_dir)
