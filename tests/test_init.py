# Copyright (C) 2020 Pet Insight  Project - All Rights Reserved

import modeling_paper


def test_base_dir():
    assert "modeling_paper" in modeling_paper.base_dir
    print(modeling_paper.base_dir)
