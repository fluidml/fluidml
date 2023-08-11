from typing import Dict

import pytest

from fluidml.config_expansion import ConfigExpansionRegistry, expand_config
from fluidml.exception import GridSearchExpansionError

# correct: 1
cfg_1 = {}

# correct: 9 (product) = 2 * 3 + 1 * 3
cfg_2 = {
    "a": [{"a": [1, 2]}, {"a": 4}],
    "b": {"c": [1, 2, 3], "d": "hallo", "e": {}},
    "c": {"test": True, "funny": {"x": [[1, 2, 3, 4]], "y": False}},
}

# correct: 8 (product)
cfg_3 = {
    "param_1": [{"a": [1, 2, "@b"]}, {"a": [1, 2, "@b"]}, "@x"],
    "param_3": [
        {
            "param_3_2": [3, 4, "@abc"],
            "param_3_3": {"a": 99, "b": [[5, 6]]},
            "bla": [1, 2, "@b"],
            "blub": {"c": [1, 2, "@abc"], "d": [3, 4, "@abc"]},
        },
        {
            "param_3_4": 99,
            "bla": [1, 2],
        },
        "@x",
    ],
}

# correct: 4 (product)
cfg_4 = {
    "param_1": [{"a": [1, 2, "@b"]}, {"a": [1, 2, "@b"]}, "@x"],
    "param_3": [
        {
            "param_3_2": [3, 4, "@b"],
            "param_3_3": {"a": 99, "b": [[5, 6]]},
            "bla": [1, 2, "@b"],
            "blub": {"c": [1, 2, "@b"], "d": [3, 4, "@b"]},
        },
        {
            "param_3_4": 99,
            "bla": [1, 2, "@b"],
        },
        "@x",
    ],
}

# correct: 72 (product) = 2*16 + 2*16 + 2*2 + 2*2
cfg_5 = {
    "param_1": [{"a": [1, 2]}, {"a": [1, 2]}],
    "param_3": [
        {
            "param_3_2": [3, 4],
            "param_3_3": {"a": 99, "b": [[5, 6]]},
            "bla": [1, 2],
            "blub": {"c": [1, 2], "d": [3, 4]},
        },
        {
            "param_3_4": 99,
            "bla": [1, 2],
        },
    ],
}

# correct: 8 (product) = 2 * 2 + 1 * 2 * 2
# note the 3 in the second row list is ignored due to zipping
cfg_6 = {
    "a": [{"a": [1, 2, "@x"]}, {"a": 4}],
    "b": {"c": [1, 2, 3, "@x"]},
    "c": {"test": [True, False], "funny": {"y": [True, False, "@x"]}},
}

# correct: 2 (product)
# note the first 2 is ignored
cfg_7 = {
    "a": [{"a": [1, 2, "@x"]}, {"a": 4}, "@x"],
    "b": {"c": [1, 2, "@x"]},
    "c": {"test": True, "funny": {"x": [[[1, 2, 3, 4]]], "y": False}},
}

# correct number of expanded configs: 4
cfg_8 = {
    "a": [{"a": [1, 2, "@x"]}, {"a": 4}, "@z"],
    "b": {"c": [1, 2, 3, "@x"], "e": {}},
    "c": {
        "test": [True, False, "@z"],
        "funny": {"x": [[[1, 2, 3, 4]]], "y": [True, False, "@x"]},
    },
}

# correct: 2 (product/zip)
cfg_9 = {
    "a": [
        {"b": [{"c": [{"d": [1, 2, "@x"]}, {"f": 3}, "@x"]}, {"g": 3}, "@x"]},
        {"e": 4},
        "@x",
    ],
}

# correct: 2 (product/zip)
cfg_10 = {
    "a": [{"b": [{"d": [1, 2, "@x"]}, {"e": 3}, "@x"]}, {"c": 4}, "@x"],
}


@pytest.mark.parametrize(
    "config, expansion_method, num_expanded_cfgs",
    [
        (cfg_1, "product", 1),
        (cfg_2, "product", 9),
        (cfg_2, "zip", 2),
        (cfg_3, "product", 8),
        (cfg_4, "product", 4),
        (cfg_5, "product", 72),
        (cfg_5, "zip", 2),
        (cfg_6, "product", 8),
        (cfg_7, "product", 2),
        (cfg_8, "product", 4),
        (cfg_9, "product", 2),
        (cfg_9, "zip", 2),
        (cfg_10, "product", 2),
        (cfg_10, "zip", 2),
        (cfg_10, "undefined", 2),
    ],
)
def test_expand_config(config: Dict, expansion_method: str, num_expanded_cfgs: int):
    if expansion_method not in ConfigExpansionRegistry.registered_ids():
        with pytest.raises(GridSearchExpansionError) as _:
            assert len(expand_config(config=config, expand=expansion_method, group_prefix="@")) == num_expanded_cfgs
    else:
        assert len(expand_config(config=config, expand=expansion_method, group_prefix="@")) == num_expanded_cfgs
