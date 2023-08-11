from typing import Any, Dict, List, Optional, Union

import pytest

from fluidml import utils


@pytest.fixture
def dict_1() -> Dict:
    return {
        "A": None,
        "B": {
            "hello": [1, 2, 3],
            "dream": True,
            "blub": None,
            "hi": {"C": False, "@D": "test"},
        },
        "@E": ["a", "b", None, {"c": None, "d": "fun"}],
        "F": {"G": None},
    }


@pytest.fixture
def dict_2() -> Dict:
    return {"A": None}


@pytest.fixture
def dict_3() -> Dict:
    return {
        "A": None,
        "B": {
            "hello": [1, 2, 3],
            "dream": True,
            "blub": None,
            "hi": {"C": (True, False), "D": "test"},
        },
        "E": {"c": None, "d": [1, 2, 3]},
    }


def test_update_merge(dict_3: Dict):
    d1 = {"a": 1, "b": [1, 2, 3], "c": {"d": "hi", "e": True}}
    d2 = {"a": 2, "b": [1, 2, 3, 4], "c": {"i": "hi", "e": False}}

    assert utils.update_merge(d1, d2) == {
        "a": (1, 2),
        "b": ([1, 2, 3], [1, 2, 3, 4]),
        "c": {"d": "hi", "i": "hi", "e": (True, False)},
    }


def test_reformat_config(dict_3: Dict):
    assert utils.reformat_config(dict_3) == {
        "A": None,
        "B": {
            "hello": [[1, 2, 3]],
            "dream": True,
            "blub": None,
            "hi": {"C": [True, False], "D": "test"},
        },
        "E": {"c": None, "d": [[1, 2, 3]]},
    }


def test_remove_none_from_dict(dict_1: Dict, dict_2: Dict):
    new_d1 = utils.remove_none_from_dict(dict_1)
    new_d2 = utils.remove_none_from_dict(dict_2)
    assert new_d1 == {
        "B": {"hello": [1, 2, 3], "dream": True, "hi": {"C": False, "@D": "test"}},
        "@E": ["a", "b", {"d": "fun"}],
        "F": {},
    }
    assert new_d2 == {}


def test_remove_prefixed_keys_from_dict(dict_1: Dict):
    new_d1 = utils.remove_prefixed_keys_from_dict(dict_1, prefix="@")
    assert new_d1 == {
        "A": None,
        "B": {"hello": [1, 2, 3], "dream": True, "blub": None, "hi": {"C": False}},
        "F": {"G": None},
    }


def test_remove_prefix_from_dict(dict_1: Dict):
    new_d1 = utils.remove_prefix_from_dict(dict_1, prefix="@")
    assert new_d1 == {
        "A": None,
        "B": {
            "hello": [1, 2, 3],
            "dream": True,
            "blub": None,
            "hi": {"C": False, "D": "test"},
        },
        "E": ["a", "b", None, {"c": None, "d": "fun"}],
        "F": {"G": None},
    }


def test_generate_run_name():
    run_name = utils.generate_run_name()
    assert len(run_name.split("-")) > 1 and isinstance(run_name, str)


@pytest.mark.parametrize(
    "type_annotation, optional",
    [
        (List[int], False),
        (Optional[str], True),
        (Union[str, None], True),
        (Dict, False),
    ],
)
def test_is_optional(type_annotation: Any, optional: bool):
    assert utils.is_optional(type_annotation) == optional


@pytest.mark.parametrize(
    "text, suffix, text_suffix_removed",
    [("some_file_name.jsonl.gz", ".jsonl.gz", "some_file_name"), ("aabcdabcd", "abcd", "aabcd")],
)
def test_remove_suffix(text: str, suffix: str, text_suffix_removed: str):
    assert utils.remove_suffix(text, suffix) == text_suffix_removed
