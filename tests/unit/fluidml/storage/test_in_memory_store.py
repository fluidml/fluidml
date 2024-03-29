from multiprocessing import Manager

import pytest

from fluidml.storage import InMemoryStore, ResultsStore


@pytest.fixture
def in_memory_store():
    with Manager() as manager:
        store = InMemoryStore(manager)
        yield store


def test_first_load(in_memory_store: ResultsStore):
    loaded_obj = in_memory_store.load("dummy_item", "dummy_task", {})
    assert loaded_obj is None


def test_save_load(in_memory_store: ResultsStore):
    test_item_name = "dummy_item"
    test_task_name = "dummy_task"
    test_obj_1 = {"dummy_result": "test_1"}
    test_obj_2 = {"dummy_result": "test_2"}
    test_config = {"config_param": 1}

    # save
    in_memory_store.save(
        obj=test_obj_1,
        name=test_item_name,
        task_name=test_task_name,
        task_unique_config=test_config,
    )

    # replace
    in_memory_store.save(
        obj=test_obj_2,
        name=test_item_name,
        task_name=test_task_name,
        task_unique_config=test_config,
    )
    # load
    loaded_json_obj = in_memory_store.load(
        name=test_item_name, task_name=test_task_name, task_unique_config=test_config
    )

    assert loaded_json_obj == test_obj_2
