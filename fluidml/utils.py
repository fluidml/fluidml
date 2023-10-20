import contextlib
import datetime
import glob
import hashlib
import json
import logging
import os
import pickle
import random
import shutil
import string
from typing import Any, Dict, Optional, Tuple, Union

from pydantic import BaseModel as BaseModel_


class BaseModel(BaseModel_):
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True


def update_merge(d1: Dict, d2: Dict) -> Union[Dict, Tuple]:
    """Recursively merges two dictionaries.

    Conflicting keys won't be updated but expanded to a tuple.

    Args:
        d1: A dictionary.
        d2: A second dictionary to be merged.

    Returns:
        A recursively merged dictionary.

    """
    if isinstance(d1, dict) and isinstance(d2, dict):
        # Unwrap d1 and d2 in new dictionary to keep non-shared keys with **d1, **d2
        # Next unwrap a dict that treats shared keys
        # If two keys have an equal value, we take that value as new value
        # If the values are not equal, we recursively merge them
        return {
            **d1,
            **d2,
            **{k: d1[k] if d1[k] == d2[k] else update_merge(d1[k], d2[k]) for k in sorted({*d1} & {*d2})},
        }
    else:
        # This case happens when values are merged
        # It bundle values in a tuple, assuming the original dicts
        # don't have tuples as values
        if isinstance(d1, tuple) and not isinstance(d2, tuple):
            combined = d1 + (d2,)
        elif isinstance(d2, tuple) and not isinstance(d1, tuple):
            combined = (d1,) + d2
        elif isinstance(d1, tuple) and isinstance(d2, tuple):
            combined = d1 + d2
        else:
            combined = (d1, d2)
        return tuple(element for i, element in enumerate(combined) if element not in combined[:i])


def reformat_config(d: Dict) -> Dict:
    """Recursively re-formats config dictionaries.

    Converts nested lists to double lists, e.g. ``["a", "b"]`` to ``[["a", "b"]]`` and
    nested tuples to normal lists, e.g. ``("a", "b")`` to ``["a", "b"]``.

    Args:
        d: A dictionary.

    Returns:
        A re-formatted dictionary.
    """

    for key, value in d.items():
        if isinstance(value, list):
            d[key] = [value]
        if isinstance(value, tuple):
            d[key] = list(value)
        elif isinstance(value, dict):
            reformat_config(value)
        else:
            continue
    return d


def remove_none_from_dict(obj: Dict) -> Dict:
    """Recursively removes ``None`` values from dictionary.

    Args:
        obj: A dictionary (e.g. a config) to be cleaned for ``None`` values.

    Returns:
        A new dictionary where recursively all ``None`` values have been removed.
    """

    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_none_from_dict(x) for x in obj if x is not None)
    elif isinstance(obj, dict):
        return {k: remove_none_from_dict(v) for k, v in obj.items() if k is not None and v is not None}
    else:
        return obj


def remove_value_from_dict(obj: Dict, value: Any) -> Dict:
    """Recursively removes ``{}`` values from dictionary.

    Args:
        obj: A dictionary (e.g. a config) to be cleaned for ``{}`` values.
        value: The value to be removed.

    Returns:
        A new dictionary where the provided value is recursively removed.
    """

    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_value_from_dict(x, value) for x in obj if x != value)
    elif isinstance(obj, dict):
        return {k: remove_value_from_dict(v, value) for k, v in obj.items() if v != value}
    else:
        return obj


def remove_prefixed_keys_from_dict(obj: Dict, prefix: Optional[str] = None) -> Dict:
    """Recursively removes key-value pairs that are prefixed.

    Args:
        obj: A dictionary.
        prefix: A string prefix indicating which key-value pairs to remove.

    Returns:
        A new dictionary where recursively all prefixed keys have been removed.
    """
    if prefix is None:
        return obj

    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_prefixed_keys_from_dict(x, prefix) for x in obj)
    elif isinstance(obj, dict):
        return {
            k: remove_prefixed_keys_from_dict(v, prefix)
            for k, v in obj.items()
            if not isinstance(k, str) or (isinstance(k, str) and not k.startswith(prefix))
        }
    else:
        return obj


def remove_prefix_from_dict(obj: Dict, prefix: Optional[str] = None) -> Dict:
    """Recursively removes the prefix from keys in a dictionary.

    Args:
        obj: A dictionary.
        prefix: A dictionary key prefix.

    Returns:
        A new dictionary where recursively all prefixes have been removed.
    """
    if prefix is None:
        return obj

    if isinstance(obj, (list, tuple, set)):
        return type(obj)(remove_prefix_from_dict(x, prefix) for x in obj)
    elif isinstance(obj, dict):
        return {
            (k.split(prefix, 1)[-1] if isinstance(k, str) and k.startswith(prefix) else k): remove_prefix_from_dict(
                v, prefix
            )
            for k, v in obj.items()
        }
    else:
        return obj


def generate_run_name() -> str:
    """Randomly creates a run name consisting of an adjective and a noun.

    Returns:
        A randomly generated run_name consisting of adjective and noun.
    """
    from fluidml import package_path

    adj_path = os.path.join(package_path, "word_lists", "adjectives.txt")
    noun_path = os.path.join(package_path, "word_lists", "nouns.txt")

    # load nouns and adjectives (source Wordnet 3.1)
    with open(noun_path, "r") as noun_f:
        nouns = [line.strip() for line in noun_f if "_" not in line]

    with open(adj_path, "r") as adj_f:
        adjectives = [line.strip() for line in adj_f]

    noun = random.choice(nouns)
    adjective = random.choice(adjectives)

    run_name = f"{adjective}-{noun}"
    return run_name


def is_optional(obj: Any):
    """Check if a type annotation is Optional

    Make sure that the annotation has the keys __module__, __args__ and __origin__
    1. __origin__ value is equal to typing.Union
    2. __args__ must be a tuple which contains the NoneType (type(None))
    3. __module__ must be set to 'typing'. If not, the annotation is not an object defined by the typing module
    """
    return (
        getattr(obj, "__origin__", None) is Union
        and type(None) in getattr(obj, "__args__", ())
        and getattr(obj, "__module__", None) == "typing"
    )


def hash_config(cfg: Dict) -> str:
    """Creates md5 hash in hex of config dictionary.

    Args:
        cfg: Json serializable unique config.

    Returns:
        Hashed config (md5 hash in hex)
    """
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


def encode_base36(number: int) -> str:
    """Encodes an integer in base36

    Args:
        number: Any integer

    Returns:
        Base36 encoded integer (contains a-z,0-9)
    """
    alphabet, base36 = string.digits + string.ascii_lowercase, ""
    while number:
        number, i = divmod(number, 36)
        base36 = alphabet[i] + base36
    return base36 or alphabet[0]


def create_unique_hash_from_config(cfg: Dict, length: int = 8) -> str:
    """Creates an 8 char long unique id for input config dict.
    Uses md5 hashing and base36 encoding to be filename compatible.

    Args:
        cfg: Config dictionary.
        length: Char length of unique run id.

    Returns:
        Unique run id in base36, based on input config.
    """
    hashed_cfg = hash_config(cfg)
    encoded_cfg = encode_base36(int(hashed_cfg, base=16))
    return encoded_cfg[:length]


@contextlib.contextmanager
def change_logging_level(level: int):
    """Context manager to temporarily change the logging lvl."""
    root_logger = logging.getLogger()
    old_logging_level = root_logger.level
    try:
        root_logger.setLevel(level)
        yield
        root_logger.setLevel(old_logging_level)
    except Exception:
        root_logger.setLevel(old_logging_level)
        raise


def remove_suffix(text: str, suffix: str) -> str:
    if text.endswith(suffix):
        return text[: -len(suffix)]
    return text


def convert_legacy_file_store_structure(path: str):
    """Converts the fluidml 0.2.X file store structure to the new 0.3.X structure.

    Args:
        path: The base directory of legacy experiments. Recursively converts the entire directory to the new structure.
    """

    # recursively get all ".load_info" directories
    load_info_dirs = list(glob.iglob(os.path.join(path, "**", ".load_info"), recursive=True))

    for load_info_dir in load_info_dirs:
        # get run dir
        run_dir = os.path.split(load_info_dir)[0]

        # load config
        config_path = os.path.join(run_dir, "config.json")
        config = json.load(open(config_path, "r"))

        # create config hash
        config_hash = create_unique_hash_from_config(cfg=config)

        # infer run's start time based on config.json creation/modification date
        start_time = datetime.datetime.fromtimestamp(os.path.getmtime(config_path))

        # infer run's end_time based on creation/modification time of last saved result object
        end_time = max(
            datetime.datetime.fromtimestamp(os.path.getmtime(path))
            for path in glob.iglob(os.path.join(run_dir, "**"), recursive=True)
            if os.path.isfile(path)
        )
        duration = end_time - start_time

        # copy legacy ".load_info" directory into new ".fluidml" directory if no ".fluidml" dir exists
        fluidml_dir = os.path.join(run_dir, ".fluidml")
        if os.path.isdir(fluidml_dir):
            continue
        shutil.copytree(src=load_info_dir, dst=fluidml_dir)

        # get saved file names based on load_info files
        names = []
        for path in glob.glob(os.path.join(fluidml_dir, ".*_load_info.p")):
            name = remove_suffix(os.path.split(path)[-1].lstrip("."), suffix="_load_info.p")
            names.append(name)

        # remove legacy completed file from list of saved objects
        filtered_names = [name for name in names if name != "completed"]

        # create new ".saved_results.json" file
        json.dump(filtered_names, open(os.path.join(fluidml_dir, ".saved_results.json"), "w"))
        # create new ".saved_results_load_info.p file
        saved_results_load_info = {
            "open_kwargs": {},
            "load_kwargs": {},
            "obj_dir": ".fluidml",
            "type_": "json",
        }
        pickle.dump(
            saved_results_load_info,
            open(os.path.join(fluidml_dir, ".saved_results_load_info.p"), "wb"),
        )

        # create new fluidml_info.json file in run_dir
        finished = "finished" if "completed" in names else "failed"
        run_id = os.path.split(run_dir)[-1]
        run_dir_name = run_id.rsplit("-", 1)
        if len(run_dir_name) == 1 and run_dir_name[0].isnumeric():
            run_name = run_dir_name[0]
            sweep_counter = run_dir_name[0]
        elif len(run_dir_name) == 1:
            run_name = run_dir_name[0]
            sweep_counter = "000"
        else:
            run_name = run_dir_name[0]
            sweep_counter = run_dir_name[1] if run_dir_name[1].isnumeric() else "000"

        fluidml_info = {
            "project_name": "fluidml",
            "run_name": run_name,
            "state": finished,
            "started": str(start_time),
            "ended": str(end_time),
            "duration": str(duration),
            "run_history": {},  # this is currently not retrieved
            "sweep_counter": sweep_counter,
            "unique_config_hash": config_hash,
            "id": run_id,
        }
        json.dump(
            fluidml_info,
            open(os.path.join(run_dir, "fluidml_info.json"), "w"),
            indent=4,
        )

        # create new .fluidml_info_load_info.p file
        fluidml_info_load_info = {"open_kwargs": {}, "load_kwargs": {}, "obj_dir": ".", "type_": "json"}
        pickle.dump(
            fluidml_info_load_info,
            open(os.path.join(fluidml_dir, ".fluidml_info_load_info.p"), "wb"),
        )

        # remove .completed files from .fluidml dir
        for file in glob.glob(os.path.join(fluidml_dir, ".completed*")):
            os.remove(file)
