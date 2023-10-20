from collections import defaultdict
from functools import partial
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

from fluidml.exception import GridSearchExpansionError


def _has_zip_identifier(params: List[Any], prefix: Optional[str] = None) -> bool:
    if prefix is None:
        return False
    return isinstance(params[-1], str) and params[-1].startswith(prefix)


# def _get_dict_obj(keys: Iterable, values: Iterable) -> Dict:
#     return {key: value for key, value in zip(keys, values)}
#
#
# def _expand_zip(param_grid: List[List[Any]], keys: List[str]):
#     max_param_list_len = max([len(param_list) for param_list in param_grid])
#     # if a parameter list holds only one element, repeat it max_param_list_len times.
#     param_grid = [x * max_param_list_len if len(x) == 1 else x for x in param_grid]
#     # check that all parameter grid lists are of same lengths
#     # if not all(len(param_grid[0]) == len(x) for x in param_grid[1:]):
#     #     raise ValueError('For method "zip" all expanded lists have to be of equal lengths.')
#
#     return [_get_dict_obj(keys, value) for value in zip(*param_grid)]


def _expand_product(
    param_grid: List[List[Any]],
    keys: List[str],
    param_grid_ids: List[List[int]],
    zip_identifiers: List[Optional[str]],
) -> List[Dict]:
    expanded_cfgs = []
    for value, ids in zip(product(*param_grid), product(*param_grid_ids)):
        d = {}
        for key, zip_identifier, val, id_ in zip(keys, zip_identifiers, value, ids):
            if zip_identifier is None:
                d[key] = val
            else:
                d[(key, zip_identifier, id_)] = val

        expanded_cfgs.append(d)
    return expanded_cfgs


def _expand(
    splits_by_keys: Dict,
    zips: Optional[Dict[str, set]] = None,
    method: str = "product",
    group_prefix: Optional[str] = None,
) -> List[Dict]:
    # get list of dict values and wrap all non-list values in lists for product expansion
    param_grid = [[params] if not isinstance(params, list) else params for params in splits_by_keys.values()]
    keys = list(splits_by_keys)

    # extract zip identifiers and zip position ids from param grid
    zip_identifiers = [
        params.pop(-1) if _has_zip_identifier(params, prefix=group_prefix) else None for params in param_grid
    ]
    param_grid_ids = [[i for i in range(len(pair))] for pair in param_grid]

    if method == "zip":
        zip_identifiers = [
            f"{group_prefix}x"
            if isinstance(params, List)
            and len(params) > 1
            and not (all(isinstance(param, Dict) for param in params))  # isinstance(params[0], Dict)
            else None
            for params in param_grid
        ]

    for zip_identifier, params in zip(zip_identifiers, param_grid):
        if zip_identifier:
            zips[zip_identifier].add(len(params))

    # TODO: Refactor logic to speed up many long lists via zip expansion (avoid product expansion + filtering)
    #  see test_config_expansion.py: cfg_11
    # if method == "zip":
    #     exp_configs: List[Dict] = _expand_zip(param_grid, keys)
    #     return exp_configs

    # expand config via product
    exp_configs: List[Dict] = _expand_product(param_grid, keys, param_grid_ids, zip_identifiers)

    return exp_configs


def expand_config_groups_and_parse_zip_identifiers(
    obj: Dict, zips: Dict, method: str = "product", group_prefix: Optional[str] = None
) -> Union[Any, List[Dict]]:
    """Recursively expands a dict on lists and parses zip identifiers.

    Based on a provided group prefix, the respective config keys are converted to tuples that keep track of the grouping
    which enables subsequent config filtering.

    Args:
        obj: A config dictionary
        zips: An initial defaultdict to store parsed zip identifiers
        method: Expansion method, e.g. "product" or "zip"
        group_prefix: Prefix to indicate zip grouping, default = "@"

    Returns:
        A List of expanded config groups, where grouped keys have been temporarily converted to tuples
    """

    if isinstance(obj, Dict):
        if not obj:  # in case of empty dict -> return dict without expanding
            return [obj]
        splits_per_key = {
            key: expand_config_groups_and_parse_zip_identifiers(child, zips, method, group_prefix)
            for key, child in obj.items()
        }
        return _expand(splits_per_key, zips, method, group_prefix)

    elif isinstance(obj, List):
        if len(obj) == 1 and isinstance(obj[0], List):
            return [obj]
        return [expand_config_groups_and_parse_zip_identifiers(item, zips, method, group_prefix) for item in obj]

    else:
        return obj


def expand_grouped_configs(obj: Dict, method: str = "product") -> Union[Any, List[Dict]]:
    """Recursively expands a dict on lists."""
    if isinstance(obj, Dict):
        if not obj:  # in case of empty dict -> return dict without expanding
            return [obj]
        splits_per_key = {key: expand_grouped_configs(child, method) for key, child in obj.items()}
        return _expand(splits_per_key, zips=None, method=method)
    elif isinstance(obj, List):
        if len(obj) == 1 and isinstance(obj[0], List):
            return obj
        return [splits for item in obj for splits in expand_grouped_configs(item, method)]
    else:
        return [obj]


def pop_zip_identifiers_from_config(obj: Dict, zips: Dict) -> Dict:
    """Recursively converts grouped keys in tuple format back to normal keys.

    Args:
        obj: A config dictionary.
        zips: A dict of parsed zip identifiers.

    Returns:
        A dictionary where recursively all prefixed keys have been removed.
    """

    if isinstance(obj, dict):
        cfg = {}
        for k, v in obj.items():
            if isinstance(k, tuple):
                k, identifier, count = k
                zips[identifier].add(count)
            cfg[k] = pop_zip_identifiers_from_config(v, zips)
        return cfg
    else:
        return obj


def filter_and_process_configs(expanded_cfgs: List[Dict]) -> List[Dict]:
    """Filters undefined configs based on previously parsed zip identifier tuple keys.

    Args:
        expanded_cfgs: Individual configs where tuple keys represent zip identifiers.

    Returns:
        Filtered and processed list of expanded configs
    """
    filtered_cfgs = []
    for cfg in expanded_cfgs:
        zip_counts = defaultdict(set)
        cleaned_cfg = pop_zip_identifiers_from_config(cfg, zip_counts)
        if all(True if len(zip_count) == 1 else False for zip_count in zip_counts.values()):
            filtered_cfgs.append(cleaned_cfg)
    return filtered_cfgs


def expand_default(config: Dict[str, Any], method: str = "product", group_prefix: Optional[str] = None) -> List[Dict]:
    if method == "zip" and group_prefix is None:
        group_prefix = "@"
    expanded_cfg_groups = expand_config_groups_and_parse_zip_identifiers(
        config, zips=defaultdict(set), method=method, group_prefix=group_prefix
    )
    expanded_cfgs = [cfg for group in expanded_cfg_groups for cfg in expand_grouped_configs(group, method)]
    filtered_cfgs = filter_and_process_configs(expanded_cfgs)
    return filtered_cfgs


def expand_config(
    config: Dict[str, Any],
    expand: Optional[str] = None,
    group_prefix: Optional[str] = None,
) -> List[Dict]:
    if expand is not None:
        try:
            expansion_method = ConfigExpansionRegistry.get(expand)
        except KeyError:
            raise GridSearchExpansionError(
                f"Expansion method {expand} is not implemented. "
                f"Choose from {list(ConfigExpansionRegistry.registered_ids())} "
                f"or register your own method via the 'ConfigExpansionRegistry.add()' function."
            )

        expanded_configs: List[Dict] = expansion_method(config, group_prefix=group_prefix)
    else:
        expanded_configs: List[Dict] = [config]

    return expanded_configs


class ConfigExpansionRegistry:
    _registry = {
        "product": partial(expand_default, method="product"),
        "zip": partial(expand_default, method="zip"),
    }

    @classmethod
    def registered_ids(cls):
        return cls._registry

    @classmethod
    def get(cls, cfg_expansion_fn_id: str) -> Callable:
        expansion_fn = cls._registry[cfg_expansion_fn_id]
        return expansion_fn

    @classmethod
    def add(cls, id_: str, expansion_fn: Callable):
        ConfigExpansionRegistry._registry[id_] = expansion_fn
