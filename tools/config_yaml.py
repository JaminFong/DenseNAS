# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
#
# Based on:
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
from io import IOBase
#from future.utils import iteritems
import copy
import logging
import numpy as np
import os
import os.path as osp
import yaml

from .collections import AttrDict
from .io import cache_url

logger = logging.getLogger(__name__)

def load_cfg(cfg_to_load):
    """Wrapper around yaml.load used for maintaining backward compatibility"""
    if isinstance(cfg_to_load, IOBase):
        cfg_to_load = ''.join(cfg_to_load.readlines())
    return yaml.load(cfg_to_load)


def load_cfg_to_dict(cfg_filename):
     with open(cfg_filename, 'r') as f:
         # yaml_cfg = AttrDict(load_cfg(f))
         yaml_cfg = load_cfg(f)
     return yaml_cfg


def merge_cfg_from_file(cfg_filename, global_config):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(load_cfg(f))
    _merge_a_into_b(yaml_cfg, global_config)


def merge_cfg_from_cfg(cfg_other, global_config):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, global_config)

def update_cfg_from_file(cfg_filename, global_config):
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(load_cfg(f))
    update_cfg_from_cfg(yaml_cfg, global_config)

def update_cfg_from_cfg(cfg_other, global_config, stack=None):
    assert isinstance(cfg_other, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(global_config, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in cfg_other.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)

        if k not in global_config:
            global_config[k] = v
            if isinstance(v, AttrDict):
                try:
                    stack_push = [k] if stack is None else stack + [k]
                    update_cfg_from_cfg(v, global_config[k], stack=stack_push)
                except BaseException:
                    raise

        else:
            # Recursively merge dicts
            v = _check_and_coerce_cfg_value_type(v, global_config[k], k, full_key)
            if isinstance(v, AttrDict):
                try:
                    stack_push = [k] if stack is None else stack + [k]
                    update_cfg_from_cfg(v, global_config[k], stack=stack_push)
                except BaseException:
                    raise
            else:
                global_config[k] = v


def merge_cfg_from_list(cfg_list, global_config):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = global_config
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), \
        '`a` (cur type {}) must be an instance of {}'.format(type(a), AttrDict)
    assert isinstance(b, AttrDict), \
        '`b` (cur type {}) must be an instance of {}'.format(type(b), AttrDict)

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v

def _key_is_deprecated(full_key):
    if full_key in _DEPRECATED_KEYS:
        logger.warn(
            'Deprecated config key (ignoring): {}'.format(full_key)
        )
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
        format(full_key, new_key, msg)
    )


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
