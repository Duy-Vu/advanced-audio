#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import List, Union
import os
import pathlib


__docformat__ = 'reStructuredText'
__all__ = ['get_files_from_dir_with_os', 'get_files_from_dir_with_pathlib']


def get_files_from_dir_with_os(dir_name: str) \
        -> List[str]:
    """Returns the files in the directory `dir_name` using the os package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[str]
    """
    return os.listdir(dir_name)


def get_files_from_dir_with_pathlib(dir_name: Union[str, pathlib.Path]) \
        -> List[pathlib.Path]:
    """Returns the files in the directory `dir_name` using the pathlib package.

    :param dir_name: The name of the directory.
    :type dir_name: str
    :return: The filenames of the files in the directory `dir_name`.
    :rtype: list[pathlib.Path]
    """
    return list(pathlib.Path(dir_name).iterdir())

# EOF