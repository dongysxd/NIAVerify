#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2022
# @Author  : LYH
# @File    : utils.py
# @IDE     : pycharm
import sys
import linecache
from enum import Enum

class DFSState(Enum):
    """
    States during DFS
    """
    UNVISITED = 'unvisited'
    VISITING = 'visiting'
    VISITED = 'visited'

class ReluState(Enum):

    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2

    @staticmethod
    def inverse(s):
        """
        Inverts a given relu state.

        Arguments:

            s:
                ReluState Item

        Returns

            ReluState Item
        """
        if s == ReluState.INACTIVE:
            return ReluState.ACTIVE

        elif s == ReluState.ACTIVE:
            return ReluState.INACTIVE

        return None

class ReluApproximation(Enum):
    ZERO = 0
    IDENTITY = 1
    PARALLEL = 2
    MIN_AREA = 3
    VENUS = 4

class OSIPMode(Enum):
    """
    Modes of operation.
    """
    OFF = 0
    ON = 1
    SPLIT = 2
