#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2022
# @Author  : LYH
# @File    : activations.py
# @IDE     : pycharm
from enum import Enum

class Activations(Enum):
    relu = 0
    linear = 1

class ReluApproximation(Enum):
    ZERO = 0
    IDENTITY = 1
    PARALLEL = 2
    MIN_AREA = 3
    VENUS_HEURISTIC = 4
    OPT_HEURISTIC = 5

class ReluState(Enum):

    ACTIVE = 0
    INACTIVE = 1
    UNSTABLE = 2

    @staticmethod
    def inverse(s):
        """
        Inverts a given relu state.

        Arguments:

            s: ReluState Item

        Returns

            ReluState Item
        """
        if s == ReluState.INACTIVE:
            return ReluState.ACTIVE
        elif s == ReluState.ACTIVE:
            return ReluState.INACTIVE
        else:
            return None
