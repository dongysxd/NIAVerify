#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : split_strategy.py
# @IDE     : pycharm

from enum import Enum

class SplitStrategy(Enum):
    INPUT = "input"
    NODE = "node"
    NODE_INPUT = "node then input"
    INPUT_NODE = "input then node"
    INPUT_NODE_ALT = "alrernate node input"
    NONE = "no splitting"

    class NodeSplitStrategy(Enum):
        ONE_SPLIT = "one split per dependency graph"
        MULTIPLE_SPLITS = "multiple splits per dependency graph"

    @staticmethod
    def does_node_split(strategy):
        node_split_strategies = [
            SplitStrategy.NODE,
            SplitStrategy.NODE_INPUT,
            SplitStrategy.INPUT_NODE,
            SplitStrategy.INPUT_NODE_ALT
        ]

        return strategy in node_split_strategies
