#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : solve_result.py
# @IDE     : pycharm
from enum import Enum

class SolveResult(Enum):
    TIMEOUT = 'timeout'
    INTERRUPTED = 'interrupted'
    BRANCH_THRESHOLD = 'branch-threshold'    
    SAFE = 'safe'
    UNSAFE = 'unsafe'
    UNDECIDED = 'undecided'
