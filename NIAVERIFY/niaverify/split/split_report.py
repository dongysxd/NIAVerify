#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : split_report.py
# @IDE     : pycharm

class SplitReport:
    def __init__(self, id, jobs_count, node_split_count, input_split_count):
        self.id = id,
        self.jobs_count = jobs_count
        self.node_split_count = node_split_count
        self.input_split_count = input_split_count
