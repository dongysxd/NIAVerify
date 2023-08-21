#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : satisfaction_reader.py
# @IDE     : pycharm
from niaverify.solver.solve_result import SolveResult
import csv

class SatisfactionReader:

    def __init__(self, path):
        """
        Arguments:

            path: csv filepath
        """
        self.path = path
    
    def read_results(self):
        """
        Reads satisfaction results.

        Returns:
                
            A map of networks, specifications to satisfaction statuses
        """
        results = {}

        with open(self.path, 'r') as f:
            reader = csv.reader(f) 
            for row in reader:
                if row[2] == 'Safe':
                    results[(row[0], row[1])] = SolveResult.SAFE
                else:
                    results[(row[0], row[1])] = SolveResult.UNSAFE

        return results





