#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : mnistfc.py
# @IDE     : pycharm
import unittest
import sys
sys.path.append('..')
from niaverify.verification.niaverify import Niaverify
from niaverify.tests.verification.satisfaction_reader import SatisfactionReader

class TestMnistFC(unittest.TestCase):

    def setUp(self):
        self.true_results = SatisfactionReader('../niaverify/tests/data/mnistfc/queries.csv').read_results()
        self.properties = {
            1 : [(1, x, y) for x in range(1, 6) for y in range(1, 10)],
            2 : [(2, x, y) for x in range(1, 6) for y in range(1, 10)],
            3 : [(3, x, y) for x in range(1, 6) for y in range(1, 10)],
            4 : [(4, x, y) for x in range(1, 6) for y in range(1, 10)],
            5 : [(5, 1, 1)],
            6 : [(6, 1, 1)],
            7 : [(7, 1, 9)],
            8 : [(8, 2, 9)],
            9 : [(9, 3, 3)],
            10 : [(10, 4, 5)]
        }


    def test_fc(self):
        """
        Tests the verification results MNISTFC.
        """
        for (nn, spec) in self.true_results:

            niaverify = Niaverify(
                nn='../niaverify/tests/data/mnistfc/' + nn,
                spec='../niaverify/tests/data/mnistfc/' + spec
            )
            report = niaverify.verify()

if __name__ == '__main__':
    a = TestMnistFC()
    a.setUp()
    a.test_fc()