#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : acasxu.py
# @IDE     : pycharm

import unittest
import sys
sys.path.append('../..')
from niaverify.verification.niaverify import Niaverify
from niaverify.tests.verification.satisfaction_reader import SatisfactionReader

class TestAcasXU(unittest.TestCase):

    def setUp(self):
        self.true_results = SatisfactionReader('../tests/data/acasxu/queries.csv').read_results()
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


    def verify(self, p, x, y):
        """
        Verifies and tests the verification result of an ACASXU property.

        Arguments:           
            p:
                int of the property
            x, y:
                ints of the network number
        """

        nn =  f'nets/ACASXU_run2a_{x}_{y}_batch_2000.onnx'
        spec = f'specs/prop_{p}.vnnlib'
        niaverify = Niaverify(
            nn='../tests/data/acasxu/' + nn,
            spec='../tests/data/acasxu/' + spec
        )
        report = niaverify.verify()
       # self.assertTrue(self.true_results[(nn, spec)] == report.result)

    def test_property1(self):
        """
        Tests acasxu property 1
        """
        for p, x, y in self.properties[1]:
            #if x ==4 and y >6  or x > 4:
            self.verify(p, x, y)
            #break

    def test_property2(self):
        """
        Tests acasxu property 2
        """
        for p, x, y in self.properties[2]:
            self.verify(p, x, y)

    def test_property3(self):
        """
        Tests acasxu property 3
        """
        for p, x, y in self.properties[3]:
            self.verify(p, x, y)


    def test_property4(self):
        """
        Tests acasxu property 4
        """
        for p, x, y in self.properties[4]:
            self.verify(p, x, y)

    def test_property5(self):
        """
        Tests acasxu property 5
        """
        for p, x, y in self.properties[5]:
            self.verify(p, x, y)

    def test_property6(self):
        """
        Tests acasxu property 6
        """
        for p, x, y in self.properties[6]:
            self.verify(p, x, y)

    def test_property7(self):
        """
        Tests acasxu property 7
        """
        for p, x, y in self.properties[7]:
            self.verify(p, x, y)

    def test_property8(self):
        """
        Tests acasxu property 8
        """
        for p, x, y in self.properties[8]:
            self.verify(p, x, y)

    def test_property9(self):
        """
        Tests acasxu property 9
        """
        for p, x, y in self.properties[9]:
            self.verify(p, x, y)

    def test_property10(self):
        """
        Tests acasxu property 10
        """
        for p, x, y in self.properties[10]:
            self.verify(p, x, y)

if __name__ == '__main__':
    a = TestAcasXU()
    a.setUp()
    a.test_property1()
    a.test_property2()
    a.test_property3()
    a.test_property4()
    # a.test_property5()
    # a.test_property6()
    # a.test_property7()
    # a.test_property8()
    #a.test_property9()
   # a.test_property10()
    