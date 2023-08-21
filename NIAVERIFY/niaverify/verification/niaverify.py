#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2022
# @Author  : LYH
# @File    : niaverify.py
# @IDE     : pycharm

import os
from tqdm import tqdm
from niaverify.input.query_reader import QueryReader
from niaverify.input.vnnlib_parser import VNNLIBParser
from niaverify.network.neural_network import NeuralNetwork
from niaverify.verification.verifier import Verifier
from niaverify.solver.solve_result import SolveResult
from niaverify.common.configuration import Config

class Niaverify:

    def __init__(
        self, 
        queries=None,
        nn=None,
        spec=None,
        config=None):
        """
        Arguments:

            queries:
                csv file of queries.

            nn:
                network file or folder of networks.
            
            spec:
                specification file or folder of specifications.

            config: 
                Configuration.
        """
        if queries is not  None:
            self.queries = QueryReader().read_from_csv(queries)
        elif nn is not None and spec is not None:
            self.queries = QueryReader().read_from_file(nn, spec)
        else:
            assert False, "Expected queries file or neural network and specification files."
        self.config = Config() if config is None else config
        if  nn.split('/')[-1][0] == 'A':
            self.config.BENCHMARK ='a'
        else:
            self.config.BENCHMARK = 'm'

    def verify(self):
        results = []
        safe, unsafe, undecided, timeout = 0, 0, 0, 0
        total_time, total_safe_time, total_unsafe_time = 0, 0, 0

        print('\n\n')
        pbar = tqdm(self.queries, desc='Verifying ', ncols=125)
        for query in pbar:
            pbar.set_description('Verifying ' +
                                 os.path.basename(query[0]) +
                                 ' against ' +
                                 os.path.basename(query[1]) +
                                 '...')
            # load model
            nn = NeuralNetwork(query[0], self.config)
            nn.load()
            self.config.set_nn_defaults(nn)
            # load spec
            vnn_parser = VNNLIBParser(
                query[1],
                nn.head.input_shape,
                self.config
            )  
            spec = vnn_parser.parse()
            # # verify
            ver_report = self.verify_query(nn, spec)
            if ver_report.result == SolveResult.SAFE:
                safe += 1
                total_safe_time += ver_report.runtime
            elif ver_report.result == SolveResult.UNSAFE:
                unsafe += 1
                total_unsafe_time += ver_report.runtime
            elif ver_report.result == SolveResult.UNDECIDED:
                undecided += 1
            elif ver_report.result == SolveResult.TIMEOUT:
                timeout += 1
            total_time += ver_report.runtime
            results.append(ver_report)

        avg_safe_time = 0 if safe == 0 else total_safe_time / safe
        avg_unsafe_time = 0 if unsafe == 0 else total_unsafe_time / unsafe

        with open(self.config.LOGGER.SUMFILE, 'a') as f:
            f.write(os.path.basename(query[0])+ '   ' + os.path.basename(query[1])+'\n')
            f.write('{:<12}{:6.4f}\n'.format(ver_report.result.value, ver_report.runtime))

        if self.config.VERIFIER.CONSOLE_OUTPUT:
            print(
                '\nVerified: {}\tSAFE: {}\tUNSAFE: {}\tUndecided: {}\tTimeouts: {}\n'.format(
                    safe + unsafe,
                    safe,
                    unsafe,
                    undecided,
                    timeout
                )
            )
            print(
                'Total Time:       {:6.4f}\tAvg Time:       {:6.4f}'.format(
                    total_time,
                    total_time / len(self.queries)
                )
            )
            print(
                'Total SAFE Time:   {:6.4f}\tAvg SAFE Time:   {:6.4f}'.format(
                    total_safe_time,
                    avg_safe_time
                )
            )
            print(
                'Total UNSAFE Time: {:6.4f}\tAvg UNSAFE Time: {:6.4f}'.format(
                    total_unsafe_time,
                    avg_unsafe_time
                )
            )
    
        return results[0] if len(results) == 1 else results

    def verify_query(self, nn, spec):
        time_elapsed = 0
        ver_report = None
        for subspec in spec:
            # create verifier
            verifier = Verifier(nn, subspec, self.config)
            sub_ver_report = verifier.verify()
            if ver_report is None:
                ver_report = sub_ver_report
            else:
                ver_report.runtime += sub_ver_report.runtime
                if sub_ver_report.result == SolveResult.UNSAFE:
                    ver_report.result = SolveResult.UNSAFE
                    return ver_report
                else:
                    time_left = self.config.SOLVER.TIME_LIMIT - sub_ver_report.runtime
                    if time_left <= 0:
                        ver_report.result = SolveResult.TIMEOUT
                        return ver_report
                    else:
                        self.config.SOLVER.TIME_LIMIT =  time_left

        return ver_report
