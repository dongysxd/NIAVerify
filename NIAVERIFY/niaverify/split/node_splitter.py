#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : node_splitter.py
# @IDE     : pycharm

from niaverify.verification.verification_problem import VerificationProblem
from niaverify.common.logger import get_logger
import torch
import numpy as np

class NodeSplitter(object):

    logger = None

    def __init__(self, initial_prob, config):
        """
        Arguments:

            initial_prob:
                VerificationProblem to split.

            config:
                configuration.
        """

        self.initial_prob = initial_prob
        self.config = config
        self.split_queue = [initial_prob]
        self.subprobs = []
        if NodeSplitter.logger is None:
            NodeSplitter.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def split(self):
        """
        Splits the  verification problem on top of the split_queue into a pair
        of subproblems. Splitting is via branching on the states of the ReLU
        node with the maximum dependency degree.

        Returns:

            list of VerificationProblem
        """
        if self.initial_prob.depth >= self.config.SPLITTER.BRANCHING_DEPTH:
            return  []
        split_flag = False
        while len(self.split_queue) > 0:
            prob = self.split_queue.pop()
            if prob.depth >= self.config.SPLITTER.BRANCHING_DEPTH:
                self.add_to_subprobs(prob)
                NodeSplitter.logger.info('Depth cutoff for node splitting reached.')
            else:
                output_eq = np.zeros(prob.nn.tail.output_size)
                #output_eq = torch.FloatTensor(self._objective.get_summed_constraints())
                if hasattr(prob.spec.output_formula, 'clauses'):
                    for idx  in range(len(prob.spec.output_formula.clauses)):
                        constr_eq =prob.spec.output_formula.clauses[idx]
                        op1 = constr_eq.op1.i
                        op2 = constr_eq.op2.i
                        output_eq[op1] += -1
                        output_eq[op2] += 1
                else:
                    op1 = prob.spec.output_formula.op1.i
                    output_eq[op1] += 1

                output_eq = torch.FloatTensor(output_eq)
                print(output_eq)
                impact = prob.get_most_impactfull_neurons(output_eq, lower=False)
                sn = impact[1]
                subprobs = self.split_node_by_impact(prob, sn[0])

                if len(subprobs) != 0:
                    for subprob in subprobs:
                         if subprob.stable_ratio_change_flag is True:
                             self.add_to_subprobs(subprob)
                         else:
                             self.add_to_split_queue(subprob)
                    split_flag = True
                else:
                    self.add_to_subprobs(prob)

        return self.subprobs if split_flag is True else []
       

    def split_node_by_impact(self,prob,nodeid):
        old_stability_ratio = prob.nn.get_stability_ratio()
        old_output_range = prob.nn.get_output_range()
        subprobs = []
        mid = None
        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        if nodeid[0] == 1:
            mid = (prob.nn.head.from_node[0].bounds.lower[0][nodeid[1]]  + prob.nn.head.from_node[0].bounds.upper[0][nodeid[1]]) / 2.0
            prob1.nn.head.from_node[0].bounds.lower[0][nodeid[1]] = mid
        else:
            prob1.nn.get_node_by_depth(nodeid[0])[0].from_node[0].bounds.lower[nodeid[1]] = 0
        prob1.bound_analysis()
        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        if nodeid[0] == 1:
            prob2.nn.head.from_node[0].bounds.upper[0][nodeid[1]] = mid 
        else:
            prob2.nn.get_node_by_depth(nodeid[0])[0].from_node[0].bounds.upper[nodeid[1]] = 0
        prob2.bound_analysis()
        new_stability_ratio1 = prob1.nn.get_stability_ratio()
        new_stability_ratio2 = prob2.nn.get_stability_ratio()
        prob1.stable_ratio_change_flag = False
        prob2.stable_ratio_change_flag = False
        if prob.config.BENCHMARK == 'm':
            if new_stability_ratio1 - old_stability_ratio < 0.01:
                 prob1.stable_ratio_change_flag = True
            if new_stability_ratio2 - old_stability_ratio < 0.01:
                 prob2.stable_ratio_change_flag = True
        subprobs.append(prob1)
        subprobs.append(prob2)
        return subprobs
    
    def split_node_by_impact_2(self,prob,nodeid):
        mid = None
        prob1 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )

        if nodeid[0] == 1:
            mid = (prob.nn.head.from_node[0].bounds.lower[0][nodeid[1]]  + prob.nn.head.from_node[0].bounds.upper[0][nodeid[1]]) / 2.0
            prob1.nn.head.from_node[0].bounds.lower[0][nodeid[1]] = mid
        else:
            prob1.nn.get_node_by_depth(nodeid[0])[0].from_node[0].bounds.lower[nodeid[1]] = 0
        prob1.bound_analysis()
    
        prob2 = VerificationProblem(
            prob.nn.copy(),
            prob.spec.copy(),
            prob.depth + 1,
            self.config
        )
        if nodeid[0] == 1:
            prob2.nn.head.from_node[0].bounds.upper[0][nodeid[1]] = mid 
        else:
            prob2.nn.get_node_by_depth(nodeid[0])[0].from_node[0].bounds.upper[nodeid[1]] = 0
        prob2.bound_analysis()
        return prob1,prob2


    def add_to_subprobs(self, prob):
        """
        Adds a verification subproblem to the subproblems list.

        Arguments:
            
            prob:
                VerificationProblem

        Returns:
            
            None
        """
        prob.bounds_ver_done = True
        self.subprobs = [prob] + self.subprobs
        # self.logger.info(f'Added subproblem {prob.id} to node subproblems list.')

    def add_to_split_queue(self, prob):
        """
        Adds a verification subproblem to the split queue.

        Arguments:
            
            prob:
                VerificationProblem

        Returns:
            
            None
        """
        self.split_queue = [prob] + self.split_queue
        # self.logger.info(f'Added subproblem {prob.id} to node split queue.')

