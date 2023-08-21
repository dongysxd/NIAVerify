#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : lp_cuts.py
# @IDE     : pycharm

from niaverify.network.node import Relu
from niaverify.solver.cuts import Cuts
from niaverify.common.logger import get_logger
from timeit import default_timer as timer
from gurobipy import *
import numpy as np


class LPCuts(Cuts):

    logger = None

    def __init__(self, prob, gmodel, config):
        """
        Arguments:

            prob: 
                VerificationProblem.
            gmodel:
                Gurobi model.
            config:
                Configuration.
        """
        super().__init__(prob, gmodel, config.SOLVER.IDEAL_FREQ, config)
        if LPCuts.logger is None:
            LPCuts.logger = get_logger(__name__, config.LOGGER.LOGFILE)


    def add_cuts(self):
        """
        Adds ideal cuts.

        Arguments:

            model: Gurobi model.
        """
        cuts = self.build_cuts()

    def build_cuts(self):
        """
        Constructs ideal cuts.
        """
        ts = timer()
        
        for idx in range(len(self.prob.dep_by_lp_ls)):
            ls = []
            dep_by_lp = self.prob.dep_by_lp_ls[idx]
            for element in dep_by_lp:
                i  = self.prob.nn.get_node_by_depth(element[0][0])[0] 
                delta = self.get_var_values_lp(i, 'delta')
                ls.append((delta[element[0][1]],element[1]))

            for j in range(len(ls)):
                le = LinExpr()
                le2 = LinExpr()
                for  k  in range(len(ls)):
                    if k != j:
                        if ls[k][1] == 1:
                            le.addTerms(1, ls[k][0])
                        elif ls[k][1] == 0:
                            le.addConstant(1)
                            le.addTerms(-1,ls[k][0])
                    else:
                        if ls[k][1] == 1:
                            le2.addTerms(1, ls[k][0])
                        elif ls[k][1] == 0:
                            le2.addConstant(1)
                            le2.addTerms(-1,ls[k][0])
                self.gmodel.addConstr(le2 <= le )
    def get_var_values_lp(self, node: None, var_type: str):
        """
        Gets the variables encoding a node and their values.

        Arguments: 
            node:
                The node.
            var_type:
                The type of variables associated with the node to retrieve.
                Either 'out' or 'delta'.
        Returns:
            Pair of tensor of variables and tensor of their values.
        """
        start, end = node.get_milp_var_indices(var_type)
        delta_temp = self.gmodel._vars[start: end]

        if isinstance(node, Relu) and var_type=='delta':
            delta = np.empty(node.output_size, dtype=Var)
            delta[node.get_unstable_flag()] = np.asarray(delta_temp)
            delta = delta.reshape(node.output_shape)

        return delta



    def get_inequalities(self, node, unit, _delta):
        """
        Derives set of inequality nodes. See Anderson et al. Strong
        Mixed-Integer Programming Formulations for Trained Neural Networks

        Arguments:

            node:
                node for deriving ideal cuts.
            unit:
                index of the unit in the node.

        Returns:
            list of indices of nodes of p_layer.
        """

        in_vars, _in = self.get_var_values(node.from_node[0], 'out')
        neighbours = self.prob.nn.calc_neighbouring_units(node.from_node[0], node, unit)
        pos_connected = [i for i in neighbours if node.edge_weight(unit, i) > 0]
        ineqs = []
            
        for p_unit in pos_connected:
            l = self._get_lb(node.from_node[0], node, p_unit, unit)
            u = self._get_ub(node.from_node[0], node, p_unit, unit)
            w = node.edge_weight(unit, p_unit) 
            lhs = w * _in[p_unit]
            rhs = w * (l * (1 - _delta[unit]) + u * _delta[unit])
            if lhs < rhs: 
                ineqs.append(p_unit)

        return ineqs

    def cut_condition(self, ineqs, node, unit, _delta):
        """
        Checks required  inequality condition  on inequality nodes for adding a
        cut.  See Anderson et al. Strong Mixed-Integer Programming Formulations
        for Trained Neural Networks.
        
        Arguments:
            
            ineqs:
                list of inequality units.
            node: 
                The node for deriving ideal cuts.
            unit:
                the index of the unit in node.
            _delta:
                the value of the binary variable associated with the unit.
        
        Returns:

            bool expressing whether or not to add cuts.
        """
        in_vars, _in = self.get_var_values(node.from_node[0], 'out')
        out_vars, _out = self.get_var_values(node.to_node[0], 'out')
        s1 = 0
        s2 = 0

        for p_unit in self.prob.nn.calc_neighbouring_units(node.from_node[0], node,  unit):
            l = self._get_lb(node.from_node[0], node, p_unit, unit)
            u = self._get_ub(node.from_node[0], node, p_unit, unit)
            if p_unit in ineqs:
                s1 += node.edge_weight(unit, p_unit) * (_in[p_unit] - l * (1 - _delta[unit]))
            else:
                s2 += node.edge_weight(unit, p_unit) * u * _delta[unit]
           
            if node.has_bias() is True:
                p = node.get_bias(unit) * _delta[unit]
            else:
                p = 0
        
        return bool(_out[unit] > p + s1 + s2)


    def build_constraint(self, ineqs, node, unit, delta):
        """
        Builds the linear cut. See Anderson et al. Strong Mixed-Integer
        Programming Formulations for Trained Neural Networks.

        Arguments:
            
            ineqs:
                list of inequality nodes.
            node: 
                Node for deriving ideal cuts.
            unit: 
                The index of the unit of node.
            _delta:
                The binary variable associated with the unit.
        
        Returns:

            a pair of Grurobi linear expression for lhs and the rhs of the
            linear cut.
        """
        in_vars, _ = self.get_var_values(node.from_node[0], 'out')
        out_vars, _ = self.get_var_values(node.to_node[0], 'out')

        le = LinExpr()
        s = 0
        for p_unit in self.prob.nn.calc_neighbouring_units(node.from_node[0], node, unit):
            l = self._get_lb(node.from_node[0], node, p_unit, unit)
            u = self._get_ub(node.from_node[0], node, p_unit, unit)
            if p_unit in ineqs:
                le.addTerms(node.edge_weight(unit, p_unit), in_vars[p_unit])
                le.addConstant(- l * node.edge_weight(unit, p_unit))
                le.addTerms(l * node.edge_weight(unit, p_unit), delta[unit])
            else:
                s += node.edge_weight(unit, p_unit) * u
       
        if node.has_bias() is True:
            le.addTerms(s + node.get_bias(unit), delta[unit])
        else:
            le.addTerms(s, delta[unit])

        return out_vars[unit], le


    def _get_lb(self, p_n, n, p_idx, idx):
        """
        Helper function. Given two connected nodes, it returns the upper bound
        of the pointing node if the weight of the connecting edge negative;
        otherwise it returns the lower bound. 

        Arguments:

            p_n, n:
                two consequtive nodes.
            p_idx, idx: 
                indices of units in p_n and n.

        Returns:
                
            float of the lower or upper bound of p_idx.
        """
        
        if n.edge_weight(idx, p_idx) < 0:
            return p_n.bounds.upper[p_idx]

        else:
            return p_n.bounds.lower[p_idx]

    def _get_ub(self, p_n, n, p_idx, idx):
        """
        Helper function. Given two connected nodes, it returns the lower bound
        of the pointing node if the weight of the connecting edge negative;
        otherwise it returns the upper bound. 

        Arguments:

            p_n, n:
                two consequtive nodes.
            p_idx, idx: 
                indices of units in p_n and n.

        Returns:
                
            float of the lower or upper bound of p_n
        """
        if n.edge_weight(idx, p_idx) < 0:
            return p_n.bounds.lower[p_idx] 

        else:
            return p_n.bounds.upper[p_idx]
