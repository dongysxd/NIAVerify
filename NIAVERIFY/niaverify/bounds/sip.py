#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : ideal_formulation.py
# @IDE     : pycharm
import math
import numpy as np
import torch
from timeit import default_timer as timer
from niaverify.network.node import *
from niaverify.bounds.bounds import Bounds
from niaverify.bounds.equation import Equation
from niaverify.bounds.ibp import IBP
from niaverify.bounds.os_sip import OSSIP
from niaverify.bounds.bs_sip import BSSIP
from niaverify.common.logger import get_logger
from niaverify.common.configuration import Config
from niaverify.specification.formula import  *

torch.set_num_threads(1)

class SIP:

    logger = None

    def __init__(self, prob, config: Config, delta_flags=None):
        """
        Arguments:

            prob:
                The verification problem.
            config:
                Configuration.
        """
        self.prob = prob
        self.config = config
        self.delta_flags = delta_flags
        if SIP.logger is None and config.LOGGER.LOGFILE is not None:
            SIP.logger = get_logger(__name__, config.LOGGER.LOGFILE)

        self.ibp = IBP(self.prob, self.config)
        self.os_sip = OSSIP(self.prob, self.config)
        self.bs_sip = BSSIP(self.prob, self.config)
        
    def set_bounds(self):
        """
        Sets the bounds.
        """
        start = timer()

        if self.config.DEVICE == torch.device('cuda'):
            self.prob.cuda()

        # get relaxation slopes
        # if self.prob.nn.has_custom_relaxation_slope() is True:
        #     slopes = self.prob.nn.get_lower_relaxation_slopes(gradient=False)
        # else:
        #     slopes = None
        slopes = None
        # set bounds using area approximations
        self._set_bounds(slopes, depth=1)

        # optimise the relaxation slopes using pgd
        if self.config.SIP.SLOPE_OPTIMISATION is True and \
        self.prob.nn.has_custom_relaxation_slope() is not True:
            bs_sip = BSSIP(self.prob, self.config)
            slopes = bs_sip.optimise(self.prob.nn.tail)
            self.prob.nn.relu_relaxation_slopes = slopes
            starting_depth = self.prob.nn.get_non_linear_starting_depth()
            self._set_bounds(slopes, depth=starting_depth)

        # print(self.prob.nn.tail.bounds.lower)
        # print(self.prob.nn.tail.bounds.upper)
        # print('done')
        # import sys
        # sys.exit()

        if self.logger is not None:
            SIP.logger.info(
                'Bounds computed, time: {:.3f}, range: {:.3f}, '.format(
                    timer()-start,
                    torch.mean(
                        self.prob.nn.tail.bounds.upper - self.prob.nn.tail.bounds.lower
                    )
                )
            )

    def _set_bounds(
        self, slopes: tuple[dict]=None, delta_flags: torch.Tensor=None, depth=1
    ):
        """
        Sets the bounds.
        """
        for i in range(depth, self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                delta = self._get_delta_for_node(j, delta_flags)
                lower_slopes, upper_slopes = self._get_slopes_for_node(j, slopes)
                self._set_bounds_for_node(j, lower_slopes, upper_slopes, delta_flags)
                #print(j.depth, j.output_size, torch.mean(j.bounds.lower))
 
    def _set_bounds_for_node(
        self,
        node: Node,
        lower_slopes: dict=None,
        upper_slopes: dict=None,
        delta_flag: torch.Tensor=None
    ):

        # set interval propagation bounds
        ia_count = self.ibp.set_bounds(node, lower_slopes, upper_slopes, delta_flag)

        # check eligibility for symbolic equations
        symb_elg = self.is_symb_eq_eligible(node)

        # set one step symbolic bounds
        if self.config.SIP.ONE_STEP_SYMBOLIC is True:
            self.os_sip.forward(node, lower_slopes, upper_slopes)
            if symb_elg is True:
                os_count = self.os_sip.set_bounds(node)
 
        # recheck eligibility for symbolic equations
        non_linear_depth = self.prob.nn.get_non_linear_starting_depth()
        symb_elg = self.is_symb_eq_eligible(node) and node.depth >= non_linear_depth
 
        # set back substitution symbolic bounds
        if self.config.SIP.SYMBOLIC is True and symb_elg is True:
            if self.config.SIP.ONE_STEP_SYMBOLIC is True and \
            self.config.SIP.EQ_CONCRETISATION is True and \
            node.depth > 2:
                if os_count * 5 > ia_count:
                    concretisations = self.os_sip
                else:
                    self.os_sip.clear_equations()
                    concretisations = None
            else:
                concretisations = None

            bs = self.bs_sip.set_bounds(
                node, lower_slopes, upper_slopes, concretisations
            )

    def _get_delta_for_node(self, node: Node, delta_flags: torch.Tensor) -> torch.Tensor:
        if delta_flags is not None and node.has_relu_activation() is True:
            return delta_flags[j.to_node[0].id]
            
        return None

    def _get_slopes_for_node(self, node: Node, slopes: tuple[dict]) -> torch.Tensor:
        if slopes is None:
            return None, None
        if node.id not in slopes[0].keys():
            return None, None
        return slopes[0][node.id], slopes[1][node.id]

    def is_symb_eq_eligible(self, node: None) -> bool:
        """
        Determines whether the node implements function requiring a symbolic
        equation for bound calculation.
        """ 
        if node.depth <= 1:
            return False

        if node is self.prob.nn.tail:
            #return True
            return False

        non_eligible = [
            Input, Relu, Reshape, Flatten, Slice
        ]
        if type(node) in non_eligible or node is self.prob.nn.head:
            return False

        if node.has_relu_activation() and node.to_node[0].get_unstable_count() > 0:
            return True
    
        return False


    def runtime_bounds(self, eqlow_r, equp_r, eq, lb_r, ub_r, lb, ub, relu_states):
        """
        Modifies the given bound equations and bounds so that they are in line
        with the MILP relu states of the nodes.

        Arguments:

            eqlow: lower bound equations of a layer's output.

            equp: upper bound equations of a layer's output.

            lbounds: the concrete lower bounds of eqlow.

            ubounds: the concrete upper bounds of equp.

            relu_states: a list of floats, one for each node. 0: inactive,
            1:active, anything else: unstable.

        Returns:

            a four-typle of the lower and upper bound equations and the
            concrete lower and upper bounds resulting from stablising nodes as
            per relu_states.
        """
        if relu_states is None:
            return [eqlow_r, equp_r], [lb_r, ub_r]
        eqlow_r.coeffs[(relu_states==0),:] = 0
        eqlow_r.const[(relu_states==0)] = 0
        equp_r.coeffs[(relu_states==0),:] = 0
        equp_r.const[(relu_states==0)] = 0
        lb_r[(relu_states == 0)] = 0
        ub_r[(relu_states == 0)] = 0
        eqlow_r.coeffs[(relu_states==1),:] = eq.coeffs[(relu_states==1),:]
        eqlow_r.const[(relu_states==1)] = eq.const[(relu_states==1)]
        equp_r.coeffs[(relu_states==1),:] = eq.coeffs[(relu_states==1),:]
        equp_r.const[(relu_states==1)] = eq.const[(relu_states==1)]
        lb_r[(relu_states == 0)] = lb[(relu_states == 0)]
        ub_r[(relu_states == 0)] = ub[(relu_states == 0)]

        return [eqlow_r, equp_r], [lb_r, ub_r]


    def branching_bounds(self, relu_states, bounds):
        """
        Modifies the given  bounds so that they are in line with the relu
        states of the nodes as per the branching procedure.

        Arguments:
            
            relu_states:
                Relu states of a layer
            bounds:
                Concrete bounds of the layer
            
        Returns:
            
            a pair of the lower and upper bounds resulting from stablising
            nodes as per the relu states.
        """
        shape = relu_states.shape
        relu_states = relu_states.reshape(-1)
        lbounds = bounds[0]
        ubounds = bounds[1]
        _max = np.clip(lbounds, 0, math.inf)
        _min = np.clip(ubounds, -math.inf, 0)
        lbounds[(relu_states == ReluState.ACTIVE)] = _max[(relu_states == ReluState.ACTIVE)]
        ubounds[(relu_states == ReluState.INACTIVE)] = _min[(relu_states == ReluState.INACTIVE)]

        return lbounds, ubounds

    def simplify_formula(self, formula):
        if isinstance(formula, VarVarConstraint):
            coeffs = torch.zeros(
                (1, self.prob.nn.tail.output_size),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            const = torch.tensor(
                [0], dtype=self.config.PRECISION, device=self.config.DEVICE
            )
            if formula.sense == Formula.Sense.GT:
                coeffs[0, formula.op1.i], coeffs[0, formula.op2.i] = 1, -1
            elif formula.sense == Formula.Sense.LT:
                coeffs[0, formula.op1.i], coeffs[0, formula.op2.i] = -1, 1
            else:
                raise ValueError('Formula sense {formula.sense} not expected')
            equation = Equation(coeffs, const, self.config)
            diff = self.back_substitution(
                equation, self.prob.nn.tail, 'lower'
            )

            return formula if diff <= 0 else TrueFormula()

        if isinstance(formula, DisjFormula):
            fleft = self.simplify_formula(formula.left)
            fright = self.simplify_formula(formula.right)

            return DisjFormula(fleft, fright)

        if isinstance(formula, ConjFormula):
            fleft = self.simplify_formula(formula.left)
            fright = self.simplify_formula(formula.right)

            return ConjFormula(fleft, fright)

        if isinstance(formula, NAryDisjFormula):
            clauses = [self.simplify_formula(f) for f in formula.clauses]

            return NAryDisjFormula(clauses)

        if isinstance(formula, NAryConjFormula):
            clauses = [self.simplify_formula(f) for f in formula.clauses]

            return NAryConjFormula(clauses)

        return formula 
