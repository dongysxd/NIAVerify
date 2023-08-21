#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : verification_problem.py
# @IDE     : pycharm

import torch

from niaverify.bounds.sip import SIP
from niaverify.network.node import Relu,Gemm
from niaverify.bounds.equation import Equation

class VerificationProblem(object):

    prob_count = 0

    def __init__(self, nn, spec, depth, config):
        """
        Arguments:
            nn:
                NeuralNetwork.
            spec:
                Specification.
            depth:
                Depth of the problem in the branch-and-bound tree.
            config:
                Configuration.
        """
        VerificationProblem.prob_count += 1
        self.id = VerificationProblem.prob_count
        self.nn = nn
        self.spec = spec
        # couple neural network with spec
        self.nn.head.from_node.insert(0, spec.input_node)
        self.depth = depth
        self.config = config
        self.stability_ratio = -1
        self.output_range = 0
        self.bounds_ver_done = False
        self.inc_ver_done = False
        self.pgd_ver_done = False
        #A
        self.preattack_ver_done = False
        self.stable_ratio_change_flag = False
        self.final_queue_flag = False
        self.dep_by_lp = []
        self.dep_by_lp_ls = []
        self.lp_ver_done = False
        self._sip_bounds_computed = False
        self.device = torch.device('cpu')


    def bound_analysis(self, delta_flags=None):
        """
        Computes bounds the network.

        Arguments:
            delta_flags:
                list of current values of Gurobi binary variables; required
                when the bounds are computed at runtime
        """
        sip = self.set_bounds(delta_flags)
        if sip is not None:
            self.stability_ratio = self.nn.get_stability_ratio()
            self.output_range = self.nn.get_output_range()
            if self.config.SIP.SIMPLIFY_FORMULA is True:
                self.spec.output_formula = sip.simplify_formula(self.spec.output_formula)
            return True

        else:
            return False

    def set_bounds(self, delta_flags=None):
        """
        Computes bounds the network.

        Arguments:
            delta_flags:
                list of current values of Gurobi binary variables; required
                when the bounds are computed at runtime
        """
        # check if bounds are already computed
        if delta_flags is None:
            if self._sip_bounds_computed:
                return None

        # compute bounds
        sip = SIP(self, self.config, delta_flags)
        sip.set_bounds()
        # flag the computation
        if delta_flags is None:
            self._sip_bounds_computed = True

        return sip

    def score(self, initial_fixed_ratio):
        return (self.stability_ratio - initial_fixed_ratio) 

    def worth_split(self, subprobs, initial_fixed_ratio):
        pscore0 = self.score(initial_fixed_ratio)
        pscore1 = subprobs[0].score(initial_fixed_ratio)
        pscore2 = subprobs[1].score(initial_fixed_ratio)

        _max = max(pscore1, pscore2)
        _min = min(pscore1, pscore2) 

        if pscore0 >= _max:
            return False
        elif _min > pscore0:
            return True
        elif  (pscore1 + pscore2)/2 > pscore0:
            return True
        else:
            return False
 
    def check_bound_tightness(self, subprobs):
        out0 = self.nn.layers[-1]
        for sp in subprobs:
            if sp.output_range > self.output_range:
                return False
        return True

        for sp in subprobs:
            out1 = sp.nn.layers[-1]
            for i in out0.get_outputs():
                b0 = out0.post_bounds.lower[i]
                b1 = out1.post_bounds.lower[i]
                if b1 < b0:
                    return False
                b0 = out0.post_bounds.upper[i]
                b1 = out1.post_bounds.upper[i]
                if b1 > b0:
                    return False
        return True

    def lp_analysis(self):
        ver_report = VerificationReport()
        if self.spec.output_formula is None:
            return True
        self.bound_analysis()
        lower_bounds = self.nn.layers[-1].post_bounds.lower
        upper_bounds = self.nn.layers[-1].post_bounds.upper
        if self.satisfies_spec(self.spec.output_formula, lower_bounds, upper_bounds):  
            ver_report.result = SolveResult.SAFE

    def satisfies_spec(self):
        if self.spec.output_formula is None:
            return True
        if not self._sip_bounds_computed:
            raise Exception('Bounds not computed')
        
        return self.spec.is_satisfied(
            self.nn.tail.bounds.lower,
            self.nn.tail.bounds.upper
        )

    def get_var_indices(self, nodeid, var_type):
        """
        Returns the indices of the MILP variables associated with a given
        layer.

        Arguments:
                
            nodeid:
                the id of the node for which to retrieve the indices of the
                MILP variables.
            var_type:
                either 'out' for the output variables or 'delta' for the binary
                variables.

        Returns:
        
            pair of ints indicating the start and end positions of the indices
        """
        assert nodeid in self.nn.node or nodeid == self.spec.input_npde.id, \
            f"Node id {nodeid} not  recognised."

        if  nodeid == self.spec.input_node.id:
            return 0, self.spec.input_node.out_vars.size

        start, end = self.spec.input_node.out_vars.size, 0

        for i in range(self.nn.tail.depth + 1):
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                if j.id == nodeid:

                    if var_type == 'out':
                        end = start + j.out_vars.size

                    elif var_type == 'delta':
                        start += j.out_vars.size
                        end = start + j.get_unstable_count()

                    else:
                        raise ValueError(f'Var type {var_type} is not recognised')

                    return start, end

                else:
                    start += j.get_milp_var_size()

    def detach(self):
        """
        Detaches and clones the bound tensors.
        """
        self.nn.detach()
        self.spec.detach()

    def clean_vars(self):
        """
        Nulls out all MILP variables associate with the verification problem.
        """
        self.nn.clean_vars()
        self.spec.clean_vars()
        
    def cuda(self):
        """
        Moves all data to gpu memory
        """
        if self.device == torch.device('cpu'):
            self.nn.cuda()
            self.spec.cuda()
            self.device = torch.device('cuda')

    def cpu(self):
        """
        Moves all data to cpu memory
        """
        if self.device == torch.device('cuda'):
            self.nn.cpu()
            self.spec.cpu()
            self.device = torch.device('cpu')

    def to_string(self):
        return self.nn.model_path  + ' against ' + self.spec.to_string()

    #A
    def get_most_impactfull_neurons(self, output_equation: torch.Tensor = None, lower: bool = True, include_inp_node:bool = True) -> tuple:

        """
        Returns a sorted list over the neurons heuristically determined to have the
        most impact on the weighted output.

        Args:
            output_equation:
                The coefficients of the output equation. The tensor should be of length
                NxM where N is the number of equations and M is the number of outputs
                in the network.
            lower:
                If true, the returned equation is lower-bounding, otherwise upper
                bounding.
        Returns:
            A tuple (impact, indices) where impact contains the sorted list of
            estimated impacts per neuron and indices contains the corresponding
            [node_num, neuron_num] pairs.
        """

        self.set_relaxations()
        self.update_intermediate_bounds()
        self._calc_non_linear_node_direct_impact(output_equation, lower=lower)
        self._calc_non_linear_node_indirect_impact()

        impacts = []
        indices = []

        for i in range(self.nn.tail.depth + 1):
            nodes = self.nn.get_node_by_depth(i)
            for node in nodes:
                if isinstance(node, Relu) is True:
                    impact = node.impact
                    neuron_indices = node.non_lin_indices.unsqueeze(1)
                    node_nums = torch.zeros(len(neuron_indices), dtype=torch.long).unsqueeze(1) + node.depth
                    indices.append(torch.cat((node_nums, neuron_indices), dim=1))
                    impacts.append(impact)

        if self.config.SPLITTER.INPUT_NODE_SPLIT is True and include_inp_node is True:

            self._calc_input_node_indirect_impact()
            indices.append(torch.cat((torch.ones(self.nn.head.input_size, dtype=torch.long).unsqueeze(1),
                                        torch.LongTensor(list(range(self.nn.head.input_size))).unsqueeze(1)), dim=1))
            impacts.append(self.nn.head.impact)

        impacts = torch.cat(impacts, dim=0)
        indices = torch.cat(indices, dim=0)

        sorted_idx = torch.argsort(impacts, descending=True)

        return impacts[sorted_idx], indices[sorted_idx]
    
    def _calc_input_node_indirect_impact(self):

        """
        Estimates the indirect impact of the input nodes on the output equation

        It is assumed that the direct impact is already calculated for all nodes. The
        indirect impact is stored in node.intermediate_bounds["impact"].
        """

        node  = self.nn.head

        #bounds_width_input = (node.bounds_concrete_pre[0][:, 1] - node.bounds_concrete_pre[0][:, 0]).cpu()
        bounds_width_input = (node.bounds.upper - node.bounds.lower).cpu()

        #node.impact = torch.zeros(node.in_size, dtype=self._precision)
        node.impact = torch.zeros(node.input_size, dtype=self.config.PRECISION)
        for i in range(2,self.nn.tail.depth + 1):
            nodes = self.nn.get_node_by_depth(i)
            for other_node in nodes:
                 if isinstance(other_node, Relu) is True:

                    idx = other_node.non_lin_indices
                    if len(idx) == 0:
                        continue

                    bounds_width = (other_node.from_node[0].bounds.upper[idx] - other_node.from_node[0].bounds.lower[idx]).cpu()
                    # bounds_width = (other_node.bounds_concrete_pre[0][idx, 1] -
                    #                 other_node.bounds_concrete_pre[0][idx, 0]).cpu()
                    impact = bounds_width_input.view(1, -1)/2 * (abs(other_node.intermediate_bounds[1]["lower"][:, :-1]) +
                                                                abs(other_node.intermediate_bounds[1]["upper"][:, :-1]))
                    relative_impact = impact/bounds_width.view(-1, 1)
                    indirect_impact = torch.sum(other_node.impact.view(-1, 1) * relative_impact,
                                                dim=0)
                    node.impact += indirect_impact * 0.75


    def _calc_non_linear_node_direct_impact(self, output_equation: torch.Tensor, lower: bool = True):

        """
        Estimates the impact of the nodes non-linear nodes on the output equation

        The direct impact is stored in node.intermediate_bounds["impact"]

        Args:
            output_equation:
                The coefficients of the output equation. The tensor should be of length
                NxM where N is the number of equations and M is the number of outputs
                in the network.
            lower:
                If true, the returned equation is lower-bounding, otherwise upper
                bounding.
        """

        xx =  self.convert_output_bounding_equation(output_equation.view(1, -1), lower=lower)


        #output_node = self.nodes[-1]
        output_node = self.nn.tail
        for i in range(1,self.nn.tail.depth + 1):
            node = self.nn.get_node_by_depth(i)[0]
            if isinstance(node, Relu) is True:
                idx = torch.flatten(torch.nonzero(node.unstable_flag))
                if lower:
                    symb_bounds_in_neg = output_node.intermediate_bounds[node.depth]['lower'][:, idx].clone()
                    symb_bounds_in_neg[symb_bounds_in_neg > 0] = 0
                    biases = symb_bounds_in_neg * node.relaxations[1, idx, 1].cpu().view(1, -1)
                else:
                    symb_bounds_in_pos = output_node.intermediate_bounds[node.depth]['upper'][:, idx].clone()
                    symb_bounds_in_pos[symb_bounds_in_pos < 0] = 0
                    biases = symb_bounds_in_pos * node.relaxations[1, idx, 1].cpu().view(1, -1)

                node.impact = torch.sum(biases, dim=0) + 1e-6


    def _calc_non_linear_node_indirect_impact(self):

        """
        Estimates the indirect impact of the non-linear nodes on the output equation

        It is assumed that the direct impact is already calculated for all nodes. The
        indirect impact is added to node.intermediate_bounds["impact"].
        """

        for i in range(self.nn.tail.depth,0, -1):
            node = self.nn.get_node_by_depth(i)[0]
            if isinstance(node, Relu) is True:
                this_idx =  torch.flatten(torch.nonzero(node.unstable_flag))
                if this_idx.shape[0] == 0:
                    continue
                relaxations = node.relaxations.cpu()
                for j in range(node.depth+1,self.nn.tail.depth+1):
                    succ_node = self.nn.get_node_by_depth(j)[0]
                    if isinstance(succ_node, Relu) is True:
                        succ_idx =torch.flatten(torch.nonzero(succ_node.unstable_flag))
                        if succ_idx.shape[0] == 0:
                            continue

                        succ_intermediate_bounds = succ_node.intermediate_bounds[i]
                        #boundsx = torch.cat((succ_node.from_node[0].bounds.lower.unsqueeze(1), succ_node.from_node[0].bounds.upper.unsqueeze(1)), dim=0)
                        succ_bounds = torch.cat((succ_node.from_node[0].bounds.lower.unsqueeze(1), succ_node.from_node[0].bounds.upper.unsqueeze(1)), dim=1)[succ_idx]
                        symb_bounds_in_neg_low = succ_intermediate_bounds['lower'][:, this_idx].clone()
                        symb_bounds_in_neg_low[symb_bounds_in_neg_low > 0] = 0

                        symb_bounds_in_pos_up = succ_intermediate_bounds['upper'][:, this_idx].clone()
                        symb_bounds_in_pos_up[symb_bounds_in_pos_up < 0] = 0

                        biases_low = symb_bounds_in_neg_low * relaxations[1, this_idx, 1].view(1, -1)
                        biases_up = symb_bounds_in_pos_up * relaxations[1, this_idx, 1].view(1, -1)

                        relative_impact = torch.clip(torch.abs(biases_low / succ_bounds[:, 0].view(-1, 1).cpu()), 0, 1)
                        relative_impact += torch.clip(torch.abs(biases_up / succ_bounds[:, 1].view(-1, 1).cpu()), 0, 1)

                        indirect_impact = relative_impact * succ_node.impact.view(-1, 1)
                        node.impact += (indirect_impact.sum(dim=0) * 0.75)


    def convert_output_bounding_equation(self, output_equations: torch.Tensor,
                                         lower: bool = False,
                                         bias_sep_constraints: bool = False):
         #-> Optional[torch.Tensor]:

        """
        Converts an equation wrt. output-variables to a lower/upper bounding equation
        with respect to the input-variables.

        Args:
            output_equations:
                The coefficients of the output equation. The tensor should be of length
                NxM where N is the number of equations and M is the number of outputs
                in the network.
            lower:
                If true, the returned equation is lower-bounding, otherwise upper
                bounding.
            bias_sep_constraints:
                If true, the bias values from relaxation are calculated as separate
                values.
        Returns:
                [a_0, a_1 ... a_n,cb] in the resulting equation sum(a_i*x_i) + c where
                x_i are the networks input variables.

                If bias_sep_constraints is true, the coeffs
                [a_0, a_1 ... a_n, b_0, b_1 ... b_n, c] are calculated instead where
                a_i are the coeffs we get when always propagating through lower
                relaxations, while b_i indicate the effect change in equation
                by propagating through the upper relaxation of the i'th non-linear
                node instead of lower.
        """
        sip = SIP(self, self.config, delta_flags=None)
        const = torch.tensor(
                [0], dtype=self.config.PRECISION, device=self.config.DEVICE
            )
        symb_eq = Equation(output_equations, const, self.config)
        return sip.bs_sip.back_substitution2(
            symb_eq, self.nn.tail, 'upper', i_flag=None, slopes=None, os_sip=None,calc_last_node=True
        )


#A
    def set_relaxations(self):
        for i in range(self.nn.tail.depth + 1):
            nodes = self.nn.get_node_by_depth(i)
            for j in nodes:
                if isinstance(j, Relu) is True:
                    j._set_relaxation_for_node()

    def update_intermediate_bounds(self):
        for i in range(self.nn.tail.depth + 1):
            nodes = self.nn.get_node_by_depth(i)
            for node in nodes:
                if isinstance(node, Relu) is True:
                    non_lin_this =  [i for i, x in enumerate(node.origin_non_lin_indices) if x in node.non_lin_indices]
                    for key1 in node.intermediate_bounds:
                        for key2 in node.intermediate_bounds[key1]:
                            node.intermediate_bounds[key1][key2] = node.intermediate_bounds[key1][key2][non_lin_this]
