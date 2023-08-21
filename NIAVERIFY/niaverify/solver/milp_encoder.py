#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : milp_encoder.py
# @IDE     : pycharm
import torch
import numpy as np
from gurobipy import *
import gurobipy as gp
from gurobipy import GRB
from niaverify.network.node import *
from niaverify.common.utils import ReluState
from niaverify.verification.verification_problem import VerificationProblem
from niaverify.common.configuration import Config
from niaverify.common.logger import get_logger
from timeit import default_timer as timer

class MILPEncoder:

    logger = None
    
    def __init__(self, prob: VerificationProblem, config: Config):
        """
        Arguments:
            nn:
                NeuralNetwork. 
            spec:
                Specification.
            config:
                Configuration
        """

        self.prob = prob
        self.config = config
        if MILPEncoder.logger is None:
            MILPEncoder.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def encode(self, linear_approx=False):
        """
        Builds a Gurobi Model encoding the  verification problem.
   
        Arguments:
            linear_approx: 
                whether to use linear approximation for Relu nodes.
        Returns:
            Gurobi Model.
        """
        start = timer()

        self.test = True

        with gurobipy.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()

            gmodel = Model(env=env)

            self.add_node_vars(gmodel, linear_approx)

            gmodel.update()

            self.add_node_constrs(gmodel, linear_approx)
            self.add_output_constrs(self.prob.nn.tail, gmodel)

            gmodel.update()

            MILPEncoder.logger.info(
                'Encoded verification problem {} into {}, time: {:.2f}'.format(
                    self.prob.id, 
                    "LP" if linear_approx is True else "MILP",
                    timer() - start
                )
            )

            return gmodel


    def encode2(self, linear_approx=False):
        """
        Builds a Gurobi Model encoding the  verification problem.
   
        Arguments:
            linear_approx: 
                whether to use linear approximation for Relu nodes.
        Returns:
            Gurobi Model.
        """
        start = timer()

        self.test = True

        with gurobipy.Env(empty=True) as env:
            env.setParam('OutputFlag', 0)
            env.setParam('LogToConsole', 0)
            env.start()

            gmodel = Model(env=env)

            self.add_node_vars2(gmodel, linear_approx)

            gmodel.update()

            self.add_node_constrs2(gmodel, linear_approx)

            gmodel.update()

            MILPEncoder.logger.info(
                'Encoded verification problem {} into {}, time: {:.2f}'.format(
                    self.prob.id, 
                    "LP" if linear_approx is True else "MILP",
                    timer() - start
                )
            )

            return gmodel
    def add_node_vars(self, gmodel: Model, linear_approx: bool=False):
        """
        Assigns MILP variables for encoding each of the outputs of a given
        node.

        Arguments:
            gmodel:
                The gurobi model
            linear_approx: 
                whether to use linear approximation for Relu nodes.
        """
        self.add_output_vars(self.prob.spec.input_node, gmodel)

        for i in range(self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if j.has_relu_activation() is True:
                    continue

                elif isinstance(j, Relu):
                    self.add_output_vars(j, gmodel)
                    if linear_approx is not True:
                        self.add_relu_delta_vars(j, gmodel)

                elif type(j) in [Flatten, Slice, Unsqueeze, Reshape]:
                    j.out_vars = j.forward(j.from_node[0].out_vars)

                elif isinstance(j, Concat):
                    j.out_vars = j.forward([k.out_vars for k in j.from_node])

                elif type(j) in [Gemm, MatMul, Conv, ConvTranspose, Sub, BatchNormalization, MaxPool]:
                    self.add_output_vars(j, gmodel)

                else:
                    raise TypeError(f'The MILP encoding of node {j} is not supported')
 
    def add_output_vars(self, node: Node, gmodel: Model):
        """
        Creates a real-valued MILP variable for each of the outputs of a given
        node.
   
        Arguments: 
            node:
                The node.
            gmodel:
                The gurobi model
        """ 
        if node.bounds.size() > 0:
            node.out_vars = np.array(
                gmodel.addVars(
                    node.output_size.item(),
                    lb=node.bounds.lower.flatten(),
                    ub=node.bounds.upper.flatten()
                ).values()
            ).reshape(node.output_shape)
        else:
            if isinstance(node, Relu):
                node.out_vars = np.array(
                    gmodel.addVars(
                        (node.output_size,), lb=0, ub=GRB.INFINITY
                    ).values()
                ).reshape(node.output_shape)
            else:
                node.out_vars = np.array(
                    gmodel.addVars(
                        (node.output_size,), lb=-GRB.INFINITY, ub=GRB.INFINITY
                    ).values()
                ).reshape(node.output_shape)
 
    def add_relu_delta_vars(self, node: Relu, gmodel: Model):
        """
        Creates a binary MILP variable for encoding each of the units in a given
        ReLU node. The variables are prioritised for branching according to the
        depth of the node.
   
        Arguments: 
        
            node:
                The Relu node. 
            gmodel:
                The gurobi model
        """
        assert(isinstance(node, Relu)), "Cannot add delta variables to non-relu nodes."
   
        node.delta_vars = np.empty(shape=node.output_size, dtype=Var)
        if node.get_unstable_count() > 0:
            node.delta_vars[node.get_unstable_flag().flatten()] = np.array(
                gmodel.addVars(
                    node.get_unstable_count().item(), vtype=GRB.BINARY
                ).values()
            )
        node.delta_vars = node.delta_vars.reshape(node.output_shape)

    def add_node_constrs(self, gmodel, linear_approx: bool=False):
        """
        Computes the output constraints of a node given the MILP variables of its
        inputs. It assumes that variables have already been added.

        Arguments:

            gmodel:
                The gurobi model.
            linear_approx: 
                whether to use linear approximation for Relu nodes.
        """
        for i in range(self.prob.nn.tail.depth + 1):
            nodes = self.prob.nn.get_node_by_depth(i)
            for j in nodes:
                if j.has_relu_activation() is True:
                    continue
                
                elif isinstance(j, Relu):
                    self.add_relu_constrs(j, gmodel, linear_approx)

                elif type(j) in [Flatten, Concat, Slice, Unsqueeze, Reshape]:
                    pass

                elif type(j) in [Gemm, Conv, ConvTranspose, MatMul, Sub, Add, BatchNormalization]:
                    self.add_affine_constrs(j, gmodel)

                elif isinstance(j, MaxPool):
                    self.add_maxpool_constrs(j, gmodel)

                else:
                    raise TypeError(f'The MILP encoding of node {j} is not supported')

    
    def add_affine_constrs(self, node: Gemm, gmodel: Model):
        """
        Computes the output constraints of an affine node given the MILP
        variables of its inputs. It assumes that variables have already been
        added.
    
        Arguments:
            node: 
                The node. 
            gmodel:
                Gurobi model.
        """
        if type(node) not in [Gemm, Conv, ConvTranspose, MatMul, Sub, Add, BatchNormalization]:
            raise TypeError(f"Cannot compute sub onstraints for {type(node)} nodes.")

        if type(node) in ['Sub', 'Add'] and node.const is not None:
            output = node.forward(
                node.from_node[0].out_vars, node.from_node[1].out_vars
            )
 
        else:
            output = node.forward(node.from_node[0].out_vars)

        for i in node.get_outputs():
            gmodel.addConstr(node.out_vars[i] == output[i])
 

    def add_relu_constrs(self, node: Relu, gmodel: Model, linear_approx=False):
        """
        Computes the output constraints of a relu node given the MILP variables
        of its inputs.

        Arguments:  
            node: 
                Relu node. 
            gmodel:
                Gurobi model.
        """
        assert(isinstance(node, Relu)), "Cannot compute relu constraints for non-relu nodes."
         
        inp = node.from_node[0].forward(node.from_node[0].from_node[0].out_vars)
        out, delta = node.out_vars, node.delta_vars
        l, u = node.from_node[0].bounds.lower, node.from_node[0].bounds.upper

        for i in node.get_outputs():
            if l[i] >= 0 or node.state[i] == ReluState.ACTIVE:
                # active node as per bounds or as per branching
                gmodel.addConstr(out[i] == inp[i])

            elif u[i] <= 0:
                # inactive node as per bounds
                gmodel.addConstr(out[i] == 0)

            elif node.dep_root[i] == False and node.state[i] == ReluState.INACTIVE:
                # non-root inactive node as per branching
                gmodel.addConstr(out[i] == 0)

            elif node.dep_root[i] == True and node.state[i] == ReluState.INACTIVE:
                # root inactive node as per branching
                gmodel.addConstr(out[i] == 0)
                gmodel.addConstr(inp[i] <= 0)

            else:
                l_i, u_i = l[i].item(), u[i].item()
                # unstable node
                if linear_approx is True:
                    gmodel.addConstr(out[i] >= inp[i])
                    gmodel.addConstr(out[i] >= 0)
                    gmodel.addConstr(out[i] <= (u_i / (u_i - l_i)) * (inp[i] - l_i))
                else:
                    gmodel.addConstr(out[i] >= inp[i])
                    gmodel.addConstr(out[i] <= inp[i] - l_i * (1 - delta[i]))
                    gmodel.addConstr(out[i] <= u_i * delta[i])


    def add_maxpool_constrs(self, node: MaxPool, gmodel: Model):
        """
        Computes the output constraints of a maxpool node given the MILP variables
        of its inputs.

        Arguments:  
            node: 
                MaxPool node. 
            gmodel:
                Gurobi model.
        """
        assert(isinstance(node, MaxPool)), "Cannot compute maxpool constraints for non-maxpool nodes."
  
        inp = node.from_node[0].out_vars
        padded_inp = Conv.pad(inp, node.pads).reshape((node.in_ch(), 1) + inp.shape[-2:])
        im2col = Conv.im2col(
            padded_inp, node.kernel_shape, node.strides
        )

        idxs = np.arange(node.output_size).reshape(
            node.output_shape_no_batch()
        ).transpose(1, 2, 0).reshape(-1, node.in_ch())

        for i in itertools.product(*[range(j) for j in idxs.shape]):
            gmodel.addConstr(
                np.take(node.out_vars, idxs[i]) == max_(im2col[:, i[0], i[1]].tolist())
            )


    def add_output_constrs(self, node: Node, gmodel: Model):
        """
        Creates MILP constraints for the output of the output layer.
   
        Arguments:
            
            node:
                The output node.
            gmodel:
                The gurobi model.
        """
        constrs = self.prob.spec.get_output_constrs(gmodel, node.out_vars.flatten())
        for i in constrs:
            gmodel.addConstr(i)

    def add_dep_constrs(self, gmodel):
        """
        Adds dependency constraints.

        Arguments:

            gmodel:
                The gurobi model.
        """
        dg = DependencyGraph(
            self.prob.nn,
            self.config.SOLVER.INTRA_DEP_CONSTRS,
            self.config.SOLVER.INTER_DEP_CONSTRS,
            self.config
        )
        dg.build()

        for i in dg.nodes:
            for j in dg.nodes[i].adjacent:
                # get the nodes in the dependency
                lhs_node, lhs_idx = dg.nodes[i].nodeid, dg.nodes[i].index
                delta1 = self.prob.nn.node[lhs_node].delta_vars[lhs_idx]
                rhs_node, rhs_idx = dg.nodes[j].nodeid, dg.nodes[j].index
                delta2 = self.prob.nn.node[rhs_node].delta_vars[rhs_idx]
                dep = dg.nodes[i].adjacent[j]

                # add the constraint as per the type of the dependency
                if dep == DependencyType.INACTIVE_INACTIVE:
                    gmodel.addConstr(delta2 <= delta1)

                elif dep == DependencyType.INACTIVE_ACTIVE:
                    gmodel.addConstr(1 - delta2 <= delta1)

                elif dep == DependencyType.ACTIVE_INACTIVE:
                    gmodel.addConstr(delta2 <= 1 - delta1)

                elif dep == DependencyType.ACTIVE_ACTIVE:
                    gmodel.addConstr(1 - delta2 <= 1 - delta1)


    def add_node_vars2(self, gmodel: Model, linear_approx: bool=False):
        """
        Assigns MILP variables for encoding each of the outputs of a given
        node.

        Arguments:
            gmodel:
                The gurobi model
            linear_approx: 
                whether to use linear approximation for Relu nodes.
        """
        self.add_output_vars(self.prob.spec.input_node, gmodel)


    def add_node_constrs2(self, gmodel, linear_approx: bool=False):
        """
        Computes the output constraints of a node given the MILP variables of its
        inputs. It assumes that variables have already been added.

        Arguments:

            gmodel:
                The gurobi model.
            linear_approx: 
                whether to use linear approximation for Relu nodes.
        """

        inp_vars = self.prob.spec.input_node.out_vars[0]

        output_eq = np.zeros(self.prob.nn.tail.output_size)
        #output_eq = torch.FloatTensor(self._objective.get_summed_constraints())
        const = 0
        if hasattr(self.prob.spec.output_formula, 'clauses'):
            for idx  in range(len(self.prob.spec.output_formula.clauses)):
                constr_eq =self.prob.spec.output_formula.clauses[idx]
                op1 = constr_eq.op1.i
                op2 = constr_eq.op2.i
                output_eq[op1] += -1
                output_eq[op2] += 1
        else:
            op1 = self.prob.spec.output_formula.op1.i
            output_eq[op1] += 1
            const = -1 * self.prob.spec.output_formula.op2
        #output_eq = self.prob.spec.output_formula.clauses
        output_eq = torch.FloatTensor(output_eq)
        eq = self.prob.convert_output_bounding_equation(output_eq.view(1, -1), lower=False)
        eq_matrix= np.array(eq.matrix[0])
        eq_const = np.array(eq.const[0])

        expr = gp.LinExpr()
        constant  = const + eq_const

        for j in range(len(eq_matrix)):
            expr += eq_matrix[j] * inp_vars[j]
        gmodel.addLConstr(expr , GRB.GREATER_EQUAL,-1 * constant)
