#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : milp_soler.py
# @IDE     : pycharm
from gurobipy import *
from niaverify.solver.milp_encoder import MILPEncoder
from niaverify.solver.ideal_formulation import IdealFormulation
from niaverify.solver.solve_result import SolveResult
from niaverify.solver.solve_report import SolveReport
from niaverify.split.split_strategy import SplitStrategy
from niaverify.common.logger import get_logger
from timeit import default_timer as timer
import numpy as np
# A
from niaverify.solver.lp_cuts import LPCuts
class MILPSolver:

    logger = None
    num_add_cut = 0
    def __init__(self, prob, config, lp=False):
        """
        Arguments:

            prob:
                VerificationProblem.            
            config:
                Configuration. 
            lp:
                Whether to use linear relaxation.

        """

        MILPSolver.prob = prob
        MILPSolver.config = config
        MILPSolver.status = SolveResult.UNDECIDED
        MILPSolver.lp = lp
        if MILPSolver.logger is None:
            MILPSolver.logger = get_logger(__name__, config.LOGGER.LOGFILE)

    def solve(self):
        """
        Builds and solves the MILP program of the verification problem.

        Returns:
            SolveReport
        """
        start = timer()

        # need to convert network to numpy for the milp encoding 
        MILPSolver.prob.cpu()

        # encode into milp
        me = MILPEncoder(MILPSolver.prob, MILPSolver.config)

        if MILPSolver.lp == True:
            gmodel = me.encode(linear_approx=True)
            #gmodel = me.encode2(linear_approx=True)
        else:
            gmodel = me.encode()

        # Set gurobi parameters
        gmodel.setParam('OUTPUT_FLAG', 1 if MILPSolver.config.SOLVER.PRINT_GUROBI_OUTPUT  == True else 0)
        if MILPSolver.config.SOLVER.TIME_LIMIT != -1: 
            gmodel.setParam('TIME_LIMIT', MILPSolver.config.SOLVER.TIME_LIMIT)
        if not MILPSolver.config.SOLVER.DEFAULT_CUTS: 
            MILPSolver.disable_default_cuts(gmodel)
        # gmodel.setParam('FeasibilityTol', MILPSolver.config.SOLVER.FEASIBILITY_TOL)
        gmodel._vars = gmodel.getVars()
        # set callback cuts 
        MILPSolver.id_form = IdealFormulation(
            MILPSolver.prob,
            gmodel,
            MILPSolver.config
        )
        # ADD
        if MILPSolver.lp is not True and len(self.prob.dep_by_lp_ls) > 0 :
            MILPSolver.lp_cuts = LPCuts(
                MILPSolver.prob,
                gmodel,
                MILPSolver.config
            )
            MILPSolver.lp_cuts.build_cuts()
        # Optimise
        if MILPSolver.config.SOLVER.callback_enabled() and MILPSolver.lp is not True:
            gmodel.optimize(MILPSolver._callback)
            #gmodel.optimize()
        else:
            gmodel.optimize()

        runtime, cex =  timer() - start, None
        if MILPSolver.status == SolveResult.BRANCH_THRESHOLD:
            result = SolveResult.BRANCH_THRESHOLD
        elif gmodel.status == GRB.OPTIMAL:
            cex_shape = MILPSolver.prob.spec.input_node.input_shape
            cex = np.zeros(cex_shape)
            for i in itertools.product(*[range(j) for j in cex_shape]):
                cex[i] = MILPSolver.prob.spec.input_node.out_vars[i].x
            result = SolveResult.UNSAFE
        elif gmodel.status == GRB.TIME_LIMIT:
            result = SolveResult.TIMEOUT
        elif gmodel.status == GRB.INTERRUPTED:
            result = SolveResult.INTERRUPTED
        elif gmodel.status == GRB.INFEASIBLE or gmodel.status == GRB.INF_OR_UNBD:
            result = SolveResult.SAFE
        else:
            result = SolveResult.UNDECIDED
        
        MILPSolver.logger.info(
            'Verification problem {} solved, '
            'LP: {}, '
            'time: {:.2f}, '
            'result: {}.'
            .format(
                MILPSolver.prob.id,
                MILPSolver.lp,
                runtime,
                result.value))

        return SolveReport(result, runtime, cex)



    @staticmethod
    def _callback(model, where):
        """
        Gurobi callback function.
        """
        if where == GRB.Callback.MIPNODE:
            if model.cbGet(GRB.Callback.MIPNODE_STATUS) == GRB.Status.OPTIMAL:
                if MILPSolver.config.SOLVER.IDEAL_CUTS == True:
                    MILPSolver.id_form.add_cuts()
                #if MILPSolver.config.SOLVER.dep_cuts_enabled():
                    #print("addcutt_________________________")
                    #MILPSolver.num_add_cut += 1
                   # print(MILPSolver.num_add_cut)
                    #MILPSolver.dep_cuts.add_cuts()
        elif MILPSolver.config.SOLVER.MONITOR_SPLIT == True and \
        MILPSolver.config.SPLITTER.SPLIT_STRATEGY != SplitStrategy.NONE and \
        where == GRB.Callback.MIP:
            MILPSolver.monitor_milp_nodes(model)
        # elif where == GRB.Callback.MIP:
        #      MILPSolver.monitor_milp_nodes(model)


    @staticmethod
    def monitor_milp_nodes(model): 
        """
        Monitors the number of MILP nodes solved. Terminates the MILP if the
        number exceeds the BRANCH_THRESHOLD.

        Arguments:

            model: Gurobi model.
        """
        nodecnt = model.cbGet(GRB.Callback.MIP_NODCNT)
        # print(nodecnt)
        # with open(MILPSolver.config.LOGGER.SUMFILE, 'a') as f:
        #     f.write(str(nodecnt)+'\n')
        if nodecnt > MILPSolver.config.SOLVER.BRANCH_THRESHOLD:
            MILPSolver.status = SolveResult.BRANCH_THRESHOLD
            model.terminate()


    @staticmethod
    def disable_default_cuts(gmodel):
        """
        Disables Gurobi default cuts.

        Arguments:

            gmodel: Gurobi Model.

        Returns:

            None.
        """
        gmodel.setParam('PreCrush', 1)
        gmodel.setParam(GRB.Param.CoverCuts,0)
        gmodel.setParam(GRB.Param.CliqueCuts,0)
        gmodel.setParam(GRB.Param.FlowCoverCuts,0)
        gmodel.setParam(GRB.Param.FlowPathCuts,0)
        gmodel.setParam(GRB.Param.GUBCoverCuts,0)
        gmodel.setParam(GRB.Param.ImpliedCuts,0)
        gmodel.setParam(GRB.Param.InfProofCuts,0)
        gmodel.setParam(GRB.Param.MIPSepCuts,0)
        gmodel.setParam(GRB.Param.MIRCuts,0)
        gmodel.setParam(GRB.Param.ModKCuts,0)
        gmodel.setParam(GRB.Param.NetworkCuts,0)
        gmodel.setParam(GRB.Param.ProjImpliedCuts,0)
        gmodel.setParam(GRB.Param.StrongCGCuts,0)
        gmodel.setParam(GRB.Param.SubMIPCuts,0)
        gmodel.setParam(GRB.Param.ZeroHalfCuts,0)
        gmodel.setParam(GRB.Param.GomoryPasses,0)


