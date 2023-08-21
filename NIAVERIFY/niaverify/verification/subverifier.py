#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : subverifier.py
# @IDE     : pycharm

import numpy as np
import itertools
import traceback
import torch
from timeit import default_timer as timer
from queue import Queue
from niaverify.common.logger import get_logger
from niaverify.solver.milp_solver import MILPSolver
from niaverify.solver.solve_report import SolveReport
from niaverify.solver.solve_result import SolveResult
from niaverify.verification.pgd import ProjectedGradientDescent
from niaverify.verification.preattack import PreAttack
from niaverify.verification.verification_problem import VerificationProblem
from niaverify.split.node_splitter import NodeSplitter
class SubVerifier:

    logger = None
    TIMEOUT = 3600
    id_iter = itertools.count()

    #A
    num_pre = 0
    num_pgd = 0
    num_sum = 0
    def __init__(self, config, jobs_queue=None, reporting_queue=None):
        """
        Arguments:
            config:
                Configuration.
            jobs_queue:
                Queue of verification problems to read from.
            reporting_queue:
                Queue of solve reports to enqueue the verification results.
        """
        super(SubVerifier, self).__init__()
        self.id = next(SubVerifier.id_iter)
        self.jobs_queue = jobs_queue
        self.reporting_queue = reporting_queue
        self.config = config
        if SubVerifier.logger is None:
            SubVerifier.logger = get_logger(__name__ + str(self.id), config.LOGGER.LOGFILE)



    def run(self):
        if self.jobs_queue is None:
            raise ValueError('jobs_queue shoud be Queue.')
        if self.reporting_queue is None:
            raise ValueError('reporting_queue shoud be Queue.')

        while True:
            try:
                prob = self.jobs_queue.get(timeout=self.TIMEOUT)
                if prob.final_queue_flag is True:
                    SubVerifier.logger.info(
                        'SubVerifier {} started job {}, '.format(
                            self.id, prob.id
                        )
                    )
                    slv_report = self.verify(prob)
                    SubVerifier.logger.info(
                        'SubVerifier {} finished job {}, result: {}, time: {:.2f}.'.format(
                            self.id,
                            prob.id,
                            slv_report.result.value,
                            slv_report.runtime
                        )
                    )
                    self.reporting_queue.put(slv_report)
                else:
                    ##A
                    ndsplit = NodeSplitter(prob,self.config)
                    lp_queue = Queue()
                    lp_queue.put(prob)
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
                    init_dep = prob.depth
                    while not lp_queue.empty()  :
                        p  = lp_queue.get()
                        if p.config.BENCHMARK == 'a':
                            num = 200
                        else:
                            num = 3
                        if p.depth == init_dep + num:
                            lp_queue.put(p)
                            break
                        xx  = timer()
                        if p.config.BENCHMARK == 'a':
                            impact = p.get_most_impactfull_neurons(output_eq, lower=False)
                        else:
                            impact = p.get_most_impactfull_neurons(output_eq, lower=False, include_inp_node=False)

                        sn = impact[1]
                        p1,p2 = ndsplit.split_node_by_impact_2(p, sn[0])
                        # print(p1.depth)
                        slv_report1 = self.verify_lp(p1)
                        slv_report2 = self.verify_lp(p2)
                        print(slv_report1.result)
                        print(slv_report2.result)
                        if slv_report1.result == SolveResult.SAFE  and slv_report2.result == SolveResult.SAFE:
                            continue

                        if slv_report1.result == SolveResult.UNSAFE:
                            self.reporting_queue.put(slv_report1)
                            break
                        if slv_report2.result == SolveResult.UNSAFE:
                            self.reporting_queue.put(slv_report2)
                            break
                        if slv_report1.result == SolveResult.UNDECIDED:
                            p1.dep_by_lp = list(p.dep_by_lp)
                            p1.dep_by_lp.append((sn[0],1))
                            lp_queue.put(p1)
                        elif slv_report1.result == SolveResult.SAFE:
                            dep_by_lp = list(p.dep_by_lp)
                            dep_by_lp.append((sn[0],0))
                            prob.dep_by_lp_ls.append(dep_by_lp)

                        if slv_report2.result == SolveResult.UNDECIDED:
                            p2.dep_by_lp = list(p.dep_by_lp)
                            p2.dep_by_lp.append((sn[0],0))
                            lp_queue.put(p2)
                        elif slv_report2.result == SolveResult.SAFE:
                            dep_by_lp = list(p.dep_by_lp)
                            dep_by_lp.append((sn[0],1))
                            prob.dep_by_lp_ls.append(dep_by_lp)
                            #pass
                    if lp_queue.qsize() == 0:
                        self.reporting_queue.put(SolveReport(SolveResult.SAFE, 0, None))
                    else:
                        slv_report = self.verify(prob)
                        SubVerifier.logger.info(
                            'SubVerifier {} finished job {}, result: {}, time: {:.2f}.'.format(
                                self.id,
                                prob.id,
                                slv_report.result.value,
                                slv_report.runtime
                            )
                        )
                        self.reporting_queue.put(slv_report)

            except Exception as error:
                print(traceback.format_exc())
                SubVerifier.logger.info(
                    f"Subprocess {self.id} terminated because of {str(error)}."
                )
                break


    def verify(self, prob):
        start = timer()

        if prob.inc_ver_done is not True:
            slv_report = self.verify_incomplete(prob)
            if slv_report.result != SolveResult.UNDECIDED:
                return slv_report

        slv_report = self.verify_complete(prob)
        slv_report.runtime = timer() - start
        return slv_report


    def verify_incomplete(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using projected gradient
        descent and symbolic interval propagation.

        Returns:
            SolveReport
        """
        start = timer()
        prob.inc_ver_done = True
        slv_report = SolveReport(SolveResult.UNDECIDED, 0, None)
        SubVerifier.num_sum += 1

        #try pgd
        if slv_report.result == SolveResult.UNDECIDED and self.config.VERIFIER.PGD is True and prob.pgd_ver_done is not True:
            subreport = self.verify_pgd(prob)
            slv_report.result = subreport.result
            slv_report.cex = subreport.cex
            SubVerifier.num_pgd =0
            if slv_report.result == SolveResult.UNSAFE:
                SubVerifier.num_pgd += 1
                print(SubVerifier.num_pgd)
                print("solved by pgd+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # try bound analysis
        if slv_report.result == SolveResult.UNDECIDED and \
        prob.bounds_ver_done is not True:
            subreport = self.verify_bounds(prob)
            for i in range(1,prob.nn.tail.depth+1):
                nodes = prob.nn.get_node_by_depth(i)
            slv_report.result = subreport.result
            if slv_report.result == SolveResult.SAFE:
                print("solved by bound_analysis++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        # try LP analysis
        if slv_report.result == SolveResult.UNDECIDED and \
        self.config.VERIFIER.LP is True and \
        prob.lp_ver_done is not True:
            subreport = self.verify_lp(prob)
            slv_report.result = subreport.result
            slv_report.cex = subreport.cex
            if slv_report.result == SolveResult.SAFE:
                print("solved by LP+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        prob.detach()
        prob.clean_vars()
        prob.spec.input_node.bounds.detach()
        slv_report.runtime = timer() - start
        return slv_report

    def verify_pgd(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using projected gradient
        descent.

        Returns:
            SolveReport
        """
        start = timer()

        prob.pgd_ver_done = True

        pgd = ProjectedGradientDescent(self.config)
        cex = pgd.start(prob)
        if cex is not None:
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via PGD'
            )
            return SolveReport(SolveResult.UNSAFE, timer() - start, cex)

        SubVerifier.logger.info(
            f'PGD done. Verification problem {prob.id} could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start, cex)


#a
    def verify_preattack(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using projected gradient
        descent.

        Returns:
            SolveReport
        """
        start = timer()

        prob.preattack_ver_done = True

        pgd = PreAttack(self.config,prob)
        cex = pgd.pre_process_attack(prob)
        if cex is not None:
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via PGD'
            )
            return SolveReport(SolveResult.UNSAFE, timer() - start, cex)

        SubVerifier.logger.info(
            f'PGD done. Verification problem {prob.id} could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start, cex)


    def verify_bounds(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using bounds.

        Returns:
            SolveReport
        """
        start = timer()

        prob.bound_ver_done = True
        prob.bound_analysis()
        if prob.satisfies_spec():
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via bound analysis')

            return SolveReport(SolveResult.SAFE, timer() - start, None)

        SubVerifier.logger.info(
            f'Bound analysis done. Verification problem {prob.id} could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start, None)

    def verify_lp(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using linear relaxation.

        Returns:
            SolveReport
        """
        start = timer()

        prob.lp_ver_done = True

        solver = MILPSolver(prob, self.config, lp=True)
        slv_report =  solver.solve()

        if slv_report.result == SolveResult.SAFE:
            SubVerifier.logger.info(
                f'Verification problem {prob.id} was solved via LP')
            return slv_report

        elif slv_report.result == SolveResult.UNSAFE and self.config.VERIFIER.PGD_ON_LP is True:
            cex = torch.tensor(
                slv_report.cex, dtype=self.config.PRECISION, device=prob.device
            )
            pgd = ProjectedGradientDescent(self.config)
            cex = pgd.start(prob, init_adv=cex, device=prob.device)
            if cex is not None:
                SubVerifier.logger.info(
                    f'Verification problem {prob.id} was solved via PGD on LP'
                )
                return SolveReport(SolveResult.UNSAFE, timer() - start, cex)

        SubVerifier.logger.info(
            'LP analysis done. Verification problem could not be solved'
        )

        return SolveReport(SolveResult.UNDECIDED, timer() - start)

    def verify_complete(self, prob: VerificationProblem):
        """
        Attempts to solve a verification problem using MILP.

        Returns:
            SolveReport
        """
        solver = MILPSolver(prob, self.config)
        slv_report = solver.solve()
        prob.detach()
        prob.clean_vars()
        return slv_report

