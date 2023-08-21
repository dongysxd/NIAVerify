#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2022
# @Author  : LYH
# @File    : preattack.py
# @IDE     : pycharm

import torch
import torch.optim as optim
from typing import Callable, Optional
class PreAttack:
    def __init__(self, config,prob):
        self.config = config
        self.idx = 0
        self.prob = prob

    def grad_descent_loss(self, y: torch.Tensor) -> torch.Tensor:

        """
        Returns the loss for gradient descent.

        Args:
            y:
                The neural network output
        Returns:
            The loss function
        """

        constr_eq = self.prob.spec.output_formula.clauses[self.idx]
        op1 = constr_eq.op1.i
        op2= constr_eq.op2.i
        loss = torch.zeros(1).to(device=self.config.DEVICE)
        loss += torch.clamp((-1 * y[op1]  + y[op2]), 0, 1e8)
        return loss


    def pre_process_attack(self,init_adv: torch.tensor=None, device=torch.device('cpu')):

        """
        Performs a simple adversarial attack in an attempt to find a counter example.

        Note that this method should never be called within the main verification
        loop in verify(), it is meant for external use only.
        """

        counter_example = None
        for idx in range(len(self.prob.spec.output_formula.clauses)-1,-1,-1 ):
            self.idx = idx
            mid_point = self.generate_mid_adv(self.prob.spec.input_node.bounds)
            loss_func =  self.grad_descent_loss

            counter_example = self._grad_descent_counter_example(potential_cex=mid_point,
                                                                 loss_func=loss_func,
                                                                 do_grad_descent=True)

            if counter_example is not None:
                torch.cuda.empty_cache()
                return counter_example
        torch.cuda.empty_cache()

        return counter_example


    def _grad_descent_counter_example(self, potential_cex, loss_func: Callable,
                                      do_grad_descent: bool = True):

        """
        Runs gradient descent updating the input to find true counter examples.

        Args:
            potential_cex:
                The counter example candidate.
            loss_func:
                The loss function used for gradient descent.
            do_grad_descent:
                If true, gradient descent is performed to find a counter-example.

        Returns:
            The counter example if found, else None.
        """
        lower = self.prob.spec.input_node.bounds.lower[0,:,0].to(device=self.config.DEVICE).clone()
        upper = self.prob.spec.input_node.bounds.upper[0,:,0].to(device=self.config.DEVICE).clone()
        x = potential_cex.to(device=self.config.DEVICE).clone()
        x = x.view(-1)
        idx1 = x <  lower
        idx2 = x >  upper
        x.data[idx1] = lower[idx1]
        x.data[idx2] = upper[idx2]
        x = x.view(*self.prob.spec.input_node.input_shape)
        x.requires_grad = True
        y = self.prob.nn.forward(x)
        if self.prob.spec.is_satisfied(y, y) is not True:
            return x.detach()

        optimizer = optim.Adam([x], lr=0.1, betas=(0.5, 0.9))

        old_loss = 1e10

        for i in range(5):
            optimizer.zero_grad()
            loss = loss_func(y)
            loss.backward()
            optimizer.step()
            x = x.view(-1)
            idx1 = x <  lower
            idx2 = x >  upper
            x.data[idx1] = lower[idx1]
            x.data[idx2] = upper[idx2]
            x = x.view(*self.prob.spec.input_node.input_shape)
            y = self.prob.nn.forward(x)
            if self.prob.spec.is_satisfied(y, y) is not True:
                return x.detach()

            if ((old_loss - loss) / old_loss) <  0.01:
                return None

            old_loss = loss

        return None



    def generate_mid_adv(self, bounds):
        adv = (bounds.lower.clone().to(device=self.config.DEVICE) + bounds.upper.clone().to(self.config.DEVICE))/2.0
        return adv.float()

    

    def generate_random_adv(self, bounds):
        adv = torch.zeros_like(bounds.lower)
        idxs = bounds.lower < bounds.upper
        distribution = torch.distributions.uniform.Uniform(
            bounds.lower[idxs], bounds.upper[idxs]
        )
        partial_adv = distribution.sample(torch.Size([1]))
        partial_adv = torch.squeeze(partial_adv, 0)

        adv[idxs] = partial_adv

        return adv

    def fast_gradient_signed(
        self,
        prob,
        x,
        eps,
        device=torch.device('cpu')
    ):
        """
        Fast Gradient Signed Method.

        Arguments: 
            prob:
                Verification Problem.
            x:
                Input tensor.
            eps:
                Epsilon.
            targeted:
                Whether or not the attack is targeted.
        Returns: 
            A tensor for the adversarial example.
        """
        x = x.clone().detach().to(self.config.PRECISION).requires_grad_(True)

        true_label = prob.spec.is_adversarial_robustness()

        if true_label == -1:
            output = prob.nn.forward(x).flatten()
            loss = prob.spec.get_mse_loss(output)

        else:
            output_flag =  prob.spec.get_output_flag(prob.nn.tail.output_shape)
            output = prob.nn.forward(x)[output_flag].flatten()[None, :]
            true_label = torch.sum(output_flag.flatten()[0: true_label])
            y = torch.tensor([true_label], device=device)
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(output, y)

        # Compute gradient
        # loss = -loss
        loss.backward()

        # compute perturbation
        perturbation = eps * torch.sign(x.grad)

        if torch.all(perturbation == 0):
            adv = self.generate_random_adv(prob.spec.input_node.bounds)

        else:
            adv = torch.clamp(
                x + perturbation,
                prob.spec.input_node.bounds.lower,
                prob.spec.input_node.bounds.upper
            )


        return adv
