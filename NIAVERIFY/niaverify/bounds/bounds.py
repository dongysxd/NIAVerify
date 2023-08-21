#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : bounds.py
# @IDE     : pycharm
import numpy as np

class Bounds:
    def __init__(self, lower=None, upper=None):
        self.lower = lower
        self.upper = upper

    def size(self):
        """
        Calculates the size of the bounds.
        """
        return 0 if self.lower is None else self.lower.nelement()

    def normalise(self, mean, std):
        """
        Normalises the bounds

        Arguments:
            mean:
                normalisation mean
            std:
                normalisation standard deviation
        Returns
            None
        """
        self.lower = ( self.lower - mean ) / std
        self.upper = ( self.upper - mean ) / std

    def clip(self, min_value, max_value):
        """
        Clips the  bounds

        Arguments:
            min_value:
                valid lower bound
            max_value:
                valid upper bound
        Returns:
            None
        """
        self.lower = np.clip(self.lower, min_value, max_value) 
        self.upper = np.clip(self.upper, min_value, max_value) 

    def get_range(self):
        """
        Returns the range of the bounds.
        """
        return self.upper - self.lower

    def copy(self):
        """
        Copies the bounds.
        """
        self.detach()
        lower = self.lower.clone() if self.lower is not None else None
        upper = self.upper.clone() if self.upper is not None else None

        return Bounds(lower, upper)

    def detach(self):
        """ 
        Detaches and clones the bound tensors. 
        """
        if self.lower is not None:
            self.lower = self.lower.detach().clone()
        if self.upper is not None:
            self.upper = self.upper.detach().clone()

    def cpu(self):
        """ 
        Moves bounds to cpu memory.
        """
        lower = self.lower.cpu() if self.lower is not None else None
        upper = self.upper.cpu() if self.upper is not None else None

        return Bounds(lower, upper)

    def cuda(self):
        """ 
        Moves bounds to gpu memory.
        """
        lower = self.lower.cuda() if self.lower is not None else None
        upper = self.upper.cuda() if self.upper is not None else None

        return Bounds(lower, upper)
