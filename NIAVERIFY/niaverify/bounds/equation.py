#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) 2022 - 2023 ICTT  All Rights Reserved 
#
# @Time    : 2023
# @Author  : LYH
# @File    : equation.py
# @IDE     : pycharm
from __future__ import annotations

from niaverify.network.node import *
from niaverify.common.configuration import Config

torch.set_num_threads(1)

class Equation():
    """
    The Equation class.
    """

    def __init__(self, matrix: torch.Tensor, const: torch.Tensor, config: Config):
        """
        Arguments:

            matrix:
                2D matrix (of size nxm). Each row i represents node's i
                equation coefficients.
            const:
                Vector (of size n). Each row i represents node's i equation
                constant term.
        """
        self.matrix = matrix
        self.const = const
        self.config = config
        self.size, self.coeffs_size = matrix.shape


    def copy(self, matrix=None, const=None) -> Equation:
        return Equation(
            self.matrix.detach().clone(),
            self.const.detach().clone(),
            self.config
        )


    def zero(self) -> Equation:
        """
        Returns whether the equation is the zero function.
        """
        return self.size == 0

    def add(self, eq: Equation) -> Equation:
        """
        Adds the equation to another.

        Arguments:
            eq:
                The equation to add.
        Returns.
            An equation representing the sum.
        """
        return Equation(
            self.matrix + eq.matrix,
            self.const + eq.const,
            self.config
        )
 
    def get_plus_matrix(self) -> torch.Tensor:
        """
        Clips the coeffs to be only positive.
        """

        return torch.clamp(self.matrix, 0, math.inf)


    def get_minus_matrix(self, keep_in_memory=True) -> torch.Tensor:
        """
        Clips the coeffs to be only negative.
        """     
        return torch.clamp(self.matrix, -math.inf, 0)


    def concrete_values(
        self, lower: torch.Tensor, upper:torch.Tensor, bound: str
    ) -> torch.Tensor:
        if bound == 'lower':
            return self.min_values(lower, upper)
        
        elif bound == 'upper':
            return self.max_values(lower, upper)
       
        else:
            raise ValueError(f'Bound {bound} is not recognised.')
        

    def max_values(self, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
        """
        Computes the upper bounds of the equations.
    
        Arguments:

            lower:
                The lower bounds for the variables in the equation. 
            upper:
                The upper bounds for the variables in the equation.
        
        Returns: 

            The upper bounds of the equation.
        """
        return  self.interval_dot('upper', lower, upper)


    def min_values(self, lower, upper):
        """
        Computes the lower bounds of the equations.

        Arguments:

            lower:
                The lower bounds for the variables in the equation. 
            upper:
                The upper bounds for the variables in the equation.

        Returns:

            The lower bounds of the equations.
        """

        return self.interval_dot('lower', lower, upper)


    def interval_dot(
        self, bound: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the interval dot product with either a matrix or an Equation.
        """
        if isinstance(lower, Equation) and isinstance(upper, Equation):
            return self._interval_dot_eq(bound, lower, upper)

        elif isinstance(lower, torch.Tensor) and isinstance(upper, torch.Tensor):
            return self._interval_dot_tensor(bound, lower, upper)

        else: 
            raise TypeError(f'Got {type(lower)} and {type(upper)} but expected either trensor or Equation')

    def _interval_dot_eq(
        self, bound: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the interval dot product with an Equation.
        """
        plus, minus = self.get_plus_matrix(), self.get_minus_matrix()

        if bound == 'upper':
            matrix =  plus @ upper.matrix + minus @ lower.matrix
            const = plus @ upper.const + minus @ lower.const + self.const

        elif bound == 'lower':
            matrix = plus @ lower.matrix + minus @ upper.matrix
            const = plus @ lower.const + minus @ upper.const + self.const

        else: 
            raise ValueError(f'Bound {bound} is not recognised.')

        return Equation(matrix, const, self.config)


    def _interval_dot_tensor(
        self, bound: str, lower: torch.Tensor, upper: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the interval dot product with either a tensor
        """
        if bound == 'upper':
            return self.get_plus_matrix() @ upper + \
                self.get_minus_matrix() @ lower + \
                self.const

        elif bound == 'lower':
            return  self.get_plus_matrix() @ lower + \
                self.get_minus_matrix() @ upper + \
                self.const

        else: 
            raise ValueError(f'Bound {bound} is not recognised.')

    def get_relu_slope(
        self, 
        node: Node,
        slope_type: str,
        bound: str,
        out_flag: torch.tensor=None,
        slopes: torch.tensor=None
    ) -> torch.Tensor:
        if out_flag is None:
            if slope_type  == 'lower':
                sl = torch.ones(
                    node.output_size,
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                sl[node.to_node[0].get_inactive_flag().flatten()] = 0.0
                idxs = node.to_node[0].get_unstable_flag().flatten()
                #idxs = node.get_unstable_flag().flatten()
                if slopes is None:
                    slopes = node.to_node[0].get_lower_relaxation_slope()

                    # if(sum(idxs) != len(slopes[0]))  and bound =='lower':
                    #     print(sum(idxs) == len(slopes[0]))
                    # if (sum(idxs) != len(slopes[1]))  and bound =='upper':
                    #     print(sum(idxs) == len(slopes[1]))
                    # print(sum(idxs),len(slopes[0]))
                    sl[idxs] = slopes[0] if bound == 'lower' else slopes[1]
                else:
                    sl[idxs] = slopes

            elif slope_type == 'upper':
                sl = torch.zeros(
                    node.output_size,
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                idxs = node.to_node[0].get_unstable_flag().flatten()
                lower = node.bounds.lower.flatten()[idxs] 
                upper = node.bounds.upper.flatten()[idxs]
                sl[idxs] =  upper / (upper - lower)
                sl[node.to_node[0].get_active_flag().flatten()] = 1

            else:
                raise Exception(f"Slope type {slope_type} is not recognised.")

        else:
            if slope_type == 'lower':
                sl = torch.ones(
                    node.to_node[0].get_propagation_count(),
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                ) 
                upper = node.bounds.upper[out_flag].flatten()
                lower = node.bounds.lower[out_flag].flatten()
                if slopes is None:
                    idxs = lower < 0 
                    slopes = node.to_node[0].get_lower_relaxation_slope()
                    sl[idxs] = slopes[0] if bound == 'lower' else slopes[1]
                else:
                    idxs = abs(lower) >=  upper
                    sl[idxs] = 0.0

            elif slope_type == 'upper':
                lower = node.bounds.lower[out_flag].flatten()
                upper = node.bounds.upper[out_flag].flatten()
                
                sl = torch.ones(
                    node.to_node[0].get_propagation_count(),
                    dtype=self.config.PRECISION,
                    device=self.config.DEVICE
                )
                idxs = lower < 0
                sl[idxs] = upper[idxs] /  (upper[idxs] - lower[idxs])

            else:
                raise Exception(f"Slope type {slope_type} is not recognised.")

        return sl

    def get_relu_const(
        self, 
        node: Node, 
        const: torch.tensor,
        const_type :str,  
        relu_slope: torch.Tensor,
        out_flag: torch.Tensor=None
    ) -> torch.Tensor:
        """
        Derives the constant tensor of a relu relaxation.

        Arguments:
            node:
                Node with relu activation.
            const:
                The constant tensor of the node.
            const_type:
                Either 'lower' relu const or 'upper' relu const.
            relu_slope:
                The slope tensor of the relu relaxation.
            out_flag:
                Binary flag of output indices to return.

        Returns:
            The constant tensor of a relu relaxation.
        """
        if out_flag is None:
            if const_type == 'lower':
                relu_const = const * relu_slope

            elif const_type == 'upper':
                idxs = node.to_node[0].get_unstable_flag().flatten()
                lower  = node.bounds.lower.flatten()[idxs]
                relu_const = const * relu_slope
                relu_const[idxs]  -= relu_slope[idxs] * lower

            else:
                raise Exception(f"Const type {const_type} is not recognised.")

        else:
            if const_type == 'lower':
                relu_const = const * relu_slope
            
            elif const_type == 'upper':
                lower = node.bounds.lower.flatten()[out_flag]
                idxs = lower < 0
                relu_const = const * relu_slope
                relu_const[idxs]  -= relu_slope[idxs] *  lower[idxs]

        return relu_const

    def get_max_pool_relaxation(self, node: Node) -> tuple:
        lower, indices = node.to_node[0].forward(node.bounds.lower, return_indices=True)
        
        idx_correction = torch.tensor(
            [i * node.in_ch_sz() for i in range(node.in_ch())],
            dtype=torch.long, 
            device=self.config.DEVICE
        ).reshape((node.in_ch(), 1, 1))
        if node.has_batch_dimension():
            idx_correction = idx_correction[None, :]
        indices = indices + idx_correction

        lower, indices = lower.flatten(), indices.flatten()
        upper = node.to_node[0].forward(node.bounds.upper).flatten()
        lower_max  = lower > upper
        not_lower_max = torch.logical_not(lower_max)

        lower_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        lower_slope[indices] = 1.0
        lower_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        lower_const[indices] = Equation.derive_const(node)[indices]

        upper_slope = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_slope[indices][lower_max] = 1.0
        upper_const = torch.zeros(
            node.input_size, dtype=self.config.PRECISION, device=self.config.DEVICE
        )
        upper_const[indices][lower_max] = Equation.derive_const(node)[indices][lower_max]
        upper_const[indices][not_lower_max] = node.bounds.upper.flatten()[indices][not_lower_max]

        return (lower_slope, upper_slope), (lower_const, upper_const)


    @staticmethod
    def derive(
        node: Node,
        config: Config,
        out_flag: torch.Tensor=None,
        in_flag: torch.Tensor=None,
        sparse: bool=False
    ) -> Equation:
        zero_eq = Equation._zero_eq(node, out_flag)
      
        if zero_eq is not None:
            return zero_eq
      
        else:
            return Equation(
                Equation._derive_matrix(node, out_flag, in_flag, sparse),
                Equation.derive_const(node, out_flag),
                config
            )

    @staticmethod 
    def _zero_eq(node: Node, flag: torch.Tensor) -> Equation:
        if isinstance(node.from_node[0], Relu) and \
        node.from_node[0].get_propagation_count() == 0:
            return Equation(
                torch.zeros(
                    (torch.sum(flag), 0),
                    dtype=node.config.PRECISION,
                    device=node.config.DEVICE
                ),
                Equation.derive_const(node, flag),
                node.config
            )

        return None

    @staticmethod
    def _derive_matrix(
            node: Node,
            out_flag: torch.Tensor=None,
            in_flag: torch.Tensor=None,
            sparse: bool=False
    ):
        if out_flag is None:
            out_flag = torch.ones(
                node.output_size, dtype=torch.bool, device=node.config.DEVICE
            )
        
        if isinstance(node, Conv):
            return Equation._derive_conv_matrix(node, out_flag, in_flag, sparse)

        if isinstance(node, Gemm):
            return Equation._derive_gemm_matrix(node, out_flag, in_flag)

        if isinstance(node, MatMul):
            return Equation._derive_matmul_matrix(node, out_flag, in_flag)
        
        if isinstance(node, Add):
            return Equation._derive_add_matrix(node, out_flag)

        if isinstance(node, Flatten):
            return Equation._derive_flatten_matrix(node, out_flag, in_flag)

        if isinstance(node, BatchNormalization):
            return Equation._derive_batch_normalization_matrix(node, out_flag, in_flag)

        raise NotImplementedError(f'{type(node)} equations')

    @staticmethod 
    def _derive_conv_matrix(
            node: Node,
            out_flag: torch.Tensor,
            in_flag: torch.Tensor=None,
            sparse: bool=False
    ):
        flag_size = torch.sum(out_flag).item()

        prop_flag = torch.zeros(
            node.get_input_padded_size(), dtype=torch.bool, device=node.config.DEVICE
        )
        if in_flag is None:
            prop_flag[node.get_non_pad_idxs()] = True
            max_index = node.input_size
        else:
            prop_flag[node.get_non_pad_idxs()] = in_flag.flatten()
            max_index = node.from_node[0].get_propagation_count()

        pad = torch.ones(
            node.get_input_padded_size(),
            dtype=torch.long,
            device=node.config.DEVICE
        ) * max_index
        pad[prop_flag] = torch.arange(max_index, device=node.config.DEVICE)

        im2col = Conv.im2col(
            pad.reshape(node.get_input_padded_shape()),
            (node.krn_height, node.krn_width),
            node.strides,
            device=node.config.DEVICE
        )
        indices = torch.arange(
            node.out_ch_sz,
            device=node.config.DEVICE
        ).repeat(node.out_ch)[out_flag]
        conv_indices = im2col[:, indices]

        indices = torch.repeat_interleave(
            torch.arange(node.out_ch, device=node.config.DEVICE), node.out_ch_sz, dim=0
        )[out_flag]
        conv_weights = node.kernels.permute(1, 2, 3, 0).reshape(-1, node.out_ch)[:, indices]
       
        if sparse is True:
            matrix =[
                {
                    conv_indices[i, eq]: conv_weights[i, eq].item()
                    for i in range(np.prod(node.kernels.shape[1:]))
                    if not conv_indices[i, eq] == max_index
                }
                for eq in range(flag_size)
            ]
        
        else:
            matrix = torch.zeros(
                (flag_size, max_index + 1),
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            )
            matrix[torch.arange(flag_size), conv_indices] = conv_weights
            matrix = matrix[:, :max_index]

        return matrix

    @staticmethod 
    def _derive_gemm_matrix(
            node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor=None
    ):
        if in_flag is None:
            matrix = node.weights[out_flag, :]
        else:
            matrix = node.weights[out_flag, :][:, in_flag]

        return matrix


    @staticmethod 
    def _derive_matmul_matrix(
            node: Node, out_flag: torch.Tensor, in_flag:torch.Tensor=None
    ):
        if in_flag is None:
            matrix = node.weights[out_flag, :]
        else:
            matrix = node.weights[out_flag, :][:, in_flag]

        return matrix

    @staticmethod 
    def _derive_batch_normalization_matrix(
            node: Node, out_flag: torch.Tensor, in_flag: torch.Tensor=None
    ):
        in_ch_sz = node.in_ch_sz()
        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()
        scale_var = scale / var

        matrix = torch.zeros(
            (node.output_size, node.output_size),
            dtype=node.config.PRECISION,
            device=node.config.DEVICE
        )
        matrix[torch.eye(node.output_size).bool()] = scale_var

        return matrix[out_flag, :]

    @staticmethod 
    def _derive_add_matrix(node: Node, flag: torch.Tensor):
        return  torch.eye(
            node.input_size, dtype=node.config.PRECISION, device=node.config.DEVICE
        )[flag, :]

    @staticmethod 
    def _derive_flatten_matrix(
        node: Node, out_flag:torch.tensor=None, in_flag: torch.tensor=None
    ):
        if in_flag is None:
            matrix = torch.eye(
                node.output_size,
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            ).squeeze()[out_flag, :]
        else:
            matrix = torch.eye(
                node.output_size,
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            ).squeeze()[out_flag, :][:, in_flag]

        return matrix

    @staticmethod
    def derive_const(node: Node, flag: torch.Tensor=None):
        if isinstance(node, Conv):
            return Equation._derive_conv_const(node, flag)

        if isinstance(node, ConvTranspose):
            return Equation._derive_convtranspose_const(node, flag)

        if isinstance(node, Gemm):
            return Equation._derive_gemm_const(node, flag)

        if isinstance(node, MatMul):
            return Equation._derive_matmul_const(node, flag)

        if isinstance(node, BatchNormalization):
            return Equation._derive_batch_normalization_const(node, flag)
        
        if isinstance(node, Add):
            return Equation._derive_add_const(node, flag)

        if isinstance(node, Flatten):
            return Equation._derive_flatten_const(node, flag)

        raise NotImplementedError(f'{type(node)} equations')

    @staticmethod 
    def _derive_gemm_const(node: Node, flag: torch.Tensor): 
        if node.has_bias() is True:
            const = node.bias.detach().clone() if flag is None else node.bias[flag]
        else:
            size = node.output_size if flag is None else torch.sum(flag)
            const = torch.zeros(
                size, dtype=node.config.PRECISION, device=node.confog.DEVICE
            )
        
        return const

    @staticmethod 
    def _derive_conv_const(node: Node, flag: torch.Tensor):
        if node.has_bias() is True:
            if flag is None:
                const =  torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()
            else:
                const = torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]
        
        else:
            size = node.output_size if flag is None else torch.sum(flag)
            const = torch.zeros(
                size, dtype=node.config.PRECISION, device=node.config.DEVICE
            )

        return const

    @staticmethod 
    def _derive_convtranspose_const(node: Node, flag: torch.Tensor):
        if node.has_bias() is True:
            if flag is None:
                const = torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()
            else:
                const = torch.tile(node.bias, (node.out_ch_sz, 1)).T.flatten()[flag]
        
        else:
            size = node.output_size if flag is None else torch.sum(flag)
            const = torch.zeros(
                size, dtype=node.config.PRECISION, device=node.confog.DEVICE
            )

        return const

    @staticmethod 
    def _derive_matmul_const(node: Node, flag: torch.Tensor):
        if flag is None:
            const =  torch.zeros(
                node.output_size, dtype=node.config.PRECISION, device=node.config.DEVICE
            )
        else:
            const = torch.zeros(
                torch.sum(flag), dtype=node.config.PRECISION, device=node.config.DEVICE
            )

        return const

    @staticmethod
    def _derive_flatten_const(node: Node, flag: torch.Tensor):
        if flag is None:
            const =  torch.zeros(
                node.output_size,
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            )
        else:
            const = torch.zeros(
                torch.sum(flag).item(),
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            )

        return const

    @staticmethod
    def _derive_add_const(node: Node, flag: torch.Tensor):
        if flag is None:
            const = torch.zeros(
                node.output_size,
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            )
        else:
            const = torch.zeros(
                torch.sum(flag).item(),
                dtype=node.config.PRECISION,
                device=node.config.DEVICE
            )

        return const

    @staticmethod
    def _derive_batch_normalization_const(node: Node, flag: torch.Tensor):
        in_ch_sz = node.in_ch_sz()

        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()

        const = - input_mean / var * scale + bias
        if flag is not None:
            const = const[flag]

        return const


"""
    @staticmethod
    def interval_forward(
            node: Node,
            bound: str,
            lower_eq: Equation,
            upper_eq: Equation
    ):

        if isinstance(node, Gemm):
            return Equation._interval_gemm_forward(node, bound, lower_eq, upper_eq)

        elif isinstance(node, Conv):
            return Equation._interval_conv_forward(node, bound, lower_eq, upper_eq)

        elif isinstance(node, BatchNormalization):
            return Equation._interval_batch_normalization_forward(
                node, bound, lower_eq, upper_eq
            )

        else:
            raise NotImplementedError(f'Interval forward for node {type(node)}')

    @staticmethod
    def _interval_gemm_forward(
            node: Node,
            bound: str,
            lower_eq: Equation,
            upper_eq: Equation
    ):
        if  bound == 'lower':
            matrix = node.forward(
                lower_eq.matrix, clip='+', add_bias=False
            ) + node.forward(
                upper_eq.matrix, clip='-', add_bias=False
            )
            const = node.forward(
                lower_eq.const, clip='+', add_bias=False
            ) + node.forward(
                upper_eq.const, clip='-', add_bias=True
                
            )

        elif bound == 'upper':
            matrix = node.forward(
                lower_eq.matrix, clip='-', add_bias=False
            ) + node.forward(
                upper_eq.matrix, clip='+', add_bias=False
            )
            const = node.forward(
                lower_eq.const, clip='-', add_bias=False
            ) + node.forward(
                upper_eq.const, clip='+', add_bias=True
            )

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        return Equation(matrix, const, node.config)


    @staticmethod
    def _interval_conv_forward(
            node: Node,
            bound: str,
            lower_eq: Equation,
            upper_eq: Equation
    ):
        shape = (lower_eq.coeffs_size,) + node.input_shape_no_batch()

        if  bound == 'lower':
            matrix = node.forward(
                lower_eq.matrix.T.reshape(shape), clip='+', add_bias=False
            ) + node.forward(
                upper_eq.matrix.T.reshape(shape), clip='-', add_bias=False
            )
            const = node.forward(
                lower_eq.const.reshape(node.input_shape), clip='+', add_bias=False
            ).flatten()
            const += node.forward(
                upper_eq.const.reshape(node.input_shape), clip='-', add_bias=True
            ).flatten()

        elif bound == 'upper':
            matrix = node.forward(
                lower_eq.matrix.T.reshape(shape), clip='-', add_bias=False
            ) + node.forward(
                upper_eq.matrix.T.reshape(shape), clip='+', add_bias=False
            )
            const = node.forward(
                lower_eq.const.reshape(node.input_shape), clip='-', add_bias=False
            ).flatten()
            const += node.forward(
                upper_eq.const.reshape(node.input_shape), clip='+', add_bias=True
            ).flatten()

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        matrix = matrix.reshape(lower_eq.coeffs_size, -1).T

        return Equation(matrix, const, node.config)

    @staticmethod
    def _interval_batch_normalization_forward(
        node: Node, bound: str, lower_eq: Equation, upper_eq: Equation
    ) -> Equation:
        in_ch_sz = node.in_ch_sz()
        
        scale = torch.tile(node.scale, (in_ch_sz, 1)).T.flatten()
        bias = torch.tile(node.bias, (in_ch_sz, 1)).T.flatten()
        input_mean = torch.tile(node.input_mean, (in_ch_sz, 1)).T.flatten()
        var = torch.sqrt(node.input_var + node.epsilon)
        var = torch.tile(var, (in_ch_sz, 1)).T.flatten()
        scale_var = scale / var

        matrix = torch.zeros(
            lower_eq.matrix.shape,
            dtype=lower_eq.config.PRECISION,
            device=node.config.DEVICE
        )
        const = torch.zeros(
            lower_eq.const.shape,
            dtype=lower_eq.config.PRECISION,
            device=node.config.DEVICE
        )

        if bound == 'lower':
            idxs = scale_var < 0
            matrix[idxs, :] = (upper_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = upper_eq.const[idxs] * scale_var[idxs]
            idxs = scale_var >= 0
            matrix[idxs, :] = (lower_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = lower_eq.const[idxs] * scale_var[idxs]
            
        elif bound == 'upper':
            idxs = scale_var < 0
            matrix[idxs, :] = (lower_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = lower_eq.const[idxs] * scale_var[idxs]
            idxs = scale_var >= 0
            matrix[idxs, :] = (upper_eq.matrix[idxs, :].T * scale_var[idxs]).T
            const[idxs] = upper_eq.const[idxs] * scale_var[idxs]

        else:
            raise ValueError(f"Bound type {bound} could not be recognised.")

        batch_const = - input_mean / var * scale + bias
        const += batch_const 

        return Equation(matrix, const, node.config)

    



    def transpose(
            self, node: Node, out_flag: torch.Tensor=None, in_flag: torch.Tensor=None
    ):
        if type(node) in [Gemm, Conv, ConvTranspose]:
            tr_eq =  self._transpose_affine(node, in_flag, out_flag)

        elif isinstance(node, MatMul):
            tr_eq = self._transpose_linear(node, in_flag, out_flag)

        elif isinstance(node, Slice):
            tr_eq = self._transpose_slice(node)

        elif isinstance(node, Concat):
            tr_eq = self._transpose_concat(node)

        elif isinstance(node, Sub):
            tr_eq = self._transpose_sub(node)

        elif isinstance(node, Add):
            tr_eq = self._transpose_add(node)

        elif isinstance(node, Reshape):
            tr_eq = self._transpose_reshape(node)

        else:
            raise NotImplementedError(f'Equation transpose for {type(node)}')

        return tr_eq if isinstance(tr_eq, list) else [tr_eq]

    def _transpose_affine(self, node: Node, in_flag: torch.tensor, out_flag: torch.tensor):
        if in_flag is None:
            shape = (self.size,) + node.output_shape_no_batch()
        else:
            shape = self.matrix.shape

        matrix =  node.transpose(self.matrix.reshape(shape), in_flag, out_flag)
        matrix = matrix.reshape(self.size, -1)

        const = Equation.derive_const(node, out_flag)
        const = (self.matrix @ const) + self.const

        return Equation(matrix, const, self.config)

    def _transpose_linear(self, node: Node, in_flag: torch.tensor, out_flag:torch.tensor):
        if in_flag is None:
            matrix =  node.transpose(self.matrix)

        else:
            matrix =  node.transpose(self.matrix, in_flag, out_flag)

        return Equation(matrix, self.const, self.config)
        
    def _transpose_slice(self, node: Node):
        matrix = torch.zeros(
            (self.size,) + node.input_shape,
            dtype=node.config.PRECISION,
            device=self.config.DEVICE
        )
        slices = slice(0, self.size) + node.slices
        matrix[slices] = self.matrix

        return Equation(matrix, self.const, self.config)

    def _transpose_concat(self, node:Node):
        eqs, idx = [], 0
        for i in node.from_node:
            eqs.append(Equation(
                self.matrix[:, torch.arange(idx, idx + i.output_size)],
                self.const,
                self.config
            ))

        return eqs

    def _transpose_reshape(self, node:Node):
        matrix = self.matrix.reshape(
            (self.size,) + node.input_shape
        ).reshape(
            (self.size,) + node.output_shape
        ).reshape(self.size, -1)

        const = self.const.reshape(node.input_shape).flatten().reshape(node.output_shape).flatten()

        return Equation(matrix, self.const, self.config)
        

    def _transpose_sub(self, node:Node):
        if node.const is None:
            equation =  Equation(
                torch.hstack([self.matrix, -self.matrix]), self.const, self.config
            )
        else:
            equation = Equation(
                self.matrix.clone(), node.const.flatten() + self.const, self.config
            )

        return equation
 
    def _transpose_add(self, node:Node):
        if node.const is None:
            equation =  Equation(
                torch.hstack([self.matrix, self.matrix]), self.const, self.config
            )
        else:
            equation = Equation(
                self.matrix.clone(), node.const.flatten() + self.const, self.config
            )

        return equation

    def interval_transpose(
            self, 
            node: Node,
            bound: str,
            in_flag: torch.Tensor=None,
            out_flag: torch.Tensor=None,
            slopes: torch.tensor=None
    ):
        if node.has_relu_activation():
            return [
                self._relu_transpose(
                    node, bound, in_flag, out_flag, slopes
                )
            ]

        elif isinstance(node, BatchNormalization):
            return [
                self._batch_normalization_transpose(
                    node, bound, in_flag, out_flag
                )
            ]

        elif isinstance(node, MaxPool):
            return [
                self._maxpool_transpose(node, bound)
            ]

        else:
            raise NotImplementedError(f'Interval forward for node {type(node)}')

    def _relu_transpose(
            self,
            node: None,
            bound: str,
            out_flag: torch.tensor=None,
            in_flag: torch.tensor=None,
            slopes: torch.tensor = None
    ):
        lower_slope = self.get_relu_slope(node, 'lower', bound, out_flag, slopes)
        upper_slope = self.get_relu_slope(node, 'upper', bound, out_flag, slopes)
        out_flag = None if out_flag is None else out_flag.flatten()
        lower_const = Equation.derive_const(node, out_flag)
        upper_const = lower_const.detach().clone()
        lower_const = self.get_relu_const(
            node, lower_const, 'lower', lower_slope, out_flag
        )
        upper_const =  self.get_relu_const(
             node, upper_const, 'upper', upper_slope, out_flag
         )

        _plus, _minus = self.get_plus_matrix(), self.get_minus_matrix()

        if bound == 'lower':
            plus, minus = _plus * lower_slope, _minus * upper_slope
            const = _plus @ lower_const + _minus @ upper_const

        elif bound == 'upper':
            plus, minus = _plus  * upper_slope, _minus * lower_slope
            const = _plus @ upper_const + _minus @ lower_const

        else:
            raise ValueError(f'Bound {bound} is not recognised.')

        if out_flag is None:
            shape = (self.size,) + node.output_shape_no_batch()
        else:
            shape = self.matrix.shape

        print(node)
        input()
        matrix = node.transpose(plus.reshape(shape), in_flag, out_flag) + \
            node.transpose(minus.reshape(shape), in_flag, out_flag)

        const += self.const

        return Equation(matrix, const, self.config)


    

    def forward(self, node: Node, bound=None, slopes=None):
        if isinstance(node, Gemm):
            equation =  self._forward_gemm(node)

        elif isinstance(node, Conv):
            equation = self._forward_conv(node)

        elif isinstance(node, MatMul):
            equation = self._forward_matmul(node)

        elif isinstance(node, Slice):
            equation = self._forward_slice(node)

        elif isinstance(node, Relu):
            equation = self._forward_relu(node, bound, slopes)

        elif isinstance(node, MaxPool):
            equation = self._forward_maxpool(node, bound)

        elif type(node) in [Flatten, Reshape, Unsqueeze]:
            equation = self

        else:
            raise NotImplementedError(f'Equation forward for {type(node)}')

        return equation


    def _forward_gemm(self, node: Node):
        matrix = node.forward(self.matrix, add_bias=False)
        const = node.forward(self.const, add_bias=True)

        return Equation(matrix, const, self.config)

    def _forward_conv(self, node: Node):
        shape = (self.coeffs_size,) + node.input_shape_no_batch()
        matrix = node.forward(self.matrix.T.reshape(shape), add_bias=False)
        matrix = matrix.reshape(self.coeffs_size, -1).T
        const = node.forward(
            self.const.reshape(node.input_shape), add_bias=True
        ).flatten()

        return Equation(matrix, const, self.config)

    def _forward_matmul(self, node: Node):
        matrix = node.forward(self.matrix)
        const = node.forward(self.const)

        return Equation(matrix, const, self.config)

    def _forward_slice(self, node: Node):
        shape = (self.size,) + node.input_shape_no_batch()
        slices = [slice(0, self.size)] + node.slices
        matrix = self.matrix.reshape(shape)[slices]
        const = self.const.reshape(node.input_shape_no_batch())[node.slices]

        return Equation(matrix, const, self.config)

    def _forward_relu(self, node: Node, bound: str, slopes: torch.Tensor=None):
        slope = self.get_relu_slope(node.from_node[0], bound, bound, None, slopes)
        matrix = (self.matrix.T * slope).T
        const = self.get_relu_const(
            node.from_node[0], self.const, bound, slope
        )

        return Equation(matrix, const, self.config)

    def _forward_maxpool(self, node: Node, bound: str) -> Equation:
        lower, indices = node.forward(
            node.from_node[0].bounds.lower, return_indices=True
        )
        idx_correction = torch.tensor(
            [
                i * node.from_node[0].in_ch_sz() 
                for i in range(node.from_node[0].in_ch())
            ],
            dtype=torch.long,
            device=self.config.DEVICE    
        ).reshape((node.from_node[0].in_ch(), 1, 1))
        if node.has_batch_dimension():
            idx_correction = idx_correction[None, :]
        indices = indices + idx_correction

        lower, indices = lower.flatten(), indices.flatten()
        upper = node.forward(node.from_node[0].bounds.upper).flatten()
        lower_max  = lower > upper
        not_lower_max = torch.logical_not(lower_max)

        matrix = torch.zeros(
            (node.output_size, node.input_size),
            dtype=node.config.PRECISION, 
            device=node.config.DEVICE
        )
        const = torch.zeros(
            node.output_size, dtype=node.config.PRECISION, device=node.config.DEVICE
        )

        if bound == 'lower':
            matrix = self.matrix[indices, :]
            const = self.const[indices]

        elif bound == 'upper':
            matrix[lower_max, :] = self.matrix[indices, :][lower_max, :]
            const[lower_max] = self.const[indices][lower_max]
            const[not_lower_max] = \
                node.from_node[0].bounds.upper.flatten()[indices][not_lower_max]

        return Equation(matrix, const, node.config)


    def _get_flags(self, node: Node):
        stab = node.from_node[0].get_propagation_count()
        if stab / node.input_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            in_flag = None
        else:
            in_flag = node.from_node[0].get_propagation_flag()

        stab = node.get_propagation_count()
        if stab / node.output_size >= self.config.SIP.STABILITY_FLAG_THRESHOLD:
            out_flag = None
        else:
            out_flag = node.get_propagation_flag()

        return in_flag, out_flag

    def get_relu_relaxation(self, node: Node, bound:str, out_flag: torch.tensor, slopes: torch.tensor) -> tuple:
        if out_flag is None:
            lower_slope = torch.ones(
                node.output_size,
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )

            lower, upper = node.bounds.lower.flatten(), node.bounds.upper.flatten()

            lower_slope[node.to_node[0].get_inactive_flag().flatten()] = 0.0
            idxs = node.to_node[0].get_unstable_flag().flatten()

            if slopes is None:
                slopes = node.to_node[0].get_lower_relaxation_slope()
                lower_slope[idxs] = slopes[0] if bound == 'lower' else slopes[1]
            else:
                lower_slope[idxs] = slopes

            upper_slope = torch.zeros(
                node.output_size,
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            idxs = node.to_node[0].get_unstable_flag().flatten()
            lower, upper = lower[idxs], upper[idxs]
            upper_slope[idxs] =  upper / (upper - lower)
            upper_slope[node.to_node[0].get_active_flag().flatten()] = 1

            lower_const = Equation.derive_const(node)
            upper_const = lower_const.detach().clone()

            lower_const *= lower_slope
            upper_const *= upper_slope

            upper_const[idxs]  -= upper_slope[idxs] *  lower

        else:
            lower_slope = torch.ones(
                node.to_node[0].get_propagation_count(),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
       
            upper = node.bounds.upper[out_flag].flatten()
            lower = node.bounds.lower[out_flag].flatten()
            if node.to_node[0].lower_relaxation_slope is not None:
                idxs = abs(lower) >=  upper
                lower_slope[idxs] = 0.0
            else:
                idxs = lower < 0 
                bf_slope = node.to_node[0].lower_relaxation_slope
                lower_slope[idxs] = bf_slope[0] if bound == 'lower' else bf_slope[1]

            upper_slope = torch.ones(
                node.to_node[0].get_propagation_count(),
                dtype=self.config.PRECISION,
                device=self.config.DEVICE
            )
            idxs = lower < 0
            upper_slope[idxs] = upper[idxs] /  (upper[idxs] - lower[idxs])

            lower_const = Equation.derive_const(
                node, out_flag.flatten()
            )
            upper_const = lower_const.detach().clone()
            lower_const *= lower_slope
            upper_const *= upper_slope
            upper_const[idxs]  -= upper_slope[idxs] *  lower[idxs]

        return lower_slope, upper_slope, lower_const, upper_const


""" 
