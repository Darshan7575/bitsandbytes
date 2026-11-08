# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from math import ceil, prod
import warnings

import torch

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import MockArgs, Optimizer8bit as BaseOptimizer8bit


class OptimizerQuantized(BaseOptimizer8bit):
    def __init__(self, params, defaults, optim_bits=4, is_paged=False):
        """
        Base quantized optimizer class.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            optim_bits (`int`, defaults to 4):
                The number of bits of the optimizer state. It should be one of [2, 4, 8] bits.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """

        # hardcoded `optim_bits` to -1 to prevent the base class from creating qmaps. For ReAdam, the qmap is baked into the kernel.
        super().__init__(params, defaults, optim_bits=-1, is_paged=is_paged)

        # over-riding the keys, since many of them are irrelevant for ReAdam.
        self.non_castable_tensor_keys = {
            "qmap2",  # incase, we have 1D params, we use 8 bit quantized state2
            "max2",
            "new_max2",
            "state1",  # buffer for storing U_t
            "state2",  # fallback buffer for storing full S_t for 1D params where adafactor factorization is not possible
            "state2_row",  # buffer for row sum of adafactor compressed second moment
            "state2_col",  # buffer for column sum of adafactor compressed second moment
            "absmax2",
            "unorm_vec",  # update norm scales
        }

        self.fill_qmap()

    def fill_qmap(self):
        self.name2qmap["udynamic"] = F.create_dynamic_map(signed=False)

    def get_config(self, gindex, pindex, group):
        config = {}
        config["betas"] = group["betas"]
        config["eps"] = group["eps"]
        config["weight_decay"] = group["weight_decay"]
        config["lr"] = group["lr"]
        config["optim_bits"] = self.args.optim_bits
        config["decay_ratio"] = self.args.decay_ratio
        config["min_quant_size"] = self.args.min_quant_size
        config["max_unorm"] = self.args.max_unorm
        config["skip_zeros"] = self.args.skip_zeros

        if (gindex, pindex) in self.mng.index2config:
            config.update(self.mng.index2config[(gindex, pindex)])

        # Also check pid2config as a fallback so that override_config works
        # regardless of whether it was called before or after register_parameters.
        p = self.param_groups[gindex]["params"][pindex]
        if id(p) in self.mng.pid2config:
            config.update(self.mng.pid2config[id(p)])

        return config

    def get_state_buffer(self, p, dtype=torch.float32, optim_bits=8, factor=None):
        shape = list(p.shape)
        if factor is None:
            if dtype == torch.uint8:
                # For state1, when we use 2/4 bits, we need to pack them into 8 bits (uint8). The size of unified memory needs to be updated accordingly.
                packed_factor = 8 // optim_bits
                shape[-1] = ceil(shape[-1] / packed_factor)
        elif factor == "row":
            shape[-1] = 1
        elif factor == "col":
            shape[-2] = 1
        else:
            raise ValueError(f"Invalid factor value: {factor}")

        if p.device.type == "cpu":
            if self.is_paged and not getattr(self, "_cpu_paged_warned", False):
                warnings.warn(
                    "Paged optimizers are not supported on CPU. Falling back to non-paged optimizer behavior.",
                    stacklevel=2,
                )
                self._cpu_paged_warned = True
            return torch.zeros(shape, dtype=dtype, device=p.device)
        if not self.is_paged or prod(shape) < 1e5:
            return torch.zeros(shape, dtype=dtype, device=p.device)
        else:
            # > 1 MB
            buff = F.get_paged(*shape, dtype=dtype, device=p.device)
            F.fill(buff, 0)
            self.page_mng.paged_tensors.append(buff)
            return buff

    def prefetch_state(self, p):
        if self.is_paged:
            state = self.state[p]
            s1 = state["state1"]
            is_paged = getattr(s1, "is_paged", False)
            if is_paged:
                F.prefetch_tensor(state["state1"])
                if "state2" in state:
                    F.prefetch_tensor(state["state2"])
                else:
                    F.prefetch_tensor(state["state2_row"])
                    F.prefetch_tensor(state["state2_col"])


class Optimizer2State(OptimizerQuantized):
    def __init__(
        self,
        optimizer_name,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        optim_bits=4,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
        max_unorm=0.0,
        skip_zeros=False,
        is_paged=False,
    ):
        """
        Base 2-state update optimizer class for ReAdam.

        Arguments:
            optimizer_name (`str`):
                The name of the optimizer.
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple`, defaults to (0.9, 0.999)):
                The beta values for the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value for the optimizer.
            weight_decay (`float`, defaults to 0.0):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 4):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
            max_unorm (`float`, defaults to 0.0):
                The maximum value to normalize each block with.
            skip_zeros (`bool`, defaults to `False`):
                Whether to skip zero values for sparse gradients and models to ensure correct updates.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.

        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if isinstance(betas, str):
            # format: '(beta1, beta2)'
            betas = betas.replace("(", "").replace(")", "").strip().split(",")
            betas = [float(b) for b in betas]
        for i in range(len(betas)):
            if not 0.0 <= betas[i] < 1.0:
                raise ValueError(f"Invalid beta parameter at index {i}: {betas[i]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults, optim_bits, is_paged)

        if args is None:
            args = {}
            args["optim_bits"] = optim_bits
            args["decay_ratio"] = decay_ratio
            args["min_quant_size"] = min_quant_size
            args["max_unorm"] = max_unorm
            args["skip_zeros"] = skip_zeros

            self.args = MockArgs(args)
        else:
            self.args = args

        self.optimizer_name = optimizer_name

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config["optim_bits"] not in [2, 4, 8]:
            raise NotImplementedError(f"Amount of optimizer bits not supported: {config['optim_bits']}")

        state = self.state[p]
        state["step"] = 0

        if p.numel() >= config["min_quant_size"]:
            state["state1"] = self.get_state_buffer(p, dtype=torch.uint8, optim_bits=config["optim_bits"])

            if p.dim() > 1:
                # do adafactor compression only if the param is not 1D and has size greater than the minimum size required for quantization
                # TODO (darshan): In theory, these row/column stats can be quantized, but the memory gain is very marginal with additional compute overhead, and is not worth the effort.
                state["state2_row"] = self.get_state_buffer(p, dtype=torch.float32, factor="row")
                state["state2_col"] = self.get_state_buffer(p, dtype=torch.float32, factor="col")
            else:
                import pdb

                pdb.set_trace()
                if state["step"] == 0:
                    if "udynamic" not in self.name2qmap:
                        self.fill_qmap()
                    self.name2qmap["udynamic"] = self.name2qmap["udynamic"].to(p.device)

                # here, we use 8 bit DTQ quantization from bnb for the quantization of state2
                state["state2"] = self.get_state_buffer(p, dtype=torch.uint8, optim_bits=8)
                state["qmap2"] = self.name2qmap["udynamic"]

                blocksize = 256
                n = p.numel()
                blocks = (n // blocksize) + bool(n % blocksize)

                state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

        else:
            state["state1"] = self.get_state_buffer(p, dtype=torch.float32)
            state["state2"] = self.get_state_buffer(p, dtype=torch.float32)

        if config["max_unorm"] > 0.0:
            state["unorm_vec"] = torch.zeros((1,), device=p.device)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        # avoid update error from non-contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        # import pdb; pdb.set_trace()

        if state["state1"].dtype == torch.float:
            # as the size of parameter is smaller than `min_quant_size`, we use full-precision adam here.
            F.optimizer_update_32bit(
                "adam",
                grad,
                p,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                state["state2"],
                config["betas"][1],
                0.0,  # beta3 = 0.0
                0.0,  # alpha = 0.0
                config["weight_decay"],
                1.0,  # gnorm_scale = 1.0
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8:
            if "state2" in state:
                F.optimizer_update_readam_1D(
                    grad,
                    p,
                    state["state1"],
                    state["state2"],
                    config["optim_bits"],
                    config["betas"][0],
                    config["betas"][1],
                    config["eps"],
                    step,
                    config["lr"],
                    state["qmap2"],
                    state["absmax2"],
                    config["weight_decay"],
                    config["decay_ratio"],
                    gnorm_scale=1.0,
                    skip_zeros=config["skip_zeros"],
                )
            elif "state2_row" in state:
                F.optimizer_update_readam(
                    grad,
                    p,
                    state["state1"],
                    state["state2_row"],
                    state["state2_col"],
                    config["optim_bits"],
                    config["betas"][0],
                    config["betas"][1],
                    config["eps"],
                    step,
                    config["lr"],
                    config["weight_decay"],
                    config["decay_ratio"],
                    gnorm_scale=1.0,
                    skip_zeros=config["skip_zeros"],
                )


class ReAdamW(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=4,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
        is_paged=False,
    ):
        """
        Base ReAdamW optimizer.

        NOTE: We do not implement the full precision ReAdamW as it is nothing but AdamW functioning with 32-bit precision.
              Use `bnb.optim.AdamW` with `optim_bits=32` instead.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 4):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits,
            decay_ratio,
            args,
            min_quant_size,
            is_paged=is_paged,
        )


class ReAdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=8,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
        is_paged=False,
    ):
        """
        8-bit ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 8):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
                Note: This parameter is not used in ReAdamW8bit as it always uses 8-bit quantization.
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        if optim_bits != 8:
            # any other value is invalid since ReAdamW8bit always uses 8-bit quantization
            raise ValueError("ReAdamW8bit only supports optim_bits=8 (default value for compatibility)")

        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,  # Hardcoded to 8 bits
            decay_ratio,
            args,
            min_quant_size,
            is_paged=is_paged,
        )


class ReAdamW4bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=4,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
        is_paged=False,
    ):
        """
        4-bit ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 4):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
                Note: This parameter is not used in ReAdamW8bit as it always uses 8-bit quantization.
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        if optim_bits != 4:
            # any other value is invalid since ReAdamW8bit always uses 4-bit quantization
            raise ValueError("ReAdamW4bit only supports optim_bits=4 (default value for compatibility)")

        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            4,  # Hardcoded to 4 bits
            decay_ratio,
            args,
            min_quant_size,
            is_paged=is_paged,
        )


class ReAdamW2bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=2,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
        is_paged=False,
    ):
        """
        2-bit ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 2):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
                Note: This parameter is not used in ReAdamW8bit as it always uses 8-bit quantization.
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
            is_paged (`bool`, defaults to `False`):
                Whether the optimizer is a paged optimizer or not.
        """
        if optim_bits != 2:
            # any other value is invalid since ReAdamW2bit always uses 2-bit quantization
            raise ValueError("ReAdamW2bit only supports optim_bits=2 (default value for compatibility)")

        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            2,  # Hardcoded to 2 bits
            decay_ratio,
            args,
            min_quant_size,
            is_paged=is_paged,
        )


class PagedReAdamW(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=4,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
    ):
        """
        Paged ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 4):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
        """
        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            optim_bits,
            min_quant_size,
            args,
            min_quant_size,
            is_paged=True,
        )


class PagedReAdamW8bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=8,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
    ):
        """
        Paged 8-bit ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 8):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
                Note: This parameter is not used in ReAdamW8bit as it always uses 8-bit quantization.
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
        """
        if optim_bits != 8:
            # any other value is invalid since ReAdamW8bit always uses 8-bit quantization
            raise ValueError("ReAdamW8bit only supports optim_bits=8 (default value for compatibility)")

        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            8,  # Hardcoded to 8 bits
            decay_ratio,
            args,
            min_quant_size,
            is_paged=True,
        )


class PagedReAdamW4bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=4,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
    ):
        """
        Paged 4-bit ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 4):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
                Note: This parameter is not used in ReAdamW8bit as it always uses 8-bit quantization.
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
        """
        if optim_bits != 4:
            # any other value is invalid since ReAdamW8bit always uses 4-bit quantization
            raise ValueError("ReAdamW4bit only supports optim_bits=4 (default value for compatibility)")

        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            4,  # Hardcoded to 4 bits
            decay_ratio,
            args,
            min_quant_size,
            is_paged=True,
        )


class PagedReAdamW2bit(Optimizer2State):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        optim_bits=2,
        decay_ratio=10.0,
        args=None,
        min_quant_size=4096,
    ):
        """
        Paged 2-bit ReAdamW optimizer.

        Arguments:
            params (`torch.Tensor`):
                The input parameters to optimize.
            lr (`float`, defaults to 1e-3):
                The learning rate.
            betas (`tuple(float, float)`, defaults to (0.9, 0.999)):
                The beta values are the decay rates of the first and second-order moment of the optimizer.
            eps (`float`, defaults to 1e-8):
                The epsilon value prevents division by zero in the optimizer.
            weight_decay (`float`, defaults to 1e-2):
                The weight decay value for the optimizer.
            optim_bits (`int`, defaults to 2):
                The number of bits of the optimizer state. Should be one of [2, 4, 8].
                Note: This parameter is not used in ReAdamW8bit as it always uses 8-bit quantization.
            decay_ratio (`float`, defaults to 10.0):
                The threshold for the ratio between `U_t` and `Z_t` beyond which a co-ordinate is considered to be in the decay regime.
            args (`object`, defaults to `None`):
                An object with additional arguments.
            min_quant_size (`int`, defaults to 4096):
                The minimum number of elements of the parameter tensors for 2/4/8-bit optimization.
        """
        if optim_bits != 2:
            # any other value is invalid since ReAdamW8bit always uses 2-bit quantization
            raise ValueError("ReAdamW4bit only supports optim_bits=2 (default value for compatibility)")

        super().__init__(
            "readam",
            params,
            lr,
            betas,
            eps,
            weight_decay,
            2,  # Hardcoded to 2 bits
            decay_ratio,
            args,
            min_quant_size,
            is_paged=True,
        )
