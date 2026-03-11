from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
import torch
from torch import Tensor

from src.model.diagonal_gaussian import DiagonalGaussian
from .flow import Flow, FlowCfg


@dataclass
class RectifiedFlowCfg(FlowCfg):
    name: Literal["rectified"]


class RectifiedFlow(Flow[RectifiedFlowCfg]):
    def a(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define data weight a_t for t"""
        return 1-t

    def a_prime(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of data weight a_t for t"""
        return torch.full(
            (1,), fill_value=-1.0, device=t.device, dtype=t.dtype
        ).expand_as(t)

    def b(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define noise weight b_t for t"""
        return t

    def b_prime(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of noise weight b_t for t"""
        return torch.ones(
            (1,), device=t.device, dtype=t.dtype
        ).expand_as(t)
    
    def get_eps_from_ut_and_x(
        self,
        ut: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return ut + x

    def get_ut_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return eps - x      

    def get_ut_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        # a_prime / a * zt - (a_prime * b / a - b_prime) * eps
        return 1 / self.a(t) * (eps - zt)
    
    def get_x_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return eps - ut

    def get_x_from_ut_and_zt(
        self,
        ut: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
         return zt - (1 - self.a(t)) * ut

    def conditional_p_ut(
        self,
        mean_theta: Float[Tensor, "*batch"],
        z_t: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        temperature: float | int = 1,
        sigma_theta: Float[Tensor, "*#batch"] | None = None,
        v_theta: Float[Tensor, "*#batch"] | None = None
    ) -> DiagonalGaussian:
        # mean_theta and sigma_theta are the mean and standard deviation of ut
        gamma_p = self.gamma(t, t_star, alpha)
        a_t, a_t_star = self.a(t), self.a(t_star)
        mean = a_t_star * (z_t - t*mean_theta) \
            + (z_t + a_t*mean_theta) * gamma_p
        var = self.sigma(t, t_star, alpha, v_theta) ** 2
        if sigma_theta is not None:
            var = var + ((a_t_star*t-a_t*gamma_p) * sigma_theta) ** 2
        if temperature != 1:
            var = temperature ** 2 * var
        return DiagonalGaussian(mean, var=var)
