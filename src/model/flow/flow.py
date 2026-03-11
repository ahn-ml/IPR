from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
import torch
from torch import Tensor
from torch.nn import Module

from src.type_extensions import Parameterization
from ..diagonal_gaussian import DiagonalGaussian


@dataclass
class FlowCfg:
    name: str
    variance: Literal["fixed_small", "fixed_large", "learned_range"] = "fixed_small"


T = TypeVar("T", bound=FlowCfg)


class Flow(Module, ABC, Generic[T]):
    cfg: T

    def __init__(
        self,
        cfg: T,
        parameterization: Parameterization = "ut"
    ) -> None:
        super(Flow, self).__init__()
        self.cfg = cfg
        self.parameterization = parameterization
        if parameterization == "eps":
            self.conditional_p = self.conditional_p_eps
        elif parameterization == "ut":
            self.conditional_p = self.conditional_p_ut
        else:
            raise ValueError(f"Unknown parameterization {parameterization}")

    @abstractmethod
    def a(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define data weight a_t for t"""
        pass
    
    @abstractmethod
    def a_prime(
        self,
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of data weight a_t for t"""
        pass

    @abstractmethod
    def b(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define noise weight b_t for t"""
        pass
    
    @abstractmethod
    def b_prime(
        self, 
        t: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        """Define derivative of noise weight b_t for t"""
        pass

    def sigma_small(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes the lower bound of the variance for the approximate reverse distribution p"""
        b_t_star = self.b(t_star)
        sigma = alpha * b_t_star * torch.sqrt(
            1-(self.a(t)*b_t_star / (self.a(t_star)*self.b(t)))**2
        )
        sigma[(t == 0).logical_or_(t_star == 1)] = 0
        return sigma

    def sigma_large(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes the upper bound of the variance for the approximate reverse distribution p"""
        b_t = self.b(t)
        sigma = alpha * b_t * torch.sqrt(
            1-(self.a(t)*self.b(t_star) / (self.a(t_star)*b_t))**2
        )
        sigma[(t == 0).logical_or_(t_star == 1)] = 0
        return sigma        

    def sigma(
        self,
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
        v_theta: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        """
        Computes the variance for the approximate reverse distribution p
        NOTE assumes the normalization of v_theta to be done already
        """
        if self.cfg.variance == "fixed_small":
            return self.sigma_small(t, t_star, alpha)
        elif self.cfg.variance == "fixed_large":
            return self.sigma_large(t, t_star, alpha)
        else:
            assert v_theta is not None
            sigma_large = self.sigma_large(t, t_star, alpha).expand_as(v_theta)
            sigma_small = self.sigma_small(t, t_star, alpha).expand_as(v_theta)
            sigma = torch.zeros_like(v_theta)
            mask = (sigma_large != 0).logical_and_(sigma_small != 0).expand_as(v_theta)
            v_theta = v_theta[mask]
            # this can change the dtype in AMP
            nnz_sigma = torch.exp(
                v_theta * torch.log(sigma_large[mask]) \
                + (1 - v_theta) * torch.log(sigma_small[mask])
            )
            sigma = torch.zeros(mask.shape, dtype=nnz_sigma.dtype, device=nnz_sigma.device)
            sigma[mask] = nnz_sigma
            return sigma

    def gamma(
        self,
        t: Float[Tensor, "*batch"],
        t_star: Float[Tensor, "*batch"],
        alpha: float | int
    ) -> Float[Tensor, "*batch"]:
        """Computes sqrt(b_{t*}^2-sigma_small(t,t*)^2)"""
        b_t_star = self.b(t_star)
        gamma = b_t_star * torch.sqrt(
            1-alpha**2*(1-(self.a(t)*b_t_star / (self.a(t_star)*self.b(t)))**2)
        )
        gamma[t == 0] = 0
        ones_mask = t_star == 1
        gamma[ones_mask] = b_t_star[ones_mask]
        return gamma
    
    @staticmethod
    def sample_eps(
        x: Float[Tensor, "*batch"]
    ) -> Float[Tensor, "*batch"]:
        return torch.randn_like(x)
    
    def get_eps_from_ut_and_x(
        self,
        ut: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_eps_from_ut_and_zt(
        self,
        u_t: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()
    
    def get_eps_from_x_and_zt(
        self,
        x: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        assert torch.all(t > 0)
        return (zt - self.a(t) * x) / self.b(t)

    def get_eps(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if eps is not None:
            return eps
        assert sum(a is not None for a in (ut, x, zt)) == 2
        if ut is not None:
            if x is not None:
                return self.get_eps_from_ut_and_x(ut, x, t)
            return self.get_eps_from_ut_and_zt(ut, zt, t)
        return self.get_eps_from_x_and_zt(x, zt, t)

    def get_zt_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_zt_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        # NOTE equivalent to q_sample in OpenAI's diffusion implementation
        return self.a(t) * x + self.b(t) * eps
    
    def get_zt_from_ut_and_x(
        self,
        ut: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_zt(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if zt is not None:
            return zt
        assert sum(a is not None for a in (eps, ut, x)) == 2
        if eps is not None:
            if ut is not None:
                return self.get_zt_from_eps_and_ut(eps, ut, t)
            return self.get_zt_from_eps_and_x(eps, x, t)
        return self.get_zt_from_ut_and_x(ut, x, t)

    def get_ut_from_eps_and_x(
        self,
        eps: Float[Tensor, "*batch"],
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        return self.a_prime(t) * x + self.b_prime(t) * eps

    def get_ut_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        a_ratio = self.a_prime(t) / self.a(t)
        return a_ratio * zt - (a_ratio * self.b(t) - self.b_prime(t)) * eps
    
    def get_ut_from_x_and_zt(
        self,
        x: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_ut(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if ut is not None:
            return ut
        assert sum(a is not None for a in (eps, x, zt)) == 2
        if eps is not None:
            if x is not None:
                return self.get_ut_from_eps_and_x(eps, x, t)
            return self.get_ut_from_eps_and_zt(eps, zt, t)
        return self.get_ut_from_x_and_zt(x, zt, t)

    def get_x_from_eps_and_ut(
        self,
        eps: Float[Tensor, "*batch"],
        ut: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_x_from_eps_and_zt(
        self,
        eps: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()
    
    def get_x_from_ut_and_zt(
        self,
        ut: Float[Tensor, "*batch"],
        zt: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> Float[Tensor, "*batch"]:
        raise NotImplementedError()

    def get_x(
        self,
        t: Float[Tensor, "*#batch"],
        eps: Float[Tensor, "*batch"] | None = None,
        ut: Float[Tensor, "*batch"] | None = None,
        x: Float[Tensor, "*batch"] | None = None,
        zt: Float[Tensor, "*batch"] | None = None
    ) -> Float[Tensor, "*batch"]:
        if x is not None:
            return x
        assert sum(a is not None for a in (eps, ut, zt)) == 2
        if eps is not None:
            if ut is not None:
                return self.get_x_from_eps_and_ut(eps, ut, t)
            return self.get_x_from_eps_and_zt(eps, zt, t)
        return self.get_x_from_ut_and_zt(ut, zt, t)

    def conditional_p_eps(
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
        # mean_theta and sigma_theta are the mean and standard deviation of eps
        a_t, a_t_star = self.a(t), self.a(t_star)
        assert (a_t > 0).all()
        a_ratio = a_t_star / a_t
        eps_scale = self.gamma(t, t_star, alpha) - a_ratio * self.b(t)
        mean = a_ratio * z_t + eps_scale * mean_theta
        var = self.sigma(t, t_star, alpha, v_theta) ** 2
        if sigma_theta is not None:
            var = var + (eps_scale * sigma_theta) ** 2
        if temperature != 1:
            var = temperature ** 2 * var
        return DiagonalGaussian(mean, var=var)

    @abstractmethod
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
        pass

    def marginal_q(
        self,
        x: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"]
    ) -> DiagonalGaussian:
        return DiagonalGaussian(
            mean=self.a(t) * x,
            std=self.b(t)
        )
    
    def conditional_q(
        self,
        x: Float[Tensor, "*batch"],
        eps: Float[Tensor, "*batch"],
        t: Float[Tensor, "*#batch"],
        t_star: Float[Tensor, "*#batch"],
        alpha: float | int,
    ) -> DiagonalGaussian:
        return DiagonalGaussian(
            mean=self.a(t_star) * x + eps * self.gamma(t, t_star, alpha),
            std=self.sigma_small(t, t_star, alpha)
        )
