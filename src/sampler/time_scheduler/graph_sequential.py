from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

from jaxtyping import Bool, Float
import torch
from torch import device, Tensor

from .time_scheduler import TimeScheduler, TimeSchedulerCfg


@dataclass
class GraphSequentialCfg(TimeSchedulerCfg):
    name: Literal["graph_sequential"]
    max_order: int = 2
    max_parallel_group: int = 1
    inference_overlap: float = 0.1
    certainty_decay: float = 0.5
    eps: float = 1e-5


class GraphSequential(TimeScheduler[GraphSequentialCfg]):
    def __init__(
        self,
        cfg: GraphSequentialCfg,
        max_steps: int,
        patch_size: int,
        patch_grid_shape: Sequence[int],
        dependency_matrix: Float[Tensor, "num_patches num_patches"]
    ):
        assert cfg.eps > 0
        assert cfg.inference_overlap >= 0 and cfg.inference_overlap <= 1
        assert cfg.max_parallel_group > 0
        assert cfg.max_order >= 0
        assert cfg.certainty_decay > 0, "Certainty decay can't be smaller than 0"
        assert cfg.certainty_decay < 1, "Certainty decay can't be higher than 1"
        super(GraphSequential, self).__init__(cfg, max_steps, patch_size, patch_grid_shape, dependency_matrix)
        self.weighted_adjacency_matrix = self.get_weighted_adjacency_matrix()

    def get_weighted_adjacency_matrix(self) -> Float[Tensor, "num_patches num_patches"]:
        weighted_adjacency = torch.eye(
            self.dependency_matrix.shape[0], device=self.dependency_matrix.device
        )
        dependency_power = weighted_adjacency.clone()
        weight = 1
        for _ in range(self.cfg.max_order):
            dependency_power @= self.dependency_matrix
            dependency_power.div_(dependency_power.sum(dim=1))
            weight *= self.cfg.certainty_decay
            weighted_adjacency.add_(weight * dependency_power)
        return weighted_adjacency

    def get_known_mask(
        self,
        batch_size: int,
        device: device,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
    ) -> Bool[Tensor, "batch total_patches"]:
        # Create mask for patches with true == unknown and false == known of shape [batch total_patches]
        if mask is None:
            return torch.zeros(1, dtype=torch.bool, device=device).expand(batch_size, self.total_patches)
        return (mask[..., :: self.patch_size, :: self.patch_size] < 0.5).flatten(1)

    def propagate_certainty(
        self, 
        certainty: Float[Tensor, "*batch num_patches"]
    ) -> Float[Tensor, "*batch num_patches"]:
        if certainty.sum() < self.cfg.eps or self.cfg.max_order == 0:
            return certainty.clone()
        
        certainty = certainty @ self.weighted_adjacency_matrix.to(device=certainty.device)
        certainty.div_(certainty.max())
        return certainty

    def get_scheduling_matrix(
        self,
        batch_size: int,
        device: device,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
    ) -> Tensor:
        known_mask = self.get_known_mask(batch_size, device, mask)
        knowledge = known_mask.float()
        num_blocks = torch.zeros(batch_size, dtype=torch.long, device=device)
        # Will be a list of index tensors of shape [num_block_patches 2] 
        # indicating which patches will be denoised in each block
        # number of blocks can be at most max_steps because in last step everything
        # will be enforced to directly jump to zero
        ordering_sequence = []
        not_selectable_mask = known_mask.clone()
        block_cnt = 0
        while block_cnt < self.max_steps and knowledge.min() < self.cfg.eps:
            certainty = self.propagate_certainty(knowledge)
            certainty[not_selectable_mask] = -torch.inf
            max_certainty = certainty.max(dim=1).values.unsqueeze(1) # max_certainty for each batch element
            candidates = certainty >= max_certainty - self.cfg.eps
            num_candidates = candidates.sum(dim=1)
            num_blocks.add_(num_candidates > 0)
            split_idx = torch.cumsum(num_candidates, dim=0)[:-1]
            candidate_idx = candidates.nonzero()
            # Add the block idx for later stacking
            candidate_idx = torch.cat(
                (torch.full_like(candidate_idx[:, :1], fill_value=block_cnt), candidate_idx), 
                dim=1
            )
            # Do candidate subsampling batch-wise (should be fast because of indexing only)
            batch_candidate_idx = candidate_idx.tensor_split(split_idx.cpu())
            for bci in batch_candidate_idx:
                if self.cfg.max_parallel_group < len(bci):
                    shuffled = bci[torch.randperm(len(bci), device=device)]
                    assert bci[:,1].unique().shape[0] == 1, "All patches in a block must be from the same batch element"
                    keep = shuffled[:self.cfg.max_parallel_group]
                    remove = shuffled[self.cfg.max_parallel_group:, 1:]
                    candidates[remove[:, 0], remove[:, 1]] = False
                else:
                    keep = bci
                ordering_sequence.append(keep)
            knowledge[not_selectable_mask] += 1 - self.cfg.inference_overlap
            knowledge[candidates] = 1 - self.cfg.inference_overlap
            not_selectable_mask[candidates] = True
            knowledge.clamp_max_(1)
            block_cnt += 1

        ordering_sequence = torch.cat(ordering_sequence, dim=0) # [batch*total_patches, 3]
        # The ideal length of a block for each batch element [batch]
        # Round this down to ensure that last time is actually zero
        inference_block_length = (
            self.max_steps / ((num_blocks - 1) * (1 - self.cfg.inference_overlap) + 1)
        ).floor_() # [batch]
        
        # [block_cnt, batch]
        block_starts = (torch.arange(block_cnt, device=device).unsqueeze(1) \
            * inference_block_length * (1 - self.cfg.inference_overlap)).floor_()
        start_step = torch.zeros((batch_size, self.total_patches), device=device)
        start_step[ordering_sequence[:, 1], ordering_sequence[:, 2]] \
            = block_starts[ordering_sequence[:, 0], ordering_sequence[:, 1]]
        
        unshifted_base = torch.linspace(                                        # [max_steps]
            0, 1 - self.max_steps, self.max_steps, device=device
        )
        unshifted_base = unshifted_base.unsqueeze(1) + inference_block_length   # [max_steps, batch]
        scheduling_matrix = unshifted_base.unsqueeze(-1) + start_step           # [max_steps, batch, total_patches]
        scheduling_matrix.div_(inference_block_length.unsqueeze(1))
        scheduling_matrix.clamp_(0, 1)
        scheduling_matrix[:, known_mask] = 0
        scheduling_matrix[-1] = 0
        return scheduling_matrix

    def __call__(
        self,
        batch_size: int,
        device: device,
        mask: Float[Tensor, "batch 1 height width"] | None = None, 
    ) -> Iterator[
        tuple[
            Float[Tensor, "batch 1 height width"],
            Float[Tensor, "batch 1 height width"]
        ]
    ]:
        scheduling_matrix = self.get_scheduling_matrix(batch_size, device, mask)
        kernel = torch.ones(2 * (self.patch_size,), device=device)            

        for t_id, current_t in enumerate(
            scheduling_matrix[:-1]
        ):  # [steps, batch, frame]
            next_t = scheduling_matrix[t_id + 1]

            current_t_img = current_t.reshape(-1, 1, *self.patch_grid_shape)
            next_t_img = next_t.reshape(-1, 1, *self.patch_grid_shape)

            kron_t = torch.kron(current_t_img, kernel)
            kron_next_t = torch.kron(next_t_img, kernel)

            yield kron_t, kron_next_t
