import gc
import random
from dataclasses import dataclass, field
from math import ceil, prod
from typing import List, Literal, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from jaxtyping import Bool, Float, Int32, Int64
from torch import Tensor
from torch.nn.functional import avg_pool2d, interpolate
from tqdm import tqdm
from pytorch_lightning import seed_everything

from src.misc.mnist_classifier import MNISTClassifier, get_classifier
from src.type_extensions import SamplingOutput
from ..model import Wrapper
from .sampler import Sampler, SamplerCfg

from srmbench.evaluations import EvenPixelsEvaluation, CountingObjectsEvaluation, MnistSudokuEvaluation


@dataclass
class IPRSamplerCfg(SamplerCfg):
    name: Literal["ipr_sampler"]
    max_batch_size: int = 16
    task_name: str = "mnist_sudoku"
    mode: str = "standard"
    num_corruption_swaps: int = 3

    # IPR config
    immediate_stop: bool = False
    max_ipr_budget: int = 100
    resampling_ratio: float = 0.25
    noise_type: str = "random"
    ipr_overlap: float = 0
    ipr_steps_per_patch: int = 10
    visualize: bool = False

    # SRM config
    patch_order: str = "greedy"
    overlap: float = 0
    epsilon: float = 1e-6
    top_k: int = 1
    use_ema: bool = True
    steps_per_patch: float = 3
    stochasticity: float = 1.0
    temperature: float = 1.0


@dataclass
class TreeNode:
    z_t: Float[Tensor, "1 d_data height width"]
    scheduling_matrix: Float[Tensor, "max_steps_plus_1 1 total_patches"]
    is_unknown_map: Bool[Tensor, "1 total_patches"]
    block_starts: Float[Tensor, "1 max_blocks"]
    block_counters: Int32[Tensor, "1"]
    step_targets: Int32[Tensor, "1"]
    depth: int = 0
    parent: Optional['TreeNode'] = None
    children: List['TreeNode'] = field(default_factory=list)
    patch_id: Optional[int] = None
    step_id: int = 0

    def clone(self) -> 'TreeNode':
        return TreeNode(
            parent=self.parent,
            z_t=torch.empty_like(self.z_t).copy_(self.z_t),
            scheduling_matrix=torch.empty_like(self.scheduling_matrix).copy_(self.scheduling_matrix),
            is_unknown_map=torch.empty_like(self.is_unknown_map).copy_(self.is_unknown_map),
            block_starts=torch.empty_like(self.block_starts).copy_(self.block_starts),
            block_counters=torch.empty_like(self.block_counters).copy_(self.block_counters),
            step_targets=torch.empty_like(self.step_targets).copy_(self.step_targets),
            patch_id=self.patch_id,
            step_id=self.step_id,
            depth=self.depth,
        )


class IPR:
    def __init__(self, sampler: 'IPRSampler'):
        self.sampler = sampler
        self.cfg = sampler.cfg
        self.device = None
        self.mnist_classifier = None
        self.task_name = self.cfg.task_name
        self.evaluator = None

    def load_mnist_classifier(self):
        if self.mnist_classifier is None and self.device is not None:
            self.mnist_classifier = get_classifier(
                model_path="datasets/mnist_sudoku/mnist_classifier.pth",
                device=self.device,
            )

    @torch.no_grad()
    def solve(
        self,
        root_node,
        model,
        label,
        c_cat,
        eps,
        mask,
        masked,
        prototypes,
        device: torch.device,
    ):
        if self.task_name == "mnist_sudoku" and self.mnist_classifier is None:
            self.device = device
            self.load_mnist_classifier()
        elif self.task_name == "even_pixels" and self.evaluator is None:
            self.device = device
            self.evaluator = EvenPixelsEvaluation()
        elif self.task_name == "counting_objects" and self.evaluator is None:
            self.device = device
            self.evaluator = CountingObjectsEvaluation(
                object_variant="polygons",
                device=self.device
            )

        if mask is not None:
            mask_sequence = self.sampler.full_mask_to_sequence_mask(mask)
        else:
            mask_sequence = None

        initial_noise = root_node.z_t.clone()

        # Initial greedy denoise
        if self.cfg.mode == "standard":
            node = self._greedy(
                root_node, model, label, c_cat, eps, mask, masked, prototypes, device
            )
        else:
            node = root_node

        with torch.no_grad():
            is_valid = self._check(node.z_t.to(self.device), node.is_unknown_map.to(self.device))[0]

        if is_valid:
            print("[SRM] Found valid image")
            if self.cfg.immediate_stop:
                return node

        # Evaluate initial result
        self.sampler.evaluate_and_log(node.z_t.to(self.device), budget=0)

        # Collect snapshots for visualization
        snapshots = []
        if self.cfg.visualize:
            snapshots.append(("Iter 0", node.z_t.cpu().clone()))

        # Build patch list (all patches, or only generated-area patches if masked)
        total_patches = prod(self.sampler.patch_grid_shape)
        all_patches = list(range(total_patches))
        if mask_sequence is not None:
            all_patches = [p for p in all_patches if mask_sequence[0, p] > 0.5]

        # IPR loop: randomly select patches to reset
        for iteration in range(self.cfg.max_ipr_budget):
            num_patches_to_reset = max(1, int(len(all_patches) * self.cfg.resampling_ratio))
            patches_to_reset = random.sample(all_patches, num_patches_to_reset)

            print(f"[IPR] Iteration {iteration}: Resetting {len(patches_to_reset)} patches (random)")

            # Create new node with patches reset
            new_node, new_prototypes = self._create_reset_node(
                node=node,
                patches_to_reset=patches_to_reset,
                mask=mask,
                masked=masked,
                device=device,
                initial_noise=initial_noise
            )

            # Re-denoise
            node = self._greedy(
                new_node, model, label, c_cat, eps, mask, masked, new_prototypes, device
            )

            # Evaluate and log
            self.sampler.evaluate_and_log(node.z_t.to(self.device), budget=iteration + 1)

            with torch.no_grad():
                is_valid = self._check(node.z_t.to(self.device), node.is_unknown_map.to(self.device))[0]

            if self.cfg.visualize:
                snapshots.append((f"Iter {iteration + 1}", node.z_t.cpu().clone()))

            gc.collect()
            torch.cuda.empty_cache()

            if is_valid:
                print("[IPR] Found valid image")
                if self.cfg.immediate_stop:
                    break

        self.sampler.log_all_budgets()

        if self.cfg.visualize and snapshots:
            self._visualize_ipr_process(snapshots, save_path="ipr_process.png")

        return node

    def _find_invalid_sudoku_cells(self, z_t: Tensor) -> np.ndarray:
        """Return a 9x9 boolean mask where True = cell involved in a constraint violation."""
        from src.global_cfg import get_mnist_classifier_path
        classifier = get_classifier(get_mnist_classifier_path(), z_t.device)
        # Discretize: split into 9x9 tiles and classify
        sample = z_t[:1]  # [1, 1, H, W]
        grid_size = (9, 9)
        tile_h, tile_w = sample.shape[2] // 9, sample.shape[3] // 9
        tiles = sample.unfold(2, tile_h, tile_h).unfold(3, tile_w, tile_w).reshape(-1, 1, tile_h, tile_w)
        with torch.no_grad():
            logits = classifier(tiles)
        idx = torch.topk(logits, k=2, dim=1).indices
        pred = idx[:, 0]
        pred[pred == 0] = idx[pred == 0, 1]
        grid = pred.reshape(9, 9).cpu().numpy()  # values 1-9

        invalid = np.zeros((9, 9), dtype=bool)
        # Check rows
        for r in range(9):
            for c1 in range(9):
                for c2 in range(c1 + 1, 9):
                    if grid[r, c1] == grid[r, c2]:
                        invalid[r, c1] = invalid[r, c2] = True
        # Check columns
        for c in range(9):
            for r1 in range(9):
                for r2 in range(r1 + 1, 9):
                    if grid[r1, c] == grid[r2, c]:
                        invalid[r1, c] = invalid[r2, c] = True
        # Check 3x3 boxes
        for br in range(3):
            for bc in range(3):
                cells = [(br*3+dr, bc*3+dc) for dr in range(3) for dc in range(3)]
                for i in range(len(cells)):
                    for j in range(i + 1, len(cells)):
                        r1, c1 = cells[i]
                        r2, c2 = cells[j]
                        if grid[r1, c1] == grid[r2, c2]:
                            invalid[r1, c1] = invalid[r2, c2] = True
        return invalid

    def _overlay_invalid_cells(self, img: np.ndarray, invalid_mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
        """Overlay red on invalid cells. img: [H, W] or [H, W, C] in [0,1]. Returns [H, W, 3]."""
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = img.copy()
        h, w = img.shape[:2]
        cell_h, cell_w = h // 9, w // 9
        for r in range(9):
            for c in range(9):
                if invalid_mask[r, c]:
                    y0, y1 = r * cell_h, (r + 1) * cell_h
                    x0, x1 = c * cell_w, (c + 1) * cell_w
                    img[y0:y1, x0:x1, 0] = img[y0:y1, x0:x1, 0] * (1 - alpha) + alpha
                    img[y0:y1, x0:x1, 1] = img[y0:y1, x0:x1, 1] * (1 - alpha)
                    img[y0:y1, x0:x1, 2] = img[y0:y1, x0:x1, 2] * (1 - alpha)
        return np.clip(img, 0, 1)

    def _visualize_ipr_process(self, snapshots, save_path="ipr_process.png"):
        """Visualize IPR iterations for paper appendix.

        Shows sampled iterations horizontally. Subsamples to max ~10 if needed.
        For MNIST Sudoku, invalid cells are highlighted with red overlay.
        """
        is_sudoku = self.cfg.task_name == "mnist_sudoku"

        max_display = 7 # 10
        total = len(snapshots)

        if total <= max_display:
            selected = list(range(total))
        else:
            selected = [0]
            step = (total - 1) / (max_display - 1)
            for i in range(1, max_display - 1):
                selected.append(round(i * step))
            selected.append(total - 1)
            selected = sorted(set(selected))

        num_cols = len(selected)
        cell_size = 1.8
        fig, axes = plt.subplots(1, num_cols, figsize=(cell_size * num_cols, cell_size + 0.4))
        if num_cols == 1:
            axes = [axes]

        for ax_idx, snap_idx in enumerate(selected):
            title, z_t = snapshots[snap_idx]
            img = z_t[0].detach().permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            if is_sudoku:
                invalid_mask = self._find_invalid_sudoku_cells(z_t)
                img = self._overlay_invalid_cells(img, invalid_mask)
                n_invalid = int(invalid_mask.sum())
                title = f"{title}\n({n_invalid} err)"

            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                axes[ax_idx].imshow(img.squeeze(-1) if img.ndim == 3 else img, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
            else:
                axes[ax_idx].imshow(img, interpolation='nearest')

            axes[ax_idx].set_xlabel(title, fontsize=7, labelpad=2)
            axes[ax_idx].set_xticks([])
            axes[ax_idx].set_yticks([])
            for spine in axes[ax_idx].spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('#cccccc')

        plt.subplots_adjust(wspace=0.08)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.03)
        plt.close()
        print(f"[IPR] Visualization saved to {save_path}")

    def _create_reset_node(
        self,
        node: 'TreeNode',
        patches_to_reset: List[int],
        mask: Optional[Tensor],
        masked: Optional[Tensor],
        device: torch.device,
        initial_noise: Optional[Tensor] = None,
    ) -> Tuple['TreeNode', Tensor]:

        if len(patches_to_reset) == 0:
            return node.clone(), None

        patch_size = self.sampler.patch_size
        grid_w = self.sampler.patch_grid_shape[1]
        total_patches = prod(self.sampler.patch_grid_shape)

        z_t = torch.empty_like(node.z_t).copy_(node.z_t)
        for patch_id in patches_to_reset:
            h, w = patch_id // grid_w, patch_id % grid_w
            h0, h1 = h * patch_size, (h + 1) * patch_size
            w0, w1 = w * patch_size, (w + 1) * patch_size

            if self.cfg.noise_type == "fixed" and initial_noise is not None:
                noise = initial_noise[:, :, h0:h1, w0:w1]
            else:
                noise = torch.randn_like(z_t[:, :, h0:h1, w0:w1])
            z_t[:, :, h0:h1, w0:w1] = noise

        if mask is not None and masked is not None:
            mask_cpu = mask.cpu() if mask.device.type != 'cpu' else mask
            masked_cpu = masked.cpu() if masked.device.type != 'cpu' else masked
            z_t = masked_cpu + mask_cpu * z_t

        is_unknown_map = torch.zeros(1, total_patches, dtype=torch.bool, device=device)
        for patch_id in patches_to_reset:
            is_unknown_map[0, patch_id] = True

        num_unknown_patches = is_unknown_map.sum(dim=1).long()
        num_inference_blocks = torch.ceil(num_unknown_patches / self.cfg.top_k).int()
        n = num_inference_blocks
        p = self.cfg.ipr_steps_per_patch
        max_steps = ceil(p + (n - 1) * (1 - self.cfg.ipr_overlap) * p)

        scheduling_matrix = torch.ones(
            [max_steps + 1, 1, total_patches], device=device
        )
        scheduling_matrix *= is_unknown_map

        ideal_block_lengths = max_steps / (
            (num_inference_blocks - 1) * (1 - self.cfg.ipr_overlap) + 1
        )
        block_lengths = ideal_block_lengths.ceil().int()

        block_starts = (
            torch.arange(num_inference_blocks.max() + 1, device=device).unsqueeze(0)
            * ideal_block_lengths.unsqueeze(1)
            * (1 - self.cfg.ipr_overlap)
        ).floor_()
        block_starts[:, -1] = -1

        new_prototypes = self.sampler.get_schedule_prototypes(block_lengths)

        new_node = TreeNode(
            z_t=z_t.cpu(),
            scheduling_matrix=scheduling_matrix.cpu(),
            is_unknown_map=is_unknown_map.cpu(),
            block_starts=block_starts.cpu(),
            block_counters=torch.zeros(1, device="cpu", dtype=torch.int64),
            step_targets=torch.zeros(1, device="cpu", dtype=torch.int64),
            depth=0,
            step_id=0
        )

        return new_node, new_prototypes

    def _greedy(
        self,
        node,
        model,
        label,
        c_cat,
        eps,
        mask,
        masked,
        prototypes,
        device: torch.device,
    ):
        if node.children:
            child_idx = 0
            for i in range(len(node.children)):
                if not node.children[i].failed:
                    child_idx = i
                    break
            node = node.children[child_idx]

        max_steps = node.scheduling_matrix.shape[0] - 1
        denoising_steps = max_steps - node.step_id + 1
        denoise_out = self._parallel_denoise(
            [node], prototypes, denoising_steps,
            model, label, c_cat, eps, mask, masked,
            simulate_patch_selection=True
        )
        z_t, schedules, is_unknown, block_starts, block_counters, step_targets, current_step_ids = denoise_out

        del denoise_out

        node = TreeNode(
            z_t=z_t,
            scheduling_matrix=schedules,
            is_unknown_map=is_unknown,
            block_starts=block_starts,
            block_counters=block_counters,
            step_targets=step_targets,
            step_id=current_step_ids[0].item(),
        )
        return node

    def _check(self, x0_preds, is_unknown=None) -> List[float]:
        if self.task_name == "mnist_sudoku":
            return self.check_sudoku(x0_preds, is_unknown)
        elif self.task_name == "even_pixels":
            return self.check_even_pixels(x0_preds, is_unknown)
        elif self.task_name == "counting_objects":
            return self.check_counting_objects(x0_preds, is_unknown)

    def check_sudoku(self, x0_preds, is_unknown=None) -> List[float]:
        N, C, H, W = x0_preds.shape
        assert H == 252 and W == 252, f"Expected 252x252, got {H}x{W}"
        patch_size = 28
        grid_size = 9

        x0_patches = x0_preds.reshape(N, 1, grid_size, patch_size, grid_size, patch_size)
        x0_patches = x0_patches.permute(0, 2, 4, 1, 3, 5)
        x0_patches = x0_patches.reshape(-1, 1, patch_size, patch_size)

        with torch.no_grad():
            logits = self.mnist_classifier(x0_patches)
            preds = logits.argmax(dim=1)

        pred_board = preds.reshape(-1, grid_size * grid_size)
        results = torch.zeros(N, device=self.device, dtype=torch.bool)

        for batch_idx in range(N):
            board = pred_board[batch_idx].reshape(grid_size, grid_size)
            violations = 0

            if is_unknown is None:
                for r in range(9):
                    row = board[r, :]
                    row_non_zeros = row[row != 0]
                    violations += row_non_zeros.size(0) - torch.unique(row_non_zeros).size(0)
                for c in range(9):
                    col = board[:, c]
                    col_non_zeros = col[col != 0]
                    violations += col_non_zeros.size(0) - torch.unique(col_non_zeros).size(0)
                for bi in range(0, 9, 3):
                    for bj in range(0, 9, 3):
                        box = board[bi:bi+3, bj:bj+3].flatten()
                        box_non_zeros = box[box != 0]
                        violations += box_non_zeros.size(0) - torch.unique(box_non_zeros).size(0)
            else:
                known_patches = ~is_unknown[batch_idx].reshape(grid_size, grid_size)
                for r in range(9):
                    row = board[r, :]
                    known = known_patches[r, :]
                    row_non_zeros = row[(row != 0) & known]
                    violations += row_non_zeros.size(0) - torch.unique(row_non_zeros).size(0)
                for c in range(9):
                    col = board[:, c]
                    known = known_patches[:, c]
                    col_non_zeros = col[(col != 0) & known]
                    violations += col_non_zeros.size(0) - torch.unique(col_non_zeros).size(0)
                for bi in range(0, 9, 3):
                    for bj in range(0, 9, 3):
                        box = board[bi:bi+3, bj:bj+3].flatten()
                        known = known_patches[bi:bi+3, bj:bj+3].flatten()
                        box_non_zeros = box[(box != 0) & known]
                        violations += box_non_zeros.size(0) - torch.unique(box_non_zeros).size(0)

            results[batch_idx] = False if violations > 0 else True

        return results.cpu().tolist()

    def check_even_pixels(self, x0_preds, is_unknown=None) -> List[float]:
        N, C, H, W = x0_preds.shape
        assert H == 32 and W == 32, f"Expected 32x32, got {H}x{W}"
        patch_size = 4
        grid_size = 8
        results = torch.zeros(N, device=self.device, dtype=torch.bool)

        for batch_idx in range(N):
            if is_unknown is not None:
                known_patches = ~is_unknown[batch_idx].reshape(grid_size, grid_size)
                new_image = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        if known_patches[i][j]:
                            patch = x0_preds[batch_idx, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                            new_image.append(patch)
                if not new_image:
                    results[batch_idx] = True
                    continue
                images = torch.cat(new_image, dim=-1).unsqueeze(0)
            else:
                images = x0_preds[batch_idx].unsqueeze(0)
            res = self.evaluator.evaluate(images)

            if int(is_unknown.sum()) == 0 and res['color_imbalance_count'] != 0:
                results[batch_idx] = False
            else:
                results[batch_idx] = True

        return results.cpu().tolist()

    def check_counting_objects(self, x0_preds, is_unknown=None) -> List[float]:
        N = x0_preds.shape[0]
        results = [True] * N

        if is_unknown is not None:
            for i in range(N):
                if int(is_unknown[i].sum()) > 0:
                    continue
                sample = x0_preds[i:i+1]
                metrics = self.evaluator.evaluate(sample)
                is_uniform = metrics.get('are_vertices_uniform', 0.0) == 1.0
                nums_match = metrics.get('numbers_match_objects', 0.0) == 1.0
                if not (is_uniform and nums_match):
                    results[i] = False

        return results

    @torch.no_grad()
    def _parallel_denoise(
        self,
        nodes,
        prototypes,
        denoising_steps,
        model,
        label,
        c_cat,
        eps,
        mask,
        masked,
        simulate_patch_selection: bool = False,
        batch_size: int = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        if batch_size is None:
            batch_size = len(nodes)

        all_z_t = []
        all_schedules = []
        all_is_unknown = []
        all_block_starts = []
        all_block_counters = []
        all_step_targets = []
        all_final_step_ids = []

        max_steps = nodes[0].scheduling_matrix.shape[0] - 1
        total_nodes = len(nodes)
        prototypes_cpu = prototypes.cpu()

        for start in range(0, total_nodes, batch_size):
            end = min(start + batch_size, total_nodes)
            batch_nodes = nodes[start:end]
            num_nodes = len(batch_nodes)

            z_t = torch.cat([n.z_t.clone() for n in batch_nodes], dim=0)
            schedules = torch.cat([n.scheduling_matrix.clone() for n in batch_nodes], dim=1)
            is_unknown = torch.cat([n.is_unknown_map.clone() for n in batch_nodes], dim=0)
            block_starts = torch.cat([n.block_starts.clone() for n in batch_nodes], dim=0)
            block_counters = torch.cat([n.block_counters.clone() for n in batch_nodes], dim=0)
            step_targets = torch.cat([n.step_targets.clone() for n in batch_nodes], dim=0)

            original_step_ids = torch.tensor(
                [n.step_id for n in batch_nodes], device=z_t.device, dtype=torch.long,
            )

            total_patches = prod(self.sampler.patch_grid_shape)
            image_shape = z_t.shape[-2:]

            batch_label_rollout = label.repeat(num_nodes) if label is not None else None
            batch_c_cat_rollout = c_cat.repeat(num_nodes, 1, 1, 1) if c_cat is not None else None

            for step_offset in range(int(denoising_steps)):
                current_step_ids = original_step_ids + step_offset

                nodes_at_max_steps = current_step_ids >= max_steps
                if nodes_at_max_steps.all():
                    break

                valid_indices = ~nodes_at_max_steps
                if valid_indices.sum() == 0:
                    break

                valid_z_t = z_t[valid_indices]
                valid_schedules = schedules[:, valid_indices, :]
                valid_step_ids = current_step_ids[valid_indices]

                current_t_batch = self.sampler.get_timestep_batch_from_schedule(
                    valid_schedules, valid_step_ids, image_shape,
                )

                valid_label = batch_label_rollout[valid_indices] if batch_label_rollout is not None else None
                valid_c_cat = batch_c_cat_rollout[valid_indices] if batch_c_cat_rollout is not None else None

                valid_z_t_gpu = valid_z_t.to(self.device)
                current_t_batch_gpu = current_t_batch.to(self.device)

                mean_theta, v_theta, sigma_theta = model.forward(
                    z_t=valid_z_t_gpu.unsqueeze(1),
                    t=current_t_batch_gpu.unsqueeze(1),
                    label=valid_label,
                    c_cat=valid_c_cat,
                    sample=True,
                    use_ema=self.cfg.use_ema,
                )
                sigma_theta.squeeze_(1)

                valid_step_targets = step_targets[valid_indices]
                should_predict = valid_step_targets == valid_step_ids

                if should_predict.any():
                    block_counters[valid_indices] += should_predict.int()
                    step_targets[valid_indices] = block_starts[
                        valid_indices, block_counters[valid_indices],
                    ]

                valid_is_unknown = is_unknown[valid_indices]
                unknown_and_predict = valid_is_unknown.any(dim=1) & should_predict

                if simulate_patch_selection and unknown_and_predict.any():
                    num_valid_nodes = valid_z_t.shape[0]

                    patch_sigma_theta = avg_pool2d(
                        sigma_theta, kernel_size=self.sampler.patch_size,
                    ).reshape(num_valid_nodes, total_patches)

                    known_shift = patch_sigma_theta.max() + 1
                    patch_sigma_theta_masked = (
                        patch_sigma_theta + ~valid_is_unknown.to(self.device) * known_shift
                    )

                    patch_ids = patch_sigma_theta_masked.argmin(dim=1)
                    patch_ids_cpu = patch_ids.cpu()

                    is_unknown[valid_indices, patch_ids_cpu] = False

                    for rel_idx in torch.nonzero(unknown_and_predict).squeeze(-1):
                        rel_idx = rel_idx.item() if rel_idx.dim() == 0 else rel_idx
                        abs_idx = valid_indices.nonzero().squeeze(-1)[rel_idx].item()
                        step_id = valid_step_ids[rel_idx].item()
                        patch_id_to_update = patch_ids_cpu[rel_idx].item()

                        length_to_consider = min(
                            prototypes_cpu.shape[0], max_steps - step_id,
                        )

                        schedules[
                            step_id:step_id + length_to_consider,
                            abs_idx,
                            patch_id_to_update,
                        ] = torch.minimum(
                            prototypes_cpu[:length_to_consider, 0],
                            schedules[
                                step_id:step_id + length_to_consider,
                                abs_idx,
                                patch_id_to_update,
                            ],
                        )

                        if step_id + length_to_consider < max_steps:
                            schedules[
                                step_id + length_to_consider:,
                                abs_idx,
                                patch_id_to_update,
                            ] = 0

                    schedules[-1] = 0

                # Next step sampling
                t_next_batch = self.sampler.get_timestep_batch_from_schedule(
                    schedules, valid_step_ids + 1, image_shape,
                )
                t_next_batch_gpu = t_next_batch.to(self.device)

                p = model.flow.conditional_p(
                    mean_theta,
                    valid_z_t_gpu.unsqueeze(1),
                    current_t_batch_gpu.unsqueeze(1),
                    t_next_batch_gpu.unsqueeze(1),
                    self.cfg.stochasticity,
                    self.cfg.temperature,
                    v_theta=v_theta,
                )

                next_z_t = torch.where(
                    t_next_batch_gpu.unsqueeze(1) > 0,
                    p.sample(),
                    p.mean,
                ).squeeze(1)

                if mask is not None:
                    batch_mask = mask.repeat(num_nodes, 1, 1, 1)[valid_indices]
                    batch_masked = masked.repeat(num_nodes, 1, 1, 1)[valid_indices]
                    next_z_t = batch_masked + batch_mask * next_z_t

                z_t[valid_indices] = next_z_t.cpu()

                del valid_z_t_gpu, current_t_batch_gpu, mean_theta, v_theta, sigma_theta
                del t_next_batch_gpu, p, next_z_t

            final_step_ids = torch.minimum(
                original_step_ids + denoising_steps,
                torch.tensor(max_steps, device=original_step_ids.device, dtype=torch.long),
            )

            all_z_t.append(z_t)
            all_schedules.append(schedules)
            all_is_unknown.append(is_unknown)
            all_block_starts.append(block_starts)
            all_block_counters.append(block_counters)
            all_step_targets.append(step_targets)
            all_final_step_ids.append(final_step_ids)

        return (
            torch.cat(all_z_t, dim=0),
            torch.cat(all_schedules, dim=1),
            torch.cat(all_is_unknown, dim=0),
            torch.cat(all_block_starts, dim=0),
            torch.cat(all_block_counters, dim=0),
            torch.cat(all_step_targets, dim=0),
            torch.cat(all_final_step_ids, dim=0),
        )


class IPRSampler(Sampler[IPRSamplerCfg]):
    def __init__(self, cfg, patch_size, patch_grid_shape, dependency_matrix=None):
        super().__init__(cfg, patch_size, patch_grid_shape, dependency_matrix)
        self.algo = None
        self.budget_stats = {}
        self.sample_count = 0
        self.global_log_step = 0
        self._init_srmbench_evaluator()

    def _init_srmbench_evaluator(self):
        self.srmbench_evaluator = None
        if self.cfg.task_name == "mnist_sudoku":
            self.srmbench_evaluator = MnistSudokuEvaluation()
        elif self.cfg.task_name == "even_pixels":
            self.srmbench_evaluator = EvenPixelsEvaluation()
        elif self.cfg.task_name == "counting_objects":
            self.srmbench_evaluator = CountingObjectsEvaluation(
                object_variant="polygons",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

    def update_budget_stats(self, budget: int, metrics: dict):
        if budget not in self.budget_stats:
            self.budget_stats[budget] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                val = value.float().mean().item()
            elif isinstance(value, (int, float)):
                val = float(value)
            else:
                continue
            if key not in self.budget_stats[budget]:
                self.budget_stats[budget][key] = {"sum": 0.0, "count": 0}
            self.budget_stats[budget][key]["sum"] += val
            self.budget_stats[budget][key]["count"] += 1

    def evaluate_and_log(self, sample: torch.Tensor, budget: int):
        if self.srmbench_evaluator is None:
            return
        device = sample.device
        sample_float = sample.float()
        if hasattr(self.srmbench_evaluator, 'classifier'):
            self.srmbench_evaluator.classifier = self.srmbench_evaluator.classifier.to(device)
        with torch.no_grad():
            metrics = self.srmbench_evaluator.evaluate(sample_float)
        self.update_budget_stats(budget, metrics)

    def log_all_budgets(self):
        if wandb.run is None:
            return
        self.sample_count += 1
        log_dict = {}
        for budget, budget_metrics in self.budget_stats.items():
            for key, stats in budget_metrics.items():
                avg_value = stats["sum"] / stats["count"] if stats["count"] > 0 else 0.0
                log_dict[f"budget_{budget}/{key}"] = avg_value
        wandb.log(log_dict, step=self.sample_count, commit=True)

    def get_inference_lengths(
        self, num_inference_blocks: Int32[Tensor, "batch_size"], max_steps: int
    ) -> Float[Tensor, "batch_size"]:
        ideal_lengths = max_steps / (
            (num_inference_blocks - 1) * (1 - self.cfg.overlap) + 1
        )
        return ideal_lengths

    def get_schedule_prototypes(
        self, prototype_lengths: Int32[Tensor, "batch_size"]
    ) -> Float[Tensor, "max_prototype_length batch_size"]:
        batch_size = prototype_lengths.size(0)
        device = prototype_lengths.device
        max_prototype_length = prototype_lengths.max()
        assert prototype_lengths.min() > 0
        prototype_base = torch.linspace(
            max_prototype_length, 0, max_prototype_length + 1, device=device
        )
        prototypes = prototype_base.unsqueeze(0).expand(batch_size, -1)
        prototypes = prototypes - (max_prototype_length - prototype_lengths).unsqueeze(1)
        prototypes = prototypes / prototype_lengths.unsqueeze(1)
        assert prototypes.max() <= 1 + self.cfg.epsilon
        prototypes.clamp_(0, 1)
        prototypes = prototypes.T
        return prototypes[:-1]

    def get_timestep_from_schedule(
        self,
        scheduling_matrix: Float[Tensor, "total_steps batch_size total_patches"],
        step_id: int,
        image_shape: Sequence[int],
    ) -> Float[Tensor, "batch_size 1 height width"]:
        assert step_id < scheduling_matrix.shape[0]
        batch_size = scheduling_matrix.shape[1]
        step_id = int(step_id)
        t_patch = scheduling_matrix[step_id].reshape(batch_size, *self.patch_grid_shape)
        return interpolate(t_patch.unsqueeze(1), size=image_shape, mode="nearest-exact")

    def get_timestep_batch_from_schedule(
        self,
        scheduling_matrix: torch.Tensor,
        step_ids: torch.Tensor,
        image_shape: Sequence[int],
    ) -> Float[Tensor, "batch_size 1 height width"]:
        total_steps, batch_size, total_patches = scheduling_matrix.shape
        device = scheduling_matrix.device
        if batch_size == 0 or step_ids.shape[0] == 0:
            return torch.empty(0, 1, *image_shape, device=device)
        assert step_ids.shape[0] == batch_size
        clamped_step_ids = torch.clamp(step_ids, 0, total_steps - 1)
        t_patches = scheduling_matrix[clamped_step_ids, torch.arange(batch_size, device=device)]
        t_patches = t_patches.reshape(batch_size, *self.patch_grid_shape)
        t_batch = torch.nn.functional.interpolate(
            t_patches.unsqueeze(1), size=image_shape, mode="nearest-exact"
        )
        return t_batch

    def full_mask_to_sequence_mask(
        self,
        full_mask: Float[Tensor, "batch_size channels height width"] | None,
    ) -> Float[Tensor, "batch_size num_patches"]:
        batch_size = full_mask.shape[0]
        sequence_mask = full_mask[:, 0, ::self.patch_size, ::self.patch_size]
        sequence_mask = sequence_mask.reshape(batch_size, -1)
        return sequence_mask

    @torch.no_grad()
    def _single_sample(
        self,
        model: Wrapper,
        image_shape: Sequence[int] | None = None,
        z_t: Float[Tensor, "batch dim height width"] | None = None,
        t: Float[Tensor, "batch 1 height width"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
        masked: Float[Tensor, "batch dim height width"] | None = None,
        return_intermediate: bool = False,
        return_time: bool = False,
        return_sigma: bool = False,
        return_x: bool = False,
    ) -> SamplingOutput:
        assert z_t is not None and z_t.size(0) == 1, "Sampler assumes batch_size=1"

        gc.collect()
        self.algo = IPR(sampler=self)

        image_shape = z_t.shape[-2:] if image_shape is None else image_shape
        total_patches = prod(self.patch_grid_shape)
        device = z_t.device

        z_t, t, label, c_cat, eps = self.get_defaults(
            model, 1, image_shape, z_t, t, label, mask, masked
        )

        is_unknown_map = (
            self.full_mask_to_sequence_mask(mask)
            if mask is not None
            else torch.ones(1, total_patches, device=device)
        ) > 0.5

        if self.cfg.mode == "standard":
            num_unknown_patches = is_unknown_map.sum(dim=1).long()
            num_inference_blocks = torch.ceil(num_unknown_patches / self.cfg.top_k).int()
            n = num_inference_blocks
            p = self.cfg.steps_per_patch
            max_steps = ceil(p + (n - 1) * (1 - self.cfg.overlap) * p)

            scheduling_matrix = torch.ones(
                [max_steps + 1, 1, total_patches], device=device
            )
            scheduling_matrix *= is_unknown_map

            num_unknown_patches = is_unknown_map.sum(dim=1).long()
            num_inference_blocks = torch.ceil(num_unknown_patches / self.cfg.top_k).int()

            ideal_block_lengths = self.get_inference_lengths(num_inference_blocks, max_steps)
            block_lengths = ideal_block_lengths.ceil().int()

            block_starts = (
                torch.arange(num_inference_blocks.max() + 1, device=device).unsqueeze(0)
                * ideal_block_lengths.unsqueeze(1)
                * (1 - self.cfg.overlap)
            ).floor_()
            block_starts[:, -1] = -1

            prototypes = self.get_schedule_prototypes(block_lengths)

            root_node = TreeNode(
                z_t=z_t.cpu(),
                scheduling_matrix=scheduling_matrix.cpu(),
                is_unknown_map=is_unknown_map.cpu(),
                block_starts=block_starts.cpu(),
                block_counters=torch.zeros(1, device="cpu", dtype=torch.int64),
                step_targets=torch.zeros(1, device="cpu", dtype=torch.int64),
                depth=0,
            )

            if c_cat is not None:
                c_cat = c_cat.unsqueeze(1)

        elif self.cfg.mode == "correction":
            z_t = self._corrupt_sudoku_images(z_t)
            is_unknown_map = torch.zeros(1, total_patches, dtype=torch.bool, device=device)
            if mask is not None:
                mask = torch.ones_like(mask)
            if masked is not None:
                masked = torch.zeros_like(masked)

            root_node = TreeNode(
                z_t=z_t.cpu(),
                scheduling_matrix=None,
                is_unknown_map=is_unknown_map.cpu(),
                block_starts=None,
                block_counters=None,
                step_targets=None,
                depth=0,
            )
            prototypes = None

        ans = self.algo.solve(
            root_node, model, label, c_cat, eps, mask, masked, prototypes, device
        )

        res: SamplingOutput = {"sample": ans.z_t.to(device)}
        return res

    def _corrupt_sudoku_images(self, z_t: Tensor) -> Tensor:
        z_t_corrupted = z_t.clone()
        batch_size = z_t.shape[0]
        patch_size = self.patch_size
        grid_h, grid_w = self.patch_grid_shape
        k = self.cfg.num_corruption_swaps

        for i in range(batch_size):
            for _ in range(k):
                rows = random.sample(range(grid_h), 2)
                cols = random.sample(range(grid_w), 2)
                h1, w1 = rows[0], cols[0]
                h2, w2 = rows[1], cols[1]

                h1_0, h1_1 = h1 * patch_size, (h1 + 1) * patch_size
                w1_0, w1_1 = w1 * patch_size, (w1 + 1) * patch_size
                h2_0, h2_1 = h2 * patch_size, (h2 + 1) * patch_size
                w2_0, w2_1 = w2 * patch_size, (w2 + 1) * patch_size

                patch1 = z_t_corrupted[i, :, h1_0:h1_1, w1_0:w1_1].clone()
                patch2 = z_t_corrupted[i, :, h2_0:h2_1, w2_0:w2_1].clone()
                z_t_corrupted[i, :, h1_0:h1_1, w1_0:w1_1] = patch2
                z_t_corrupted[i, :, h2_0:h2_1, w2_0:w2_1] = patch1
        return z_t_corrupted

    def sample(
        self,
        model: Wrapper,
        batch_size: int | None = None,
        image_shape: Sequence[int] | None = None,
        z_t: Float[Tensor, "batch dim height width"] | None = None,
        t: Float[Tensor, "batch 1 height width"] | None = None,
        label: Int64[Tensor, "batch"] | None = None,
        mask: Float[Tensor, "batch 1 height width"] | None = None,
        masked: Float[Tensor, "batch dim height width"] | None = None,
        return_intermediate: bool = False,
        return_time: bool = False,
        return_sigma: bool = False,
        return_x: bool = False,
    ) -> SamplingOutput:
        assert z_t is not None, "z_t must be provided"
        batch_size = z_t.size(0)

        res: SamplingOutput = {"sample": []}
        for i in tqdm(range(batch_size)):
            seed_everything(42 + i)
            single_res = self._single_sample(
                model=model,
                image_shape=image_shape,
                z_t=z_t[i:i+1],
                t=t[i:i+1] if t is not None else None,
                label=label[i:i+1] if label is not None else None,
                mask=mask[i:i+1] if mask is not None else None,
                masked=masked[i:i+1] if masked is not None else None,
                return_intermediate=return_intermediate if batch_size == 1 else False,
                return_time=return_time,
                return_sigma=return_sigma,
                return_x=return_x,
            )
            res["sample"].append(single_res["sample"])

        res["sample"] = torch.cat(res["sample"], dim=0)
        if return_intermediate and batch_size == 1:
            res.update({k: v for k, v in single_res.items() if k != "sample"})
        return res
