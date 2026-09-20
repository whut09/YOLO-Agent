"""Paper-specific distillation mechanism losses with full-text provenance.

Every implementation below was recovered from the actual paper full text
(`runs/paper_fulltext/`, PDFs fetched from the official venues) during the
frozen-83 gap-closure loop.  Each class documents its paper, the equation it
implements, and the adaptation contract to the shared YOLO26 distillation
tensor surface (teacher detached, student differentiable, CPU-friendly).
"""

from __future__ import annotations

import math
from typing import Any

from yolo_agent.components.distillation.mechanism_losses import (
    DistillationInputs,
    DistillationLossOutput,
    DistillationMechanismLoss,
    _feature_pairs,
    _same_shape,
)


def _mask_grid_to(logits: Any, grid_hw: tuple[int, int]) -> Any:
    """Resample a (N, C, L) or (N, C, H, W) logit tensor to a (N, 1, H, W) grid.

    Mask builders must accept both dense head layouts: 4D (N,C,H,W) maps and
    3D token layouts (N,C,L) whose L is interpreted as a linear spatial index.
    """
    import torch

    x = logits.detach().float()
    if x.ndim == 4:
        if x.shape[-2:] != tuple(grid_hw):
            x = torch.nn.functional.interpolate(
                x, size=tuple(grid_hw), mode="bilinear", align_corners=False
            )
    elif x.ndim == 3:
        n, c, length = x.shape
        # Factorize the linear token index into the closest grid; positions
        # are resampled linearly to the target grid (shared-primitive policy:
        # geometric alignment only).
        side = max(1, int(round(length ** 0.5)))
        grid = x.reshape(n, c, side, -1).mean(dim=-1) if side * side != length else x.reshape(n, c, side, side)
        if grid.shape[-2:] != tuple(grid_hw):
            grid = torch.nn.functional.interpolate(
                grid, size=tuple(grid_hw), mode="bilinear", align_corners=False
            )
        x = grid
    else:
        raise ValueError(f"unsupported logits ndim for mask construction: {x.ndim}")
    return x.mean(dim=1, keepdim=True)


def _pool_channels_to(x: Any, target: int) -> Any:
    """Linearly resample the channel axis of a 4D N,C,H,W tensor to ``target``.

    Used by several paper mechanisms to compare heterogeneous teacher/student
    channel widths without introducing trainable adapters (shared-primitive
    policy: geometric alignment only, no learned projection).
    """
    import torch

    if x.shape[1] == target:
        return x
    flat = x.permute(0, 2, 3, 1).reshape(-1, 1, x.shape[1])  # N*H*W, 1, C
    pooled = torch.nn.functional.interpolate(
        flat, size=target, mode="linear", align_corners=False
    )
    return pooled.reshape(x.shape[0], x.shape[2], x.shape[3], target).permute(
        0, 3, 1, 2
    )


def _resolve_box_tensors(inputs: DistillationInputs) -> tuple[Any, Any]:
    if inputs.student_boxes is None or inputs.teacher_boxes is None:
        raise ValueError("mechanism requires student and teacher boxes")
    return _same_shape(inputs.student_boxes, inputs.teacher_boxes)


# ---------------------------------------------------------------------------
# PKD: General Distillation Framework for Object Detectors via Pearson
# Correlation Coefficient (NeurIPS 2022, 631ad9ae...), Eq. 2-4:
#     L_FPN = 1 - r(s, t)   with  r = Pearson correlation coefficient
# The paper proves normalizing features to zero-mean/unit-variance and taking
# MSE is identical to 1 - r; we implement r directly on channel-flattened
# spatial energy so heterogeneous channel counts still align per level.
# ---------------------------------------------------------------------------
class PearsonFeatureDistillationLoss(DistillationMechanismLoss):
    mechanism = "pearson_feature"

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        if epsilon <= 0.0:
            raise ValueError("pearson epsilon must be positive")
        self.epsilon = epsilon

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("pearson feature distillation requires features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        coefficients = []
        for student, teacher in pairs:
            student = student.float().mean(dim=1)  # (N, H, W) spatial energy
            teacher = teacher.detach().float().mean(dim=1)
            if student.shape[1:] != teacher.shape[1:]:
                teacher = torch.nn.functional.interpolate(
                    teacher.unsqueeze(1),
                    size=tuple(student.shape[1:]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            s = student.flatten(1)
            t = teacher.flatten(1)
            s = (s - s.mean(dim=1, keepdim=True)) / s.std(dim=1, keepdim=True).clamp_min(
                self.epsilon
            )
            t = (t - t.mean(dim=1, keepdim=True)) / t.std(dim=1, keepdim=True).clamp_min(
                self.epsilon
            )
            r = (s * t).mean(dim=1)
            coefficients.append(r.mean().detach())
            losses.append((1.0 - r).mean())
        return DistillationLossOutput(
            loss=torch.stack(losses).mean(),
            metrics={
                "mean_pearson_r": float(torch.stack(coefficients).mean().cpu()),
                "feature_level_count": float(len(pairs)),
            },
        )


# ---------------------------------------------------------------------------
# FRS: Distilling Object Detectors with Feature Richness (NeurIPS 2021,
# 29c0c0ee...), Eq. 4-8.  The richness mask is S = max_c y_t(c) taken from the
# teacher classification score map; FPN features and classification head are
# both distilled under the per-position normalized mask.  We synthesize the
# score map from the teacher logits tensor surface and reuse it for both terms.
# ---------------------------------------------------------------------------
class RichnessMaskedDistillationLoss(DistillationMechanismLoss):
    mechanism = "richness_masked"

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        if epsilon <= 0.0:
            raise ValueError("richness epsilon must be positive")
        self.epsilon = epsilon

    def _mask_from_logits(self, logits: Any, grid_hw: tuple[int, int] | None = None) -> Any:
        import torch

        # S = max over classes of the (softmax) teacher score map, evaluated
        # on a 4D grid; 3D token-layout logits are factorized first.
        scores = torch.nn.functional.softmax(logits.detach().float(), dim=1)
        if scores.ndim == 3:
            n, c, length = scores.shape
            side = max(1, int(round(length ** 0.5)))
            # Deterministic factorization: pad the tail to side*side with the
            # per-token minimum score (softmax > 0, so padding is neutral for
            # the amax below) before reshaping to the square grid.
            if side * side != length:
                pad = side * side - length
                scores = torch.nn.functional.pad(scores, (0, pad), value=0.0)
            scores = scores.reshape(n, c, side, side)
        if grid_hw is not None and scores.shape[-2:] != tuple(grid_hw):
            scores = torch.nn.functional.interpolate(
                scores, size=tuple(grid_hw), mode="bilinear", align_corners=False
            )
        mask = scores.amax(dim=1, keepdim=True)
        total = mask.sum().clamp_min(self.epsilon)
        return mask, total

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("richness distillation requires features")
        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("richness distillation requires head logits for the mask")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        _ = student  # logits term uses the mask against student head outputs
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        fpn_losses = []
        for s_feat, t_feat in pairs:
            s = s_feat.float()
            t = t_feat.detach().float()
            if s.shape[1:] != t.shape[1:]:
                # Eq. 6 adapt(): channel-agnostic projection, matching the
                # shared-primitive policy of comparing spatial energy without
                # adding trainable projectors on the YOLO26 surface.
                if s.shape[1] != t.shape[1]:
                    s_energy = s.square().mean(dim=1, keepdim=True)
                    t_energy = t.square().mean(dim=1, keepdim=True)
                    if s_energy.shape[-2:] != t_energy.shape[-2:]:
                        t_energy = torch.nn.functional.interpolate(
                            t_energy,
                            size=tuple(s_energy.shape[-2:]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    level_mask, _ = self._mask_from_logits(
                        teacher, tuple(s_energy.shape[-2:])
                    )
                    if level_mask.shape[-2:] != s_energy.shape[-2:]:
                        level_mask = torch.nn.functional.interpolate(
                            level_mask,
                            size=tuple(s_energy.shape[-2:]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    normalizer = level_mask.sum().clamp_min(self.epsilon)
                    fpn_losses.append(
                        ((s_energy - t_energy).square() * level_mask).sum() / normalizer
                    )
                    continue
                t = torch.nn.functional.interpolate(
                    t, size=tuple(s.shape[2:]), mode="bilinear", align_corners=False
                )
            level_mask, _ = self._mask_from_logits(teacher, tuple(s.shape[-2:]))
            if level_mask.shape[-2:] != s.shape[-2:]:
                level_mask = torch.nn.functional.interpolate(
                    level_mask,
                    size=tuple(s.shape[-2:]),
                    mode="bilinear",
                    align_corners=False,
                )
            normalizer = level_mask.sum().clamp_min(self.epsilon)
            fpn_losses.append(((s - t).square() * level_mask).sum() / normalizer)
        fpn_loss = torch.stack(fpn_losses).mean()
        head_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            student, torch.nn.functional.softmax(teacher.detach(), dim=1).clamp_min(1e-6)
        )
        head_mask_grid, _ = self._mask_from_logits(teacher, tuple(student.shape[-2:]) if student.ndim == 4 else None)
        head_mask = head_mask_grid.mean(dim=(2, 3), keepdim=True) + 0.5  # soften mask for head
        loss = fpn_loss + (head_loss * head_mask.squeeze()).mean()
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "fpn_masked_loss": float(fpn_loss.detach().cpu()),
                "head_masked_loss": float(head_loss.detach().mean().cpu()),
                "mean_richness": float(head_mask_grid.mean().cpu()),
            },
        )


# ---------------------------------------------------------------------------
# Distilling Image Classifiers in Object Detectors (NeurIPS 2021,
# 082a8bbf...), Eq. 1, 10, 12:  classifier-to-detector KD with
#   L_kd_cls = KL(p_t,T || p_s,T)     (temperature-softened class distributions)
#   L_kd_loc = 1/(K L M H W) sum |A_t(O^p) - A_t(O^gt)|_1   (STN feature loss)
#   L = L_det + lambda_kc L_kd_cls + lambda_kl L_kd_loc
# The YOLO26 contract has no separate ImageNet classifier; the adapted teacher
# signal is the teacher head's class distribution (preserving the cls-KD core),
# and the localization term aligns teacher-vs-student box-derived features.
# ---------------------------------------------------------------------------
class ClassifierResponseDistillationLoss(DistillationMechanismLoss):
    mechanism = "classifier_response"

    def __init__(self, *, temperature: float = 3.0) -> None:
        if temperature <= 0.0:
            raise ValueError("classifier response temperature must be positive")
        self.temperature = temperature

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        temperature = self.temperature
        cls_kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student.float() / temperature, dim=1),
            torch.nn.functional.softmax(teacher.detach().float() / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)
        loc_kd = student.new_zeros(())
        boxes_teacher: Any = None
        boxes_student: Any = None
        if inputs.student_boxes is not None and inputs.teacher_boxes is not None:
            boxes_student, boxes_teacher = _resolve_box_tensors(inputs)
            # Eq. 10/11 adapted: L1 distance between teacher-scored and
            # ground-truth-aligned box embeddings; on the YOLO26 tensor
            # surface the box vector plays the role of the pooled region.
            loc_kd = torch.nn.functional.l1_loss(
                boxes_student, boxes_teacher.detach()
            )
        loss = cls_kd + loc_kd
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "cls_kd_loss": float(cls_kd.detach().cpu()),
                "loc_kd_loss": float(loc_kd.detach().cpu()),
                "temperature": temperature,
            },
        )


# ---------------------------------------------------------------------------
# ICD: Instance-Conditional Knowledge Distillation (NeurIPS 2021,
# 892c91e0...), Eq. 6, 11:  m_ij = softmax(K_T q_i / sqrt(d)) attention from
# instance queries to teacher representations; distillation is the
# attention-weighted MSE over value features with m_ij and V_T detached:
#   L_distill = 1/(M N_r) sum_j sum_i 1[obj] <m_ij, L_MSE(V_S, V_T)>
# The YOLO26 adaptation uses per-instance box centers as queries, projected
# against teacher feature tokens; the learnable decoder is outside the shared
# primitive surface, so the attention here is parameter-free (documented as
# faithful_adaptation, not exact reproduction of the learnable G module).
# ---------------------------------------------------------------------------
class InstanceConditionalDistillationLoss(DistillationMechanismLoss):
    mechanism = "instance_conditional"

    def __init__(self, *, max_tokens: int = 128, epsilon: float = 1e-6) -> None:
        if max_tokens < 4:
            raise ValueError("instance conditional requires at least four tokens")
        self.max_tokens = max_tokens
        self.epsilon = epsilon

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("instance conditional distillation requires features")
        if inputs.student_boxes is None or inputs.teacher_boxes is None:
            raise ValueError("instance conditional distillation requires instance boxes")
        student_boxes, teacher_boxes = _resolve_box_tensors(inputs)
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        for s_feat, t_feat in pairs:
            student = s_feat.float()
            teacher = t_feat.detach().float()
            if student.shape[1:] != teacher.shape[1:]:
                teacher = torch.nn.functional.interpolate(
                    teacher, size=tuple(student.shape[2:]), mode="bilinear", align_corners=False
                )
            n, c, h, w = student.shape
            student_tokens = student.flatten(2).transpose(1, 2)  # N, HW, C
            teacher_tokens = teacher.flatten(2).transpose(1, 2)
            side_h, side_w = h, w
            if student_tokens.shape[1] > self.max_tokens:
                scale = math.sqrt(self.max_tokens / student_tokens.shape[1])
                side_h = max(1, int(h * scale))
                side_w = max(1, int(w * scale))
                student_small = torch.nn.functional.adaptive_avg_pool2d(student, (side_h, side_w))
                teacher_small = torch.nn.functional.adaptive_avg_pool2d(teacher, (side_h, side_w))
                student_tokens = student_small.flatten(2).transpose(1, 2)
                teacher_tokens = teacher_small.flatten(2).transpose(1, 2)
            # Query: instance box centers mapped to token grid coordinates.
            boxes = teacher_boxes.detach().float()
            cx = (boxes[..., 0] + boxes[..., 2]) * 0.5
            cy = (boxes[..., 1] + boxes[..., 3]) * 0.5
            query = torch.stack([cx, cy], dim=-1)  # N, K, 2
            gx = (query[..., 0].clamp(0, 1) * (side_w - 1)).long()
            gy = (query[..., 1].clamp(0, 1) * (side_h - 1)).long()
            flat_index = gx * side_w + gy
            keys = torch.nn.functional.normalize(teacher_tokens, dim=-1)
            attended = keys.gather(
                1, flat_index.unsqueeze(-1).expand(-1, -1, keys.shape[-1])
            )
            logits = torch.einsum(
                "nlc,nkc->nlk", teacher_tokens, attended
            ) / math.sqrt(teacher_tokens.shape[-1])
            m = torch.nn.functional.softmax(logits, dim=1)  # attention mask
            values = student_tokens
            if values.shape[-1] != teacher_tokens.shape[-1]:
                # Channel-agnostic comparison per the shared-primitive policy:
                # pool heterogeneous channel widths to a common width before
                # the attention-weighted MSE over value features (Eq. 11).
                common = min(values.shape[-1], teacher_tokens.shape[-1])
                # Layout is N,L,C; reshape to N*L,C and linearly resample the
                # channel axis down to the common width (keeps per-token rows).
                def _pool_channels(tokens: Any) -> Any:
                    if tokens.shape[-1] == common:
                        return tokens
                    flat = tokens.reshape(-1, 1, tokens.shape[-1])  # N*L,1,C
                    pooled = torch.nn.functional.interpolate(
                        flat, size=common, mode="linear", align_corners=False
                    )
                    return pooled.reshape(tokens.shape[0], tokens.shape[1], common)

                values_p = _pool_channels(values)
                teacher_p = _pool_channels(teacher_tokens)
                # Position-aligned token distance (N, L): value feature at
                # token l is regressed to the teacher value at the same l,
                # then attention m_ij scores each position per instance.
                token_mse = (values_p - teacher_p).square().mean(-1)  # N, L
                # m is (N, L, K): attention over tokens per instance.
                weighted = (m * token_mse.unsqueeze(-1)).sum(dim=1).mean(dim=1)
            else:
                token_mse = (values - teacher_tokens).square().mean(-1)  # N, L
                weighted = (m * token_mse.unsqueeze(-1)).sum(dim=1).mean(dim=1)
            losses.append(weighted.mean())
        loss = torch.stack(losses).mean()
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "feature_level_count": float(len(pairs)),
                "max_tokens": float(self.max_tokens),
            },
        )


# ---------------------------------------------------------------------------
# StructKD: Structural Knowledge Distillation (NeurIPS 2022, 18c0102cb...),
# Eq. 3a-4:  replace pixel-wise l_p with 1 - SSIM computed on Gaussian-weighted
# 11x11 patches with sigma=1.5, K1=0.01, K2=0.03, alpha=beta=gamma=1.0.
# ---------------------------------------------------------------------------
class StructuralSimilarityDistillationLoss(DistillationMechanismLoss):
    mechanism = "structural_similarity"

    def __init__(
        self,
        *,
        patch_size: int = 11,
        sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
    ) -> None:
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("SSIM patch size must be an odd number >= 3")
        self.patch_size = patch_size
        self.sigma = sigma
        self.k1 = k1
        self.k2 = k2

    def _gaussian_window(self, channels: int, device: Any, patch: int) -> Any:
        import torch

        coords = torch.arange(patch, dtype=torch.float32, device=device)
        coords -= (patch - 1) / 2.0
        gauss = torch.exp(-(coords**2) / (2 * self.sigma**2))
        gauss = (gauss / gauss.sum()).unsqueeze(0)
        window = gauss.t() @ gauss
        return window.expand(channels, 1, patch, patch).contiguous()

    def _ssim(self, student: Any, teacher: Any) -> Any:
        import torch

        channels = student.shape[1]
        # The paper fixes an 11x11 patch on COCO-scale maps; small synthetic
        # feature maps degrade the window to the largest valid odd size.
        min_side = min(student.shape[-2], student.shape[-1])
        if min_side < 3:
            # A 3x3 Gaussian window cannot fit a sub-3-pixel map (tiny smoke
            # fixtures).  Fall back to plain feature MSE so the mechanism
            # stays numerically defined; SSIM semantics resume on real maps.
            return (student - teacher).square().mean()
        patch = min(self.patch_size, min_side if min_side % 2 == 1 else min_side - 1)
        patch = max(patch, 3)
        window = self._gaussian_window(channels, student.device, patch)
        dynamic_range = 2.0  # normalized feature maps live in [-1, 1]
        c1 = (self.k1 * dynamic_range) ** 2
        c2 = (self.k2 * dynamic_range) ** 2
        mu_s = torch.nn.functional.conv2d(student, window, padding=0, groups=channels)
        mu_t = torch.nn.functional.conv2d(teacher, window, padding=0, groups=channels)
        mu_s_sq = mu_s.square()
        mu_t_sq = mu_t.square()
        mu_st = mu_s * mu_t
        sigma_s = torch.nn.functional.conv2d(
            student.square(), window, padding=0, groups=channels
        ) - mu_s_sq
        sigma_t = torch.nn.functional.conv2d(
            teacher.square(), window, padding=0, groups=channels
        ) - mu_t_sq
        sigma_st = torch.nn.functional.conv2d(
            student * teacher, window, padding=0, groups=channels
        ) - mu_st
        luminance = (2 * mu_st + c1) / (mu_s_sq + mu_t_sq + c1)
        contrast = (2 * sigma_st + c2) / (sigma_s + sigma_t + c2)
        structure = (sigma_st + c2 / 2) / (sigma_s.sqrt() * sigma_t.sqrt() + c2 / 2)
        ssim = luminance * contrast * structure
        return ((1.0 - ssim) / 2.0).mean()

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("structural distillation requires features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        for s_feat, t_feat in pairs:
            student = s_feat.float()
            teacher = t_feat.detach().float()
            if student.shape[1:] != teacher.shape[1:]:
                if student.shape[1] != teacher.shape[1]:
                    # Channel-agnostic SSIM: compare per-position statistics of
                    # channel-pooled energies (shared-primitive policy).
                    s_energy = student.square().mean(dim=1, keepdim=True)
                    t_energy = teacher.square().mean(dim=1, keepdim=True)
                    if s_energy.shape[-2:] != t_energy.shape[-2:]:
                        t_energy = torch.nn.functional.interpolate(
                            t_energy,
                            size=tuple(s_energy.shape[-2:]),
                            mode="bilinear",
                            align_corners=False,
                        )
                    losses.append(self._ssim(s_energy, t_energy))
                    continue
                teacher = torch.nn.functional.interpolate(
                    teacher, size=tuple(student.shape[2:]), mode="bilinear", align_corners=False
                )
            losses.append(self._ssim(student, teacher))
        return DistillationLossOutput(
            loss=torch.stack(losses).mean(),
            metrics={
                "feature_level_count": float(len(pairs)),
                "patch_size": float(self.patch_size),
            },
        )


# ---------------------------------------------------------------------------
# PGD: Prediction-Guided Distillation for Dense Object Detection (ECCV 2022,
# ecva:eccv2022:1356), Eq. 1-4:  quality q = 1[p] * p^alpha * IoU^beta,
# top-K pixels per GT, Gaussian MLE weights M = I / count, and feature/head
# distillation restricted to the masked foreground.  Adapted: quality from
# teacher logits (classification score) and teacher boxes (IoU proxy against
# student boxes), masking the MSE feature imitation.
# ---------------------------------------------------------------------------
class PredictionGuidedDistillationLoss(DistillationMechanismLoss):
    mechanism = "prediction_guided"

    def __init__(self, *, alpha: float = 0.8, beta: float = 1.0, epsilon: float = 1e-6) -> None:
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError("prediction guided exponents must be positive")
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("prediction guided distillation requires features")
        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("prediction guided distillation requires head logits")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        if teacher.ndim == 4:
            scores = torch.nn.functional.softmax(teacher.detach().float(), dim=1).amax(
                dim=1, keepdim=True
            )
            iou_proxy = torch.sigmoid(teacher.detach().float().mean(dim=1, keepdim=True))
        else:
            # 3D token-layout head output: factorize L into the closest square
            # grid (pad with zeros, neutral under amax/sigmoid averaging).
            t_f = teacher.detach().float()
            n, c, length = t_f.shape
            side = max(1, int(round(length ** 0.5)))
            if side * side != length:
                t_f = torch.nn.functional.pad(t_f, (0, side * side - length), value=0.0)
            grid = t_f.reshape(n, c, side, side)
            scores = torch.nn.functional.softmax(grid, dim=1).amax(dim=1, keepdim=True)
            iou_proxy = torch.sigmoid(grid.mean(dim=1, keepdim=True))
        quality = scores.pow(self.alpha) * iou_proxy.pow(self.beta)
        quality = quality / quality.amax().clamp_min(self.epsilon)
        masked_feature = student.new_zeros(())
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        count = 0
        for s_feat, t_feat in pairs:
            s = s_feat.float()
            t = t_feat.detach().float()
            if s.shape[1:] != t.shape[1:]:
                t = torch.nn.functional.interpolate(
                    t, size=tuple(s.shape[2:]), mode="bilinear", align_corners=False
                )
            w = quality
            if w.shape[-2:] != s.shape[-2:]:
                w = torch.nn.functional.interpolate(
                    w, size=s.shape[-2:], mode="bilinear", align_corners=False
                )
            # Channel alignment: mean-pool both tensors to the narrower channel
            # width (PGD's quality mask is channel-independent).
            common = min(s.shape[1], t.shape[1])
            s = _pool_channels_to(s, common)
            t = _pool_channels_to(t, common)
            normalizer = w.sum().clamp_min(self.epsilon)
            masked_feature = masked_feature + ((s - t).square() * w).sum() / normalizer
            count += 1
        masked_feature = masked_feature / max(count, 1)
        head_kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student, dim=1),
            torch.nn.functional.softmax(teacher.detach(), dim=1),
            reduction="batchmean",
        )
        loss = masked_feature + head_kd * quality.mean().detach()
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "masked_feature_loss": float(masked_feature.detach().cpu()),
                "head_kd_loss": float(head_kd.detach().cpu()),
                "mean_quality": float(quality.mean().cpu()),
            },
        )


# ---------------------------------------------------------------------------
# HEAD: HEtero-Assists Distillation (ECCV 2022, ecva:eccv2022:2285), Eq. 4-6:
#   AKD: assistant head mimics teacher head intermediate features (MSE)
#   CKD: teacher head directly supervises student head (MSE after adaptation)
#   L_HEAD = L_S_gt + L_A + L_C
# The YOLO26 graph owns the student head, so the assistant head is represented
# on the tensor surface: AKD reproduces teacher head features against the
# student backbone features (the assistant's role), CKD aligns student head
# outputs to the teacher head outputs.  Marked faithful_adaptation: the
# trainable assistant module is replaced by its tensor-surface effect.
# ---------------------------------------------------------------------------
class HeteroAssistDistillationLoss(DistillationMechanismLoss):
    mechanism = "hetero_assist"

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("hetero assist distillation requires features")
        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("hetero assist distillation requires head outputs")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        ckd = torch.nn.functional.mse_loss(student, teacher.detach())
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        akd_terms = []
        for s_feat, t_feat in pairs:
            s = s_feat.float()
            t = t_feat.detach().float()
            if s.shape[2:] != t.shape[2:]:
                t = torch.nn.functional.interpolate(
                    t, size=tuple(s.shape[2:]), mode="bilinear", align_corners=False
                )
            common = min(s.shape[1], t.shape[1])
            s = _pool_channels_to(s, common)
            t = _pool_channels_to(t, common)
            akd_terms.append(torch.nn.functional.mse_loss(s, t))
        akd = torch.stack(akd_terms).mean()
        loss = akd + ckd
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "akd_loss": float(akd.detach().cpu()),
                "ckd_loss": float(ckd.detach().cpu()),
                "feature_level_count": float(len(pairs)),
            },
        )


# ---------------------------------------------------------------------------
# GlobalKD: Distilling Object Detectors with Global Knowledge (ECCV 2022,
# ecva:eccv2022:2717), Eq. 1-6:  prototypes G are selected instances acting as
# common basis vectors of TS-space; the global knowledge is the projection
# coefficient alpha = P_G(f) and the loss aligns teacher/student coefficients:
#   L_global = 1/(2N) sum_i sum_j omega_ij || alpha_s - alpha_t ||^2
# We compute projections via cosine similarity to K prototype instances mined
# from the teacher tokens (top-energy positions), weights from reliability.
# ---------------------------------------------------------------------------
class GlobalPrototypeDistillationLoss(DistillationMechanismLoss):
    mechanism = "global_prototype"

    def __init__(self, *, prototypes: int = 16, max_tokens: int = 128) -> None:
        if prototypes < 2:
            raise ValueError("global prototype distillation needs >= 2 prototypes")
        self.prototypes = prototypes
        self.max_tokens = max_tokens

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("global prototype distillation requires features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        for s_feat, t_feat in pairs:
            student = s_feat.float()
            teacher = t_feat.detach().float()
            if student.shape[1:] != teacher.shape[1:]:
                teacher = torch.nn.functional.interpolate(
                    teacher, size=tuple(student.shape[2:]), mode="bilinear", align_corners=False
                )
            n, c, h, w = student.shape
            s_tokens = torch.nn.functional.normalize(
                student.flatten(2).transpose(1, 2), dim=-1
            )
            t_tokens = torch.nn.functional.normalize(
                teacher.flatten(2).transpose(1, 2), dim=-1
            )
            if s_tokens.shape[1] > self.max_tokens:
                scale = math.sqrt(self.max_tokens / s_tokens.shape[1])
                size = (max(1, int(h * scale)), max(1, int(w * scale)))
                s_small = torch.nn.functional.adaptive_avg_pool2d(student, size)
                t_small = torch.nn.functional.adaptive_avg_pool2d(teacher, size)
                s_tokens = torch.nn.functional.normalize(
                    s_small.flatten(2).transpose(1, 2), dim=-1
                )
                t_tokens = torch.nn.functional.normalize(
                    t_small.flatten(2).transpose(1, 2), dim=-1
                )
            k = min(self.prototypes, t_tokens.shape[1])
            energy = t_tokens.norm(dim=-1)  # prototype mining by energy ranking
            proto_idx = energy.topk(k, dim=1).indices
            gathered = t_tokens.gather(
                1, proto_idx.unsqueeze(-1).expand(-1, -1, t_tokens.shape[-1])
            )
            # Both detector tokens must be projected onto the SAME prototype
            # dictionary; without a trainable adapter the student channel axis
            # is resampled to the teacher width (Eq. 5 projection alignment).
            if s_tokens.shape[-1] != t_tokens.shape[-1]:
                flat = s_tokens.reshape(-1, 1, s_tokens.shape[-1])
                s_tokens = (
                    torch.nn.functional.interpolate(
                        flat, size=t_tokens.shape[-1], mode="linear", align_corners=False
                    )
                    .reshape(s_tokens.shape[0], s_tokens.shape[1], t_tokens.shape[-1])
                )
                s_tokens = torch.nn.functional.normalize(s_tokens, dim=-1)
            alpha_t = torch.einsum("nlc,nkc->nlk", t_tokens, gathered)
            alpha_s = torch.einsum("nlc,nkc->nlk", s_tokens, gathered)
            losses.append((alpha_s - alpha_t.detach()).square().mean())
        return DistillationLossOutput(
            loss=torch.stack(losses).mean(),
            metrics={
                "prototype_count": float(self.prototypes),
                "feature_level_count": float(len(pairs)),
            },
        )


# ---------------------------------------------------------------------------
# MFDC: Multi-faceted Distillation of Base-Novel Commonality (ECCV 2022,
# ecva:eccv2022:3523), Eq. 1-5:  memory-bank class prototypes; similarity of a
# proposal to novel-class prototypes becomes soft labels; recognition
# commonality distillation minimizes KL(q_cls || p_cls).  Localization and
# distribution commonalities need per-class regressors/statistics that YOLO26
# does not expose on this tensor surface; they are recorded as approximated.
# ---------------------------------------------------------------------------
class BaseNovelCommonalityDistillationLoss(DistillationMechanismLoss):
    mechanism = "base_novel_commonality"

    def __init__(self, *, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError("base-novel commonality temperature must be positive")
        self.temperature = temperature

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("base-novel commonality distillation requires logits")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        t_prob = torch.nn.functional.softmax(teacher.detach().float() / self.temperature, dim=1)
        # Memory-bank proxy: each teacher class-channel map (L2-normalized over
        # its spatial support) is a class prototype template; the similarity of
        # every student channel template to every teacher template forms the
        # prototype-similarity matrix (Eq. 2), and softmax over teacher
        # prototypes yields the soft labels for the recognition-commonality
        # KL (Eq. 3).
        def _template_maps(t: Any) -> Any:
            # 4D (N,C,H,W): normalize over spatial dims; 3D (N,C,L): pad the
            # linear index to a square grid first, then normalize spatially.
            if t.ndim == 4:
                return torch.nn.functional.normalize(t, dim=(2, 3))
            n, c, length = t.shape
            side = max(1, int(round(length ** 0.5)))
            if side * side != length:
                t = torch.nn.functional.pad(t, (0, side * side - length), value=0.0)
            return torch.nn.functional.normalize(
                t.reshape(n, c, side, side), dim=(2, 3)
            )

        s_maps = _template_maps(student.float())
        t_maps = _template_maps(teacher.detach().float())
        similarity = torch.einsum("nxhw,nyhw->nxy", s_maps, t_maps)
        q = torch.nn.functional.softmax(similarity / self.temperature, dim=2)
        aligned_q = q.mean(dim=1)  # prototype-label distribution (N, C)
        log_p_proto = torch.nn.functional.log_softmax(
            student.float().mean(dim=tuple(range(2, student.ndim))) / self.temperature,
            dim=1,
        )
        kl = (aligned_q * (aligned_q.clamp_min(1e-8).log() - log_p_proto)).sum(dim=1).mean()
        log_p = torch.nn.functional.log_softmax(student.float() / self.temperature, dim=1)
        cls_kd = torch.nn.functional.kl_div(
            log_p, t_prob, reduction="batchmean"
        ) * (self.temperature**2)
        loss = kl + cls_kd
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "prototype_similarity_kl": float(kl.detach().cpu()),
                "cls_kd_loss": float(cls_kd.detach().cpu()),
                "temperature": self.temperature,
            },
        )


# ---------------------------------------------------------------------------
# PA-BoVW: Few-Shot Object Detection by Knowledge Distillation Using
# Bag-of-Visual-Words Representations (ECCV 2022, ecva:eccv2022:6004),
# Eq. 8-9:  cosine similarity between detector features and the visual-word
# vocabulary must match the pre-learned BoVW encoding; L1 distance between the
# two K-dimension similarity maps is the distillation loss.
# ---------------------------------------------------------------------------
class BoVWConsistencyDistillationLoss(DistillationMechanismLoss):
    mechanism = "bovw_consistency"

    def __init__(self, *, vocabulary: int = 64) -> None:
        if vocabulary < 2:
            raise ValueError("bovw vocabulary must contain at least two words")
        self.vocabulary = vocabulary

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("bovw consistency distillation requires features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        losses = []
        for s_feat, t_feat in pairs:
            student = s_feat.float()
            teacher = t_feat.detach().float()
            if student.shape[1:] != teacher.shape[1:]:
                teacher = torch.nn.functional.interpolate(
                    teacher, size=tuple(student.shape[2:]), mode="bilinear", align_corners=False
                )
            n, c, h, w = student.shape
            k = min(self.vocabulary, h * w)
            # Vocabulary: the K top-energy teacher spatial positions act as
            # visual words (per-image, as in the paper's online encoding);
            # a word is the L2-normalized channel vector at that position.
            s_tokens = torch.nn.functional.normalize(
                student.flatten(2).transpose(1, 2), dim=-1
            )  # N, HW_s, C_s
            t_tokens = torch.nn.functional.normalize(
                teacher.flatten(2).transpose(1, 2), dim=-1
            )  # N, HW_t, C_t
            energy = t_tokens.norm(dim=-1)
            word_idx = energy.topk(k, dim=1).indices  # N, K
            words = t_tokens.gather(1, word_idx.unsqueeze(-1).expand(-1, -1, t_tokens.shape[-1]))
            # Both similarity maps must share the word width; the student
            # channel axis is resampled to the teacher width when they differ
            # (no trainable adapter, shared-primitive policy).
            if s_tokens.shape[-1] != words.shape[-1]:
                flat = s_tokens.reshape(-1, 1, s_tokens.shape[-1])
                s_tokens = (
                    torch.nn.functional.interpolate(
                        flat, size=words.shape[-1], mode="linear", align_corners=False
                    )
                    .reshape(n, -1, words.shape[-1])
                )
                s_tokens = torch.nn.functional.normalize(s_tokens, dim=-1)
            q_student = torch.einsum("nlc,nkc->nlk", s_tokens, words)  # N, HW_s, K
            q_teacher = torch.einsum("nlc,nkc->nlk", t_tokens, words)  # N, HW_t, K
            if q_student.shape[1] != q_teacher.shape[1]:
                q_teacher = torch.nn.functional.interpolate(
                    q_teacher.transpose(1, 2).unsqueeze(-1),
                    size=q_student.shape[1],
                    mode="linear",
                    align_corners=False,
                ).squeeze(-1).transpose(1, 2)
            losses.append(torch.nn.functional.l1_loss(q_student, q_teacher))
        return DistillationLossOutput(
            loss=torch.stack(losses).mean(),
            metrics={
                "vocabulary_size": float(min(self.vocabulary, pairs[0][0].shape[1])),
                "feature_level_count": float(len(pairs)),
            },
        )


# ---------------------------------------------------------------------------
# GLAMD: Global and Local Attention Mask Distillation (ECCV 2022,
# ecva:eccv2022:6328), Eq. 1, 8, 9, 10:  channel mask from spatial-absolute
# softmax with temperature, global spatial attention, plus local (patched)
# channel attention; L_cat + L_sat = L_at.  Patch division implemented with
# adaptive average pooling to `patch_grid` per side.
# ---------------------------------------------------------------------------
class GLAMAttentionDistillationLoss(DistillationMechanismLoss):
    mechanism = "glam_attention"

    def __init__(self, *, patch_grid: int = 2, temperature: float = 0.5, epsilon: float = 1e-6) -> None:
        if patch_grid < 1:
            raise ValueError("glam patch grid must be >= 1")
        if temperature <= 0.0:
            raise ValueError("glam temperature must be positive")
        self.patch_grid = patch_grid
        self.temperature = temperature
        self.epsilon = epsilon

    def _channel_mask(self, feature: Any) -> Any:
        import torch

        avg = feature.abs().mean(dim=(-2, -1))  # N, C
        return torch.nn.functional.softmax(avg / self.temperature, dim=1)

    def _spatial_mask(self, feature: Any) -> Any:
        import torch

        return torch.nn.functional.normalize(
            feature.float().abs().mean(dim=1).flatten(1), dim=1
        )

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("glam distillation requires features")
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features)
        cat_losses = []
        sat_losses = []
        for s_feat, t_feat in pairs:
            student = s_feat.float()
            teacher = t_feat.detach().float()
            if student.shape[1:] != teacher.shape[1:]:
                teacher = torch.nn.functional.interpolate(
                    teacher, size=tuple(student.shape[2:]), mode="bilinear", align_corners=False
                )
            # Global channel attention feature: Ac = mean over channels.
            ac_s = student.mean(dim=1, keepdim=True)
            ac_t = teacher.mean(dim=1, keepdim=True)
            global_channel = torch.nn.functional.mse_loss(ac_s, ac_t)
            # Local channel attention: patch-wise means, N patches.
            g = self.patch_grid
            if g > 1:
                s_patches = torch.nn.functional.adaptive_avg_pool2d(ac_s, g)
                t_patches = torch.nn.functional.adaptive_avg_pool2d(ac_t, g)
                local_channel = torch.nn.functional.mse_loss(s_patches, t_patches)
            else:
                local_channel = student.new_zeros(())
            cat = 0.5 * (global_channel + local_channel)
            # Spatial attention (global only per the paper).
            as_s = self._spatial_mask(student)
            as_t = self._spatial_mask(teacher)
            sat = torch.nn.functional.mse_loss(as_s, as_t)
            cat_losses.append(cat)
            sat_losses.append(sat)
        cat_loss = torch.stack(cat_losses).mean()
        sat_loss = torch.stack(sat_losses).mean()
        return DistillationLossOutput(
            loss=cat_loss + sat_loss,
            metrics={
                "channel_attention_loss": float(cat_loss.detach().cpu()),
                "spatial_attention_loss": float(sat_loss.detach().cpu()),
                "patch_grid": float(self.patch_grid),
            },
        )


# ---------------------------------------------------------------------------
# DLIM-Det: Distilling Knowledge from Large-Scale Image Models for Object
# Detection (ECCV 2024, ecva:eccv2024:11200), Eq. 5-6:  Query Position Distill
# L_QPD = sum lambda1*L1(b_t, b_s) + lambda2*G-IoU with lambda1=5, lambda2=2,
# and Query Relation Distill L_QRD = 1/N sum |A_t - A_s| over self-attention
# maps after one-to-one query matching.  The YOLO26 tensor surface has no
# DETR object queries; the adapted contract matches dense head positions as
# "queries" (top-K teacher logits) and aligns decoded boxes plus feature
# self-attention proxies.  Marked faithful_adaptation.
# ---------------------------------------------------------------------------
class QueryDistillationLoss(DistillationMechanismLoss):
    mechanism = "query_distillation"

    def __init__(self, *, lambda_position: float = 5.0, lambda_giou: float = 2.0) -> None:
        if lambda_position <= 0.0 or lambda_giou <= 0.0:
            raise ValueError("query distillation weights must be positive")
        self.lambda_position = lambda_position
        self.lambda_giou = lambda_giou

    @staticmethod
    def _pairwise(tokens: Any) -> Any:
        import torch

        normalized = torch.nn.functional.normalize(tokens, dim=-1)
        return torch.bmm(normalized, normalized.transpose(1, 2))

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("query distillation requires head logits")
        if inputs.student_boxes is None or inputs.teacher_boxes is None:
            raise ValueError("query distillation requires decoded boxes")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        s_boxes, t_boxes = _resolve_box_tensors(inputs)
        # One-to-one matching: teacher queries rank by score; student positions
        # align to the same indices (identity permutation on the dense surface).
        position_loss = torch.nn.functional.l1_loss(s_boxes, t_boxes.detach())
        # GIoU-style penalty on normalized box overlap proxy.  Boxes are
        # xyxy; slice to the first 4 channels before the coordinate split.
        s4 = s_boxes[..., :4] if s_boxes.shape[-1] >= 4 else s_boxes
        t4 = t_boxes[..., :4] if t_boxes.shape[-1] >= 4 else t_boxes
        center_s = (s4[..., ::2] + s4[..., 1::2]) * 0.5
        center_t = (t4[..., ::2] + t4[..., 1::2]) * 0.5
        giou_proxy = 1.0 - torch.nn.functional.cosine_similarity(
            center_s.flatten(1), center_t.detach().flatten(1), dim=1
        ).mean()
        qpd = self.lambda_position * position_loss + self.lambda_giou * giou_proxy
        # Relation distillation over token self-attention proxies.
        pairs = _feature_pairs(inputs.student_features, inputs.teacher_features) if (
            inputs.student_features is not None and inputs.teacher_features is not None
        ) else []
        qrd_terms = []
        for s_feat, t_feat in pairs:
            s = s_feat.float()
            t = t_feat.detach().float()
            if s.shape[2:] != t.shape[2:]:
                t = torch.nn.functional.interpolate(
                    t, size=tuple(s.shape[2:]), mode="bilinear", align_corners=False
                )
            n, c, h, w = s.shape
            side = max(4, int(math.sqrt(min(h * w, 64))))
            s_tokens = torch.nn.functional.adaptive_avg_pool2d(s, (side, side)).flatten(2).transpose(1, 2)
            t_tokens = torch.nn.functional.adaptive_avg_pool2d(t, (side, side)).flatten(2).transpose(1, 2)
            qrd_terms.append(
                torch.nn.functional.l1_loss(self._pairwise(s_tokens), self._pairwise(t_tokens))
            )
        qrd = torch.stack(qrd_terms).mean() if qrd_terms else student.new_zeros(())
        loss = qpd + qrd
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "query_position_loss": float(position_loss.detach().cpu()),
                "query_relation_loss": float(qrd.detach().cpu()),
                "lambda_position": self.lambda_position,
                "lambda_giou": self.lambda_giou,
            },
        )


# ---------------------------------------------------------------------------
# MSCD: Multi-scale Cross Distillation for Object Detection in Aerial Images
# (ECCV 2024, ecva:eccv2024:6619), Eq. 5-6:  self-distillation across scale
# branches on RoI features with adaptive weight
#   w_i = max(0, L_det(y_i, y_hat) - L_det(y_m_i, y_hat))
# positive and negative proposals accumulated separately.  On the shared
# surface the multi-scale branch signal is the teacher feature (the strong
# scale branch), and the detection-loss delta weight uses the head logits.
# ---------------------------------------------------------------------------
class CrossScaleSelfDistillationLoss(DistillationMechanismLoss):
    mechanism = "cross_scale_self"

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_features is None or inputs.teacher_features is None:
            raise ValueError("cross-scale self distillation requires features")
        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("cross-scale self distillation requires head logits")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        det_student = torch.nn.functional.binary_cross_entropy_with_logits(
            student.sigmoid().mean(dim=1), (teacher.sigmoid().mean(dim=1) > 0.5).float()
        )
        det_branch = torch.nn.functional.binary_cross_entropy_with_logits(
            teacher.sigmoid().mean(dim=1), (teacher.sigmoid().mean(dim=1) > 0.5).float()
        )
        weight = (det_student - det_branch.detach()).clamp_min(0.0)
        feature_terms = []
        for s_feat, t_feat in _feature_pairs(
            inputs.student_features, inputs.teacher_features
        ):
            s = s_feat.float()
            t = t_feat.detach().float()
            if s.shape[2:] != t.shape[2:]:
                t = torch.nn.functional.interpolate(
                    t, size=tuple(s.shape[2:]), mode="bilinear", align_corners=False
                )
            common = min(s.shape[1], t.shape[1])
            s = _pool_channels_to(s, common)
            t = _pool_channels_to(t, common)
            feature_terms.append((weight.detach() * (s - t).square().mean(dim=(1, 2, 3))).mean())
        loss = torch.stack(feature_terms).mean()
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "adaptive_weight_mean": float(weight.detach().mean().cpu()),
                "feature_level_count": float(
                    len(_feature_pairs(inputs.student_features, inputs.teacher_features))
                ),
            },
        )


# ---------------------------------------------------------------------------
# ELDET: Early-Learning Distillation with Noisy Labels (NeurIPS 2025,
# 6460e378...), Eq. 3-5:  L_kd = L_kd_cls + L_kd_loc between the early-learning
# teacher and the student, L_total = L_det + lambda * L_kd, teacher updated by
# EMA with a small decay on the classification head.  CrossKD head transfer is
# approximated by standard response KD on the shared surface (faithful
# adaptation; the cross-head module itself is runtime-specific).
# ---------------------------------------------------------------------------
class EarlyLearningDistillationLoss(DistillationMechanismLoss):
    mechanism = "early_learning"

    def __init__(self, *, temperature: float = 2.0) -> None:
        if temperature <= 0.0:
            raise ValueError("early learning temperature must be positive")
        self.temperature = temperature

    def compute(self, inputs: DistillationInputs) -> DistillationLossOutput:
        import torch

        if inputs.student_logits is None or inputs.teacher_logits is None:
            raise ValueError("early learning distillation requires logits")
        student, teacher = _same_shape(inputs.student_logits, inputs.teacher_logits)
        temperature = self.temperature
        cls_kd = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student.float() / temperature, dim=1),
            torch.nn.functional.softmax(teacher.detach().float() / temperature, dim=1),
            reduction="batchmean",
        ) * (temperature**2)
        loc_kd = student.new_zeros(())
        if inputs.student_boxes is not None and inputs.teacher_boxes is not None:
            s_boxes, t_boxes = _resolve_box_tensors(inputs)
            loc_kd = torch.nn.functional.smooth_l1_loss(s_boxes, t_boxes.detach())
        loss = cls_kd + loc_kd
        return DistillationLossOutput(
            loss=loss,
            metrics={
                "cls_kd_loss": float(cls_kd.detach().cpu()),
                "loc_kd_loss": float(loc_kd.detach().cpu()),
                "temperature": temperature,
            },
        )


PAPER_MECHANISM_LOSSES: dict[str, type[DistillationMechanismLoss]] = {
    loss.mechanism: loss
    for loss in (
        PearsonFeatureDistillationLoss,
        RichnessMaskedDistillationLoss,
        ClassifierResponseDistillationLoss,
        InstanceConditionalDistillationLoss,
        StructuralSimilarityDistillationLoss,
        PredictionGuidedDistillationLoss,
        HeteroAssistDistillationLoss,
        GlobalPrototypeDistillationLoss,
        BaseNovelCommonalityDistillationLoss,
        BoVWConsistencyDistillationLoss,
        GLAMAttentionDistillationLoss,
        QueryDistillationLoss,
        CrossScaleSelfDistillationLoss,
        EarlyLearningDistillationLoss,
    )
}


__all__ = [
    "PAPER_MECHANISM_LOSSES",
    "BaseNovelCommonalityDistillationLoss",
    "BoVWConsistencyDistillationLoss",
    "ClassifierResponseDistillationLoss",
    "CrossScaleSelfDistillationLoss",
    "EarlyLearningDistillationLoss",
    "GLAMAttentionDistillationLoss",
    "GlobalPrototypeDistillationLoss",
    "HeteroAssistDistillationLoss",
    "InstanceConditionalDistillationLoss",
    "PearsonFeatureDistillationLoss",
    "PredictionGuidedDistillationLoss",
    "QueryDistillationLoss",
    "RichnessMaskedDistillationLoss",
    "StructuralSimilarityDistillationLoss",
]
