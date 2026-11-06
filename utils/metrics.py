"""Original evaluation definitions: global PSNR, SSIM, SAM (degrees), ERGAS."""
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def _validate_pair(pred: torch.Tensor, target: torch.Tensor):
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have identical shapes, "
            f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    if pred.dim() != 4:
        raise ValueError(
            f"HSI image metrics expect [B,C,H,W], got {tuple(pred.shape)}"
        )

def psnr_calculator(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    高光谱 PSNR：将整个 [B,C,H,W] 高光谱数据整体计算 MSE，
    再基于该整体 MSE 计算一个 PSNR。

        MSE = mean((pred - target)^2)
        PSNR = 10 * log10(data_range^2 / MSE)

    不再逐通道分别计算 PSNR 后取平均。

    当前预处理数据范围为 [0,1]，因此默认 data_range=1.0。
    """
    _validate_pair(pred, target)

    pred = pred.float()
    target = target.float()

    mse = torch.mean((pred - target) ** 2)
    dr2 = torch.as_tensor(
        float(data_range) ** 2,
        device=mse.device,
        dtype=mse.dtype
    )

    if mse <= eps:
        return torch.full_like(mse, float('inf'))

    return 10.0 * torch.log10(
        dr2 / torch.clamp(mse, min=eps)
    )

def ssim_calculator(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> torch.Tensor:
    """
    高光谱 SSIM：直接将完整 [B,C,H,W] 高光谱图像输入 SSIM，
    由 torchmetrics 将光谱维 C 作为图像通道整体计算一个 SSIM。

    不再逐光谱通道单独计算 SSIM 后取平均。

    当前预处理数据范围为 [0,1]，因此默认 data_range=1.0。
    """
    _validate_pair(pred, target)

    pred = pred.float()
    target = target.float()

    metric = StructuralSimilarityIndexMeasure(
        data_range=float(data_range)
    ).to(pred.device)

    return metric(pred, target)

def sam_calculator(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    degree: bool = True
) -> torch.Tensor:
    """计算两个高光谱图像之间的平均 Spectral Angle Mapper (SAM)。"""
    pred = pred.float()
    target = target.float()

    original_dim = pred.dim()

    if original_dim == 4:  # [B,C,H,W] -> [BHW,C]
        pred = pred.permute(0, 2, 3, 1).reshape(-1, pred.shape[1])
        target = target.permute(0, 2, 3, 1).reshape(-1, target.shape[1])

    elif original_dim == 2:
        pred = pred.reshape(-1, pred.shape[1])
        target = target.reshape(-1, target.shape[1])

    elif original_dim == 1:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)

    else:
        raise ValueError(
            f"输入维度 {original_dim} 不支持，必须为 1, 2 或 4"
        )

    dot_product = (pred * target).sum(dim=1)
    pred_norm = torch.norm(pred, p=2, dim=1)
    target_norm = torch.norm(target, p=2, dim=1)

    cos_sim = dot_product / (
        pred_norm * target_norm + eps
    )

    cos_sim = torch.clamp(
        cos_sim,
        -1.0,
        1.0
    )

    sam = torch.acos(cos_sim)

    if degree:
        sam = sam * (
            180.0 / torch.pi
        )

    return sam.mean()

def ergas_calculator(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_factor
) -> torch.Tensor:
    """计算高光谱 ERGAS。"""
    _validate_pair(pred, target)

    _, C, _, _ = pred.shape

    mse_per_channel = torch.mean(
        (pred - target) ** 2,
        dim=(2, 3)
    )

    mean_target_per_channel = torch.mean(
        target,
        dim=(2, 3)
    )

    mean_target_per_channel = torch.clamp(
        mean_target_per_channel,
        min=1e-10
    )

    r = scale_factor

    ergas = (
        100.0 / r
    ) * torch.sqrt(
        (1.0 / C)
        * torch.sum(
            mse_per_channel /
            (mean_target_per_channel ** 2),
            dim=1
        )
    )

    return ergas.mean()

@torch.no_grad()
def evaluate(sr, gt, scale):
    sr, gt = sr.clamp(0, 1), gt.clamp(0, 1)
    _validate_pair(sr, gt)
    return {"psnr": psnr_calculator(sr, gt).item(), "ssim": ssim_calculator(sr, gt).item(),
            "sam": sam_calculator(sr, gt).item(), "ergas": ergas_calculator(sr, gt, scale).item()}
