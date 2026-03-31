import torch
import torch.nn as nn
from torch.nn import functional as F
import utils
from einops import rearrange


class CriterionAD(nn.Module):
    """Cosine-normalized affinity distillation loss."""

    def __init__(
        self,
        window_size=None,
        eps: float = 1e-6,
        reduction: str = "mean",
        divide_by_batch: bool = True,
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none")
        self.window_size = window_size
        self.eps = eps
        self.reduction = reduction
        self.divide_by_batch = divide_by_batch

    def forward(self, feat_teacher, feat_student) -> torch.Tensor:
        assert feat_student.shape == feat_teacher.shape, "student/teacher must both be [B, C, H, W]"
        B, _, H, W = feat_student.shape

        feat_teacher = feat_teacher.detach()

        if self.window_size is None:
            p1, p2 = H, W
        else:
            p1, p2 = self.window_size
            if (H % p1) != 0 or (W % p2) != 0:
                raise ValueError(f"H={H}, W={W} must be divisible by window_size={self.window_size}")

        S = rearrange(feat_student, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=p1, p2=p2)
        T = rearrange(feat_teacher, "b c (h p1) (w p2) -> (b h w) (p1 p2) c", p1=p1, p2=p2)

        S = F.normalize(S, p=2, dim=-1, eps=self.eps)
        T = F.normalize(T, p=2, dim=-1, eps=self.eps)

        A_s = torch.einsum("bid,bjd->bij", S, S)
        A_t = torch.einsum("bid,bjd->bij", T, T)
        loss_elts = torch.abs(A_s - A_t)

        if self.reduction == "mean":
            loss = loss_elts.mean()
        elif self.reduction == "sum":
            loss = loss_elts.sum()
        else:
            loss = loss_elts

        if self.divide_by_batch and self.reduction != "none":
            loss = loss / B

        return loss


class ChannelNorm(nn.Module):
    def __init__(self):
        super(ChannelNorm, self).__init__()

    def forward(self, featmap):
        n, c, _, _ = featmap.shape
        featmap = featmap.reshape((n, c, -1))
        featmap = featmap.softmax(dim=-1)
        return featmap


class CriterionCWD(nn.Module):
    def __init__(self, norm_type="none", temperature=4.0):
        super(CriterionCWD, self).__init__()

        if norm_type == "channel":
            self.normalize = ChannelNorm()
        else:
            self.normalize = None

        self.norm_type = norm_type
        self.default_temperature = temperature
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.temperature = temperature

    def forward(self, preds_S, preds_T, stage):
        n, c, h, w = preds_S.shape
        if stage == "stage1":
            temperature = 2.0
        elif stage == "stage2":
            temperature = 3.0
        elif stage == "stage3":
            temperature = 3.0
        else:
            temperature = self.default_temperature

        if self.normalize is not None:
            norm_s = self.normalize(preds_S / temperature)
            norm_t = self.normalize(preds_T.detach() / temperature)
        else:
            norm_s = preds_S[0]
            norm_t = preds_T[0].detach()

        norm_s = norm_s.log()
        loss = self.criterion(norm_s, norm_t)

        if self.norm_type in ("channel", "channel_mean"):
            loss /= n * c
        else:
            loss /= n * h * w

        return loss * (temperature**2)


class CriterionCE(nn.Module):
    """Cross-entropy loss for semantic segmentation."""

    def __init__(self, ignore_index=255, use_weight=True):
        super(CriterionCE, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="mean")

    def forward(self, preds, target):
        target = target[:, 0].unsqueeze(1)
        if torch.all(preds >= 0) and torch.all(preds <= 1):
            raise ValueError("Error: preds should be raw logits, not probabilities. Remove softmax from model output.")
        loss = self.criterion(preds, target)
        return loss


class CriterionBCE(nn.Module):
    """Binary cross-entropy loss for semantic segmentation (C=1)."""

    def __init__(self, ignore_index=255):
        super(CriterionBCE, self).__init__()
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        target = target[:, 0].unsqueeze(1)
        target = target.float()
        if torch.all(preds >= 0) and torch.all(preds <= 1):
            raise ValueError("Error: preds should be raw logits, not probabilities. Remove softmax from model output.")
        loss = self.criterion(preds, target)
        return loss


class CriterionBMSE(nn.Module):
    def __init__(self):
        super(CriterionBMSE, self).__init__()

    def forward(self, pred, gt, valid=None):
        pos_mask = gt.gt(0.5).float()
        neg_mask = gt.lt(0.5).float()

        if valid is None:
            valid = torch.ones_like(pos_mask)

        mse_loss = F.mse_loss(pred, gt, reduction="none")
        pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask * valid)
        neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask * valid)

        loss = (pos_loss + neg_loss) * 0.5
        return loss


class CriterionFL(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction="mean", trainable=False):
        super(CriterionFL, self).__init__()
        self.alpha = alpha
        self.trainable = trainable
        if trainable:
            self.gamma_param = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        else:
            self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        if self.trainable:
            gamma = F.softplus(self.gamma_param)
        else:
            gamma = self.gamma

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** gamma * BCE_loss

        if self.reduction == "mean":
            return F_loss.mean()
        elif self.reduction == "sum":
            return F_loss.sum()
        else:
            return F_loss


class FocalLoss2(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss2, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)

        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))

        target = target.view(-1, 1).long()

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
