import torch
import torch.nn as nn
import torch.nn.functional as F


# Focal Loss 구현
# 이는 불균형한 데이터셋에서 사용되며, 잘못 분류된 샘플에 더 많은 중요도를 부여한다.
# https://discuss.pytorch.org/t/is-this-a-correct-implementation-for-focal-loss-in-pytorch/43327/8
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction="mean"):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction,
        )


# Label Smoothing Loss 구현
# 모델이 너무 자신만만하게 예측하는 것을 방지하기 위해 사용된다.
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes=3, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# F1 Score 손실 함수 구현
# F1 Score는 precision과 recall의 조화 평균이며, 이를 손실로 사용한다.
# https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
class F1Loss(nn.Module):
    def __init__(self, classes=3, epsilon=1e-7):
        super().__init__()
        self.classes = classes
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, self.classes).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean()


# 사용 가능한 손실 함수의 진입점
_criterion_entrypoints = {
    "cross_entropy": nn.CrossEntropyLoss,
    "focal": FocalLoss,
    "label_smoothing": LabelSmoothingLoss,
    "f1": F1Loss,
}


def criterion_entrypoint(criterion_name):
    """
    주어진 손실 함수 이름에 해당하는 손실 함수

    Args:
        criterion_name (str): 반환할 손실 함수 이름

    Returns:
        callable: 주어진 이름에 해당하는 손실 함수
    """
    return _criterion_entrypoints[criterion_name]


def is_criterion(criterion_name):
    """
    주어진 손실 함수 이름이 지원되는지 확인한다.

    Args:
        criterion_name (str): 확인할 손실 함수 이름

    Returns:
        bool: 지원되면 True, 그렇지 않으며 False
    """
    return criterion_name in _criterion_entrypoints


def create_criterion(criterion_name, **kwargs):
    """
    지정된 인수를 사용하여 손실 함수 객체를 생성한다.

    Args:
        criterion_name (str): 생성할 손실 함수 이름
        **kargs: 손실 함수 생성자에 전달된 키워드 인자

    Returns:
        nn.Module: 생성된 손실 함수 객체
    """
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError("Unknown loss (%s)" % criterion_name)
    return criterion
