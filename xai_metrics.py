# 2024 Ricardo Cruz <ricardo.pdm.cruz@gmail.com> 
# Implementation of interpretability metrics.

from torcheval import metrics
import torch

class MyMetric(metrics.Metric):
    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.num = self.den = 0

    def merge_state(self, metrics):
        for metric in metrics:
            self.num += metric.num
            self.den += metric.den
        return self

    def compute(self):
        return self.num / self.den

class PointingGame(MyMetric):
    # https://link.springer.com/article/10.1007/s11263-017-1059-x
    def update(self, heatmaps, masks):
        heatmaps = torch.nn.functional.interpolate(heatmaps[:, None], masks.shape[-2:], mode='bilinear')
        heatmaps = heatmaps.view(len(heatmaps), -1)
        masks = masks.view(len(masks), -1)
        self.num += sum(masks[range(len(masks)), torch.argmax(heatmaps, 1)] != 0)
        self.den += len(heatmaps)

class DegradationScore(MyMetric):
    # if too slow, resize the heatmaps you provide to 7x7 or 8x8
    def __init__(self, model, score='acc'):
        super().__init__()
        assert score in ['acc'], f'Unknown score "{score}"'
        self.model = model
        if score == 'acc':
            self.score = lambda ypred, y: (ypred == y).float()

    def update(self, images, true_classes, heatmaps):
        lerf = self.degradation_curve('lerf', self.model, self.score, images, true_classes, heatmaps)
        morf = self.degradation_curve('morf', self.model, self.score, images, true_classes, heatmaps)
        self.num += torch.sum(torch.mean(lerf - morf, 1))
        self.den += len(images)

    def degradation_curve(self, curve_type, model, score_fn, images, true_classes, heatmaps):
        # Given an explanation map, occlude by 8x8 creating two curves: least
        # relevant removed first (LeRF) and most relevant removed first (MoRF),
        # where the score is computed for a given metric. The result is the area
        # between the two curves.
        # Schulz, Karl, et al. "Restricting the flow: Information bottlenecks for
        # attribution." arXiv preprint arXiv:2001.00396 (2020).
        assert curve_type in ['lerf', 'morf']
        assert len(images.shape) == 4
        assert len(heatmaps.shape) == 3
        descending = curve_type == 'morf'
        ix = torch.argsort(heatmaps.view(len(heatmaps), -1), descending=descending)[:, :-1]
        cc = ix % heatmaps.shape[1]
        rr = ix // heatmaps.shape[1]
        xscale = images.shape[3] // heatmaps.shape[2]
        yscale = images.shape[2] // heatmaps.shape[1]
        occlusions = images.clone()
        # in the past, I did this in a single forward pass to make it faster,
        # but due to memory contraints, it's better to call as a cycle.
        scores = []
        for c, r in zip(cc.T, rr.T):
            for i, (ci, ri) in enumerate(zip(c, r)):
                occlusions[i, :, ci*yscale:(ci+1)*yscale, ri*xscale:(ri+1)*xscale] = 0
            with torch.no_grad():
                ypred = model(occlusions)
            ypred = ypred.argmax(1)
            score = score_fn(ypred, true_classes)
            scores.append(score)
        return torch.stack(scores, 1)

class BalancedDegradationScore(DegradationScore):
    # balanced version of the degradation score
    def __init__(self, model, num_classes, score='acc'):
        super().__init__(model, score)
        self.num = torch.zeros(num_classes)
        self.den = torch.zeros(num_classes)

    def update(self, images, true_classes, heatmaps):
        lerf = self.degradation_curve('lerf', self.model, self.score, images, true_classes, heatmaps)
        morf = self.degradation_curve('morf', self.model, self.score, images, true_classes, heatmaps)
        for i, k in enumerate(true_classes):
            self.num[k] += torch.mean(lerf[i] - morf[i])
            self.den[k] += 1

    def compute(self):
        return torch.mean(self.num / self.den)

class Density(MyMetric):
    # to measure how sparse the explanation is
    def update(self, heatmaps):
        assert len(heatmaps.shape) == 3, f'heatmaps have more than three dimensions: {heatmaps.shape}'
        heatmaps = heatmaps / heatmaps.amax((1, 2), True)
        self.num += torch.sum(torch.mean(heatmaps, (1, 2)))
        self.den += len(heatmaps)

class Entropy(MyMetric):
    # to measure how sparse the explanation is
    def update(self, heatmaps):
        assert len(heatmaps.shape) == 3, f'heatmaps have more than three dimensions: {heatmaps.shape}'
        heatmaps = heatmaps / heatmaps.sum((1, 2), True)
        # avoid log(0) by replacing zeros with a very small value
        den = torch.prod(torch.tensor(heatmaps.shape[1:]))
        self.num += -torch.sum(heatmaps * torch.log2(heatmaps+1e-12) / torch.log2(den))
        self.den += len(heatmaps)

class TotalVariance(MyMetric):
    # to measure how sparse the explanation is
    def update(self, heatmaps):
        assert len(heatmaps.shape) == 3, f'heatmaps have more than three dimensions: {heatmaps.shape}'
        heatmaps = heatmaps / heatmaps.amax((1, 2), True)
        dy = torch.mean(torch.abs(heatmaps[:, 1:]-heatmaps[:, :-1]), (1, 2))
        dx = torch.mean(torch.abs(heatmaps[:, :, 1:]-heatmaps[:, :, :-1]), (1, 2))
        self.num += torch.sum((dx + dy)/2)
        self.den += len(heatmaps)
