# 2024 Ricardo Cruz <ricardo.pdm.cruz@gmail.com> 
# Implementation of interpretability metrics.

import torchmetrics
import torch

class PointingGame(torchmetrics.Metric):
    # https://link.springer.com/article/10.1007/s11263-017-1059-x
    def __init__(self):
        super().__init__()
        self.add_state('correct', default=torch.tensor(0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, heatmaps, masks):
        heatmaps = torch.nn.functional.interpolate(heatmaps[:, None], masks.shape[-2:], mode='bilinear')
        heatmaps = heatmaps.view(len(heatmaps), -1)
        masks = masks.view(len(masks), -1)
        self.correct += sum(masks[range(len(masks)), torch.argmax(heatmaps, 1)] != 0)
        self.total += len(heatmaps)

    def compute(self):
        return self.correct / self.total

class DegradationScore(torchmetrics.Metric):
    def __init__(self, model, score='acc'):
        super().__init__()
        self.model = model
        if score == 'acc':
            self.score = lambda ypred, y: (ypred == y).float()
        else:
            raise Exception(f'Unknown score: {score}')
        self.add_state('areas', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('count', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, images, true_classes, heatmaps):
        lerf = self.degradation_curve('lerf', self.model, self.score, images, true_classes, heatmaps)
        morf = self.degradation_curve('morf', self.model, self.score, images, true_classes, heatmaps)
        self.areas += torch.sum(torch.mean(lerf - morf, 1))
        self.count += len(images)

    def compute(self):
        return self.areas / self.count

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
            if type(ypred) == dict:
                ypred = ypred['class']
            else:
                ypred = ypred[0]
            ypred = ypred.argmax(1)
            score = score_fn(ypred, true_classes)
            scores.append(score)
        return torch.stack(scores, 1)

class Density(torchmetrics.Metric):
    # to measure how sparse the explanation is
    def __init__(self):
        super().__init__()
        self.add_state('l1', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, heatmaps):
        assert len(heatmaps.shape) == 3, f'heatmaps have more than three dimensions: {heatmaps.shape}'
        heatmaps = heatmaps / heatmaps.amax((1, 2), True)
        self.l1 += torch.sum(torch.mean(heatmaps, (1, 2)))
        self.total += len(heatmaps)

    def compute(self):
        return self.l1 / self.total

class Entropy(torchmetrics.Metric):
    # to measure how sparse the explanation is
    def __init__(self):
        super().__init__()
        self.add_state('entropy', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, heatmaps):
        assert len(heatmaps.shape) == 3, f'heatmaps have more than three dimensions: {heatmaps.shape}'
        heatmaps = heatmaps / heatmaps.sum((1, 2), True)
        # avoid log(0) by replacing zeros with a very small value
        den = torch.prod(torch.tensor(heatmaps.shape[1:]))
        self.entropy += -torch.sum(heatmaps * torch.log2(heatmaps+1e-12) / torch.log2(den))
        self.total += len(heatmaps)

    def compute(self):
        return self.entropy / self.total


class TotalVariance(torchmetrics.Metric):
    # to measure how sparse the explanation is
    def __init__(self):
        super().__init__()
        self.add_state('variance', default=torch.tensor(0.), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, heatmaps):
        assert len(heatmaps.shape) == 3, f'heatmaps have more than three dimensions: {heatmaps.shape}'
        heatmaps = heatmaps / heatmaps.amax((1, 2), True)
        dy = torch.mean(torch.abs(heatmaps[:, 1:]-heatmaps[:, :-1]), (1, 2))
        dx = torch.mean(torch.abs(heatmaps[:, :, 1:]-heatmaps[:, :, :-1]), (1, 2))
        self.variance += torch.sum((dx + dy)/2)
        self.total += len(heatmaps)

    def compute(self):
        return self.variance / self.total
