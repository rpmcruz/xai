# 2024 Ricardo Cruz <ricardo.pdm.cruz@gmail.com> 
# Implementation of xAI methods.

import torch, torchvision

# In the case of ResNet-50:
# layer_mid = features.layer2
# layer_act = features.layer4[-1].conv3
# layer_fc  = features.fc

# https://arxiv.org/abs/1512.04150
def CAM(model, layer_mid, layer_act, layer_fc, x, y):
    act = None
    def act_fhook(_, input, output):
        nonlocal act
        act = output
    h = layer_act.register_forward_hook(act_fhook)
    with torch.no_grad():
        model(x)
    h.remove()
    w = layer_fc.weight[y, :]
    heatmap = torch.sum(w[..., None, None]*act, 1)
    heatmap = heatmap / (torch.amax(torch.abs(heatmap), (1, 2), True).clamp(min=1e-6))
    return heatmap

# https://ieeexplore.ieee.org/document/8237336
def GradCAM(model, layer_mid, layer_act, layer_fc, x, y):
    act = w = None
    def act_fhook(_, input, output):
        nonlocal act
        act = output
    def act_bhook(_, grad_input, grad_output):
        nonlocal w
        w = torch.mean(grad_output[0], (2, 3))
    fh = layer_act.register_forward_hook(act_fhook)
    bh = layer_act.register_full_backward_hook(act_bhook)
    pred = model(x)['class']
    pred = pred[range(len(y)), y].sum()
    pred.backward()
    fh.remove()
    bh.remove()
    # in the paper, they use relu to eliminate the negative values
    # (but maybe we want them to improve our metrics like degredation score)
    heatmap = torch.sum(w[..., None, None]*act, 1)
    heatmap = heatmap / (torch.amax(torch.abs(heatmap).clamp(min=1e-6), (1, 2), True))
    return heatmap

# https://arxiv.org/abs/1704.02685
def DeepLIFT(model, layer_mid, layer_act, layer_fc, x, y):
    baseline = torch.zeros_like(x)
    x.requires_grad = True
    pred_baseline = model(baseline)['class'][range(len(y)), y]
    pred_x = model(x)['class'][range(len(y)), y]
    delta = pred_x - pred_baseline
    delta.sum().backward()
    heatmap = torch.mean((x - baseline) * x.grad, 1)
    heatmap = heatmap / (1e-5+torch.amax(torch.abs(heatmap), (1, 2), True))
    return heatmap

def Occlusion(model, layer_mid, layer_act, layer_fc, x, y):
    occ_w = x.shape[3]//7
    occ_h = x.shape[2]//7
    heatmap = torch.zeros(len(x), 7, 7, device=x.device)
    for occ_i, occ_x in enumerate(range(0, x.shape[3], occ_w)):
        for occ_j, occ_y in enumerate(range(0, x.shape[2], occ_h)):
            occ_img = x.clone()
            occ_img[:, :, occ_x:occ_y+occ_h, occ_x:occ_x+occ_w] = 0
            with torch.no_grad():
                prob = torch.softmax(model(occ_img)['class'], 1)[range(len(y)), y]
                heatmap[:, occ_j, occ_i] = 1-prob
    return heatmap

# https://openreview.net/forum?id=S1xWh1rYwB
def IBA(model, layer_mid, layer_act, layer_fc, x, y, lr=1, num_steps=100, sigma=1, beta=10):
    # one forward pass to get the mid shape
    mid_shape = None
    def mid_fhook(_, input, output):
        nonlocal mid_shape
        mid_shape = output.shape
    fh = layer_mid.register_forward_hook(mid_fhook)
    model(x)
    fh.remove()
    # apply bottleneck
    alpha = torch.nn.Parameter(5*torch.ones(mid_shape, device=x.device))
    mask = R_norm = Z = None
    def mid_fhook(_, input, output):
        nonlocal mask, R_norm, Z
        # inject gaussian noise
        mean = output.mean((2, 3), keepdim=True)
        std = output.std((2, 3), keepdim=True).clamp(min=1e-6)
        noise = torch.normal(mean, std).to(output.device)
        mask = torch.sigmoid(alpha)
        mask = torchvision.transforms.functional.gaussian_blur(mask, 5, sigma)
        R_norm = (output - mean) / std
        Z = mask*output + (1-mask)*noise
        return Z
    fh = layer_mid.register_forward_hook(mid_fhook)
    optimizer = torch.optim.Adam([alpha], lr)
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(x)['class']
        # L = LCE + βLI  (accurate, but minimize the information passed)
        # same code as the authors
        ce = torch.nn.functional.cross_entropy(output, y)
        mu_Z = R_norm * mask
        var_Z = ((1-mask) ** 2).clamp(min=1e-6)
        information_loss = torch.mean(-0.5 * (1 + torch.log(var_Z) - mu_Z**2 - var_Z))
        loss = ce + beta*information_loss
        loss.backward()
        optimizer.step()
    fh.remove()
    return mask.mean(1)
