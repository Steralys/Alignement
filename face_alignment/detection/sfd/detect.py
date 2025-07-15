import torch
import torch.nn.functional as F
from numpy import ndarray
import cv2
import numpy as np

from .bbox import *


def detect(net, img, device):
    img = img.transpose(2, 0, 1)
    # Creates a batch of 1
    img = np.expand_dims(img, 0)

    img = torch.from_numpy(img.copy()).to(device, dtype=torch.float32)

    return batch_detect(net, img, device)


def tbatch_detect(net, img_batch: torch.Tensor, device):
    """
    Inputs:
        - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
    """
    batch_size = img_batch.size(0)
    img_batch = img_batch.to(device, dtype=torch.float32)

    img_batch = img_batch.flip(-3)  # RGB to BGR
    img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        olist = net(img_batch)  # patched uint8_t overflow error

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    bboxlists = tget_predictions(olist, batch_size)
    return bboxlists

def batch_detect(net, img_batch, device):
    """
    Inputs:
        - img_batch: a torch.Tensor of shape (Batch size, Channels, Height, Width)
    """
    batch_size = img_batch.size(0)
    img_batch = img_batch.to(device, dtype=torch.float32)

    img_batch = img_batch.flip(-3)  # RGB to BGR
    img_batch = img_batch - torch.tensor([104.0, 117.0, 123.0], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        olist = net(img_batch)  # patched uint8_t overflow error

    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)

    olist = [oelem.data.cpu().numpy() for oelem in olist]

    bboxlists = get_predictions(olist, batch_size)
    return bboxlists


def tget_predictions(olist: list[torch.Tensor], batch_size: int):
    score_threshold = 0.5
    bboxlists = []
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]  # ocls: (B, 2, H, W), oreg: (B, 4, H, W)
        stride = 2 ** (i + 2)
        device = ocls.device

        # Get indices where score > threshold
        pos = torch.nonzero(ocls[:, 1, :, :] > 0.05, as_tuple=False)  # (N, 3): [batch_idx, h_idx, w_idx]

        if pos.size(0) == 0:
            continue

        for idx in range(pos.size(0)):
            b_idx, h_idx, w_idx = pos[idx]

            score = ocls[b_idx, 1, h_idx, w_idx].unsqueeze(0).unsqueeze(1)  # shape: (1, 1)
            if score.item() < score_threshold:
                continue

            axc = stride / 2 + w_idx * stride
            ayc = stride / 2 + h_idx * stride

            # Prior in (cx, cy, w, h) format
            priors = torch.tensor([[axc, ayc, stride * 4, stride * 4]], device=device)

            loc = oreg[b_idx, :, h_idx, w_idx].unsqueeze(0)  # shape: (1, 4)
            boxes = tdecode(loc, priors)  # Output shape: (1, 4)
            bbox = torch.cat((boxes, score), dim=1)  # shape: (1, 5)
            bboxlists.append((b_idx.item(), bbox))

    if not bboxlists:
        return [torch.empty((0, 5), device=device) for _ in range(batch_size)]

    # Group boxes by batch index
    output = [[] for _ in range(batch_size)]
    for b_idx, bbox in bboxlists:
        output[b_idx].append(bbox)

    # Stack per-batch boxes into tensors
    output = [
        torch.cat(boxes, dim=0) if boxes else torch.empty((0, 5), device=device)
        for boxes in output
    ]
    return output  # List[Tensor] of shape (num_boxes, 5) per batch

def get_predictions(olist: list[ndarray], batch_size):
    bboxlists = []
    variances = [0.1, 0.2]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            score = ocls[:, 1, hindex, windex][:,None]
            loc = oreg[:, :, hindex, windex].copy()
            boxes = decode(loc, priors, variances)
            bboxlists.append(np.concatenate((boxes, score), axis=1))
    
    if len(bboxlists) == 0: # No candidates within given threshold
        bboxlists = np.array([[] for _ in range(batch_size)])
    else:
        bboxlists = np.stack(bboxlists, axis=1)
    return bboxlists


def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
