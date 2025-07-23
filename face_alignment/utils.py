import torch
import os, sys, errno
import numpy as np
# import cv2

from urllib.parse import urlparse
from torch.hub import download_url_to_file, HASH_REGEX
try:
    from torch.hub import get_dir
except BaseException:
    from torch.hub import _get_torch_home as get_dir

import time
from functools import wraps
from tqdm import tqdm


def mesure_temps(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        debut = time.time()
        result = func(*args, **kwargs)
        fin = time.time()
        duree = fin - debut
        tqdm.write(f"{func.__name__} éxecutée en {duree:.4f} secondes")
        return result
    return wrapper

def crop_with_centers_scales(frames, centers, scales, out_size):
    """
    frames: (N, 3, H, W)
    centers: (N, 2) in (x, y) format (absolute coords)
    scales: (N,) where 1.0 means "reference crop"
    out_size: (H_out, W_out)

    Returns:
        cropped: (N, 3, H_out, W_out)
        crop_info: dict with keys:
            - 'x1y1': (N, 2) tensor of top-left coords (float)
            - 'size': (N, 2) tensor of crop size in pixels (w, h)
    """
    N, C, H, W = frames.shape
    device = frames.device

    # Compute crop size in original image space
    scale = scales.view(-1, 1)  # (N, 1)
    crop_w = scale[:, 0] * out_size[1]
    crop_h = scale[:, 0] * out_size[0]

    # Normalize center and scale for grid_sample
    center_x = (centers[:, 0] / (W - 1)) * 2 - 1  # (N,)
    center_y = (centers[:, 1] / (H - 1)) * 2 - 1
    scale_x = (crop_w / (W - 1))                  # (N,)
    scale_y = (crop_h / (H - 1))

    # Create normalized grid
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, out_size[0], device=device),
        torch.linspace(-1, 1, out_size[1], device=device),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=-1)[None].repeat(N, 1, 1, 1)  # (N, H_out, W_out, 2)

    grid = base_grid.clone()
    grid[..., 0] = grid[..., 0] * scale_x[:, None, None] + center_x[:, None, None]
    grid[..., 1] = grid[..., 1] * scale_y[:, None, None] + center_y[:, None, None]

    cropped = torch.nn.functional.grid_sample(frames, grid, mode='bilinear', align_corners=True)
    return cropped

def crop_csr(frames, centers, scales, rotations, out_size=512):
    """
    Crops and rotates videos around center points with given scale and rotation (square crop).
    
    Args:
        frames: (N, 3, H, W) 
        centers: (N, 2) in (x, y) format 
        scales: (N,)  - scale factor for square crop size
        rotations: (N,) - rotation in radians, counterclockwise
        out_size: int - output size (default 512)
    
    Returns:
        cropped: (N, 3, out_size, out_size) 
    """
    N, C, H, W = frames.shape
    device = frames.device
    dtype = frames.dtype

    # Normalize center
    center_x = (centers[:, 0] / (W - 1)) * 2 - 1  # (N,)
    center_y = (centers[:, 1] / (H - 1)) * 2 - 1  # (N,)
    # center = torch.stack([center_x, center_y], dim=1)  # (N, 2)

    # Scaling factors
    scale = scales.view(-1, 1)  # (N, 1)
    scale_x = (scale[:, 0] * out_size) / (W - 1)  # (N,)
    scale_y = (scale[:, 0] * out_size) / (H - 1)  # (N,)

    # Create base grid (normalized coordinates in [-1, 1])
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, out_size, device=device, dtype=dtype),
        torch.linspace(-1, 1, out_size, device=device, dtype=dtype),
        indexing='ij'
    )
    base_grid = torch.stack((grid_x, grid_y), dim=-1)  # (H, W, 2)
    base_grid = base_grid.unsqueeze(0).repeat(N, 1, 1, 1)  # (N, H, W, 2)

    # Build rotation matrix
    sin = torch.sin(rotations).view(N, 1, 1)
    cos = torch.cos(rotations).view(N, 1, 1)
    rot_matrix = torch.stack([
        torch.stack([cos, -sin], dim=-1),  # (N, 1, 2)
        torch.stack([sin,  cos], dim=-1)
    ], dim=-2).squeeze(1)  # (N, 1, 2, 2)

    # Apply rotation
    grid = base_grid @ rot_matrix  # (N, H, W, 2)

    # Apply scaling
    grid[..., 0] = grid[..., 0] * scale_x[:, None, None] + center_x[:, None, None]
    grid[..., 1] = grid[..., 1] * scale_y[:, None, None] + center_y[:, None, None]

    # Sample using grid_sample
    cropped = torch.nn.functional.grid_sample(
        frames, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )
    return cropped

def get_preds_fromhm(hm: torch.Tensor):
    """Obtain (x,y) coordinates given a set of N heatmaps.

    Arguments:
        hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape # [N, 68, 64, 64]
    hm_reshape = hm.reshape(B, C, H * W) # [N, 68, 4096]
    idx = torch.argmax(hm_reshape, dim=-1) + 1 # [N, 68]
    # scores = torch.gather(hm_reshape, -1, idx.unsqueeze(-1)).squeeze(-1)

    # Recover initial predicted x and y positions from flat indices
    preds = idx.repeat_interleave(2) # [N, 68, 2]
    preds = preds.reshape(B, C, 2).float()
    preds_x = (preds[:, :, 0] - 1) % W
    preds_y = torch.floor((preds[:, :, 1] - 1) / H)
    preds[:, :, 0] = preds_x + 1
    preds[:, :, 1] = preds_y + 1

    # Now compute the offset using gradients around the predicted location
    px = preds[:, :, 0].long() - 1  # 0-based
    py = preds[:, :, 1].long() - 1

    # Mask to exclude border positions
    valid = (px > 0) & (px < W - 1) & (py > 0) & (py < H - 1)

    # Flatten for indexing
    flat_idx = torch.arange(B * C, device=hm.device)
    hm_reshaped = hm.reshape(B * C, H, W)
    px_flat = px.reshape(-1)
    py_flat = py.reshape(-1)
    valid_flat = valid.reshape(-1)

    dx = torch.zeros_like(px_flat, dtype=torch.float32)
    dy = torch.zeros_like(py_flat, dtype=torch.float32)

    # Only compute diffs where valid
    vfi = flat_idx[valid_flat]
    vpx = px_flat[valid_flat]
    vpy = py_flat[valid_flat]
    dx[valid_flat] = hm_reshaped[vfi, vpy, vpx + 1] - hm_reshaped[vfi, vpy, vpx - 1]
    dy[valid_flat] = hm_reshaped[vfi, vpy + 1, vpx] - hm_reshaped[vfi, vpy - 1, vpx]

    # Apply offset
    offset = torch.stack([torch.sign(dx), torch.sign(dy)], dim=1).reshape(B, C, 2) * 0.25
    preds = (preds + offset - 0.5) * 4

    return preds

def load_file_from_url(url, model_dir=None, progress=True, check_hash=False, file_name=None):
    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, cached_file, hash_prefix, progress=progress)

    return cached_file

# ====================================================================================

def legacy_get_preds_fromhm(hm, center=None, scale=None):
    """Obtain (x,y) coordinates given a set of N heatmaps. If the center
    and the scale is provided the function will return the points also in
    the original coordinate frame.

    Arguments:
        hm {ndarray} -- the predicted heatmaps, of shape [B, N, W, H]

    Keyword Arguments:
        center {torch.tensor} -- the center of the bounding box (default: {None})
        scale {float} -- face scale (default: {None})
    """
    B, C, H, W = hm.shape
    hm_reshape = hm.reshape(B, C, H * W)
    idx = np.argmax(hm_reshape, axis=-1)
    scores = np.take_along_axis(hm_reshape, np.expand_dims(idx, axis=-1), axis=-1).squeeze(-1)
    preds, preds_orig = _get_preds_fromhm(hm, idx, center, scale)

    return preds, preds_orig, scores

def crop(image, center, scale, resolution=256.0):
    """Center crops an image or set of heatmaps

    Arguments:
        image {numpy.array} -- an rgb image
        center {numpy.array} -- the center of the object, usually the same as of the bounding box
        scale {float} -- scale of the face

    Keyword Arguments:
        resolution {float} -- the size of the output cropped image (default: {256.0})

    Returns:
        [type] -- [description]
    """ 
    """ Crops the image around the center. Input is expected to be an np.ndarray """
    ul = transform([1, 1], center, scale, resolution, True)
    br = transform([resolution, resolution], center, scale, resolution, True)
    newDim = np.array([br[1] - ul[1], br[0] - ul[0], image.shape[2]], dtype=np.int32)
    newImg = np.zeros(newDim, dtype=np.uint8)
    ht = image.shape[0]
    wd = image.shape[1]
    newX = np.array(
        [max(1, -ul[0] + 1), min(br[0], wd) - ul[0]], dtype=np.int32) #type: ignore
    newY = np.array(
        [max(1, -ul[1] + 1), min(br[1], ht) - ul[1]], dtype=np.int32) #type: ignore
    oldX = np.array([max(1, ul[0] + 1), min(br[0], wd)], dtype=np.int32) #type: ignore
    oldY = np.array([max(1, ul[1] + 1), min(br[1], ht)], dtype=np.int32) #type: ignore
    newImg[newY[0] - 1:newY[1], newX[0] - 1:newX[1]
           ] = image[oldY[0] - 1:oldY[1], oldX[0] - 1:oldX[1], :]
    # newImg = cv2.resize(newImg, dsize=(int(resolution), int(resolution)),
    #                     interpolation=cv2.INTER_LINEAR)
    return newImg

# @mesure_temps
# @jit(nopython=True)
def transform_np(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {numpy.array} -- the input 2D point
        center {numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = np.ascontiguousarray(np.linalg.pinv(t))

    new_point = np.dot(t, _pt)[0:2]

    return new_point.astype(np.int32)

def transform(point, center, scale, resolution, invert=False):
    """Generate and affine transformation matrix.

    Given a set of points, a center, a scale and a targer resolution, the
    function generates and affine transformation matrix. If invert is ``True``
    it will produce the inverse transformation.

    Arguments:
        point {torch.tensor} -- the input 2D point
        center {torch.tensor or numpy.array} -- the center around which to perform the transformations
        scale {float} -- the scale of the face/object
        resolution {float} -- the output resolution

    Keyword Arguments:
        invert {bool} -- define wherever the function should produce the direct or the
        inverse transformation matrix (default: {False})
    """
    _pt = torch.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = torch.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if invert:
        t = torch.inverse(t)

    new_point = (torch.matmul(t, _pt))[0:2]

    return new_point.int()

def _get_preds_fromhm(hm, idx, center=None, scale=None):
    """
    Vectorized version of the heatmap prediction extraction.
    """
    B, C, H, W = hm.shape  # e.g., 1, 68, 64, 64
    idx = idx + 1  # match behavior of original

    # Recover initial predicted x and y positions from flat indices
    preds = idx.repeat(2).reshape(B, C, 2).astype(np.float32)
    preds_x = (preds[:, :, 0] - 1) % W
    preds_y = np.floor((preds[:, :, 1] - 1) / H)
    preds[:, :, 0] = preds_x + 1
    preds[:, :, 1] = preds_y + 1

    # Now compute the offset using gradients around the predicted location
    px = preds[:, :, 0].astype(np.int64) - 1  # 0-based
    py = preds[:, :, 1].astype(np.int64) - 1

    # Mask to exclude border positions
    valid = (px > 0) & (px < W - 1) & (py > 0) & (py < H - 1)

    # Flatten for indexing
    flat_idx = np.arange(B * C)
    hm_reshaped = hm.reshape(B * C, H, W)
    px_flat = px.reshape(-1)
    py_flat = py.reshape(-1)
    valid_flat = valid.reshape(-1)

    # Only compute diffs where valid
    dx = np.zeros_like(px_flat, dtype=np.float32)
    dy = np.zeros_like(py_flat, dtype=np.float32)

    # Get gradient differences
    dx[valid_flat] = hm_reshaped[flat_idx[valid_flat], py_flat[valid_flat], px_flat[valid_flat] + 1] - \
                     hm_reshaped[flat_idx[valid_flat], py_flat[valid_flat], px_flat[valid_flat] - 1]
    dy[valid_flat] = hm_reshaped[flat_idx[valid_flat], py_flat[valid_flat] + 1, px_flat[valid_flat]] - \
                     hm_reshaped[flat_idx[valid_flat], py_flat[valid_flat] - 1, px_flat[valid_flat]]

    # Apply offset
    offset = np.stack([np.sign(dx), np.sign(dy)], axis=1).reshape(B, C, 2) * 0.25
    preds += offset

    preds -= 0.5
    preds_orig = np.zeros_like(preds)
    if center is not None and scale is not None:
        for i in range(B):
            for j in range(C):
                preds_orig[i, j] = transform_np(
                    preds[i, j], center, scale, H, True)
    return preds, preds_orig


def create_bounding_box(target_landmarks, expansion_factor=0.0):
    """
    gets a batch of landmarks and calculates a bounding box that includes all the landmarks per set of landmarks in
    the batch
    :param target_landmarks: batch of landmarks of dim (n x 68 x 2). Where n is the batch size
    :param expansion_factor: expands the bounding box by this factor. For example, a `expansion_factor` of 0.2 leads
    to 20% increase in width and height of the boxes
    :return: a batch of bounding boxes of dim (n x 4) where the second dim is (x1,y1,x2,y2)
    """
    # Calc bounding box
    x_y_min, _ = target_landmarks.reshape(-1, 68, 2).min(dim=1)
    x_y_max, _ = target_landmarks.reshape(-1, 68, 2).max(dim=1)
    # expanding the bounding box
    expansion_factor /= 2
    bb_expansion_x = (x_y_max[:, 0] - x_y_min[:, 0]) * expansion_factor
    bb_expansion_y = (x_y_max[:, 1] - x_y_min[:, 1]) * expansion_factor
    x_y_min[:, 0] -= bb_expansion_x
    x_y_max[:, 0] += bb_expansion_x
    x_y_min[:, 1] -= bb_expansion_y
    x_y_max[:, 1] += bb_expansion_y
    return torch.cat([x_y_min, x_y_max], dim=1)


def shuffle_lr(parts, pairs=None):
    """Shuffle the points left-right according to the axis of symmetry
    of the object.

    Arguments:
        parts {torch.tensor} -- a 3D or 4D object containing the
        heatmaps.

    Keyword Arguments:
        pairs {list of integers} -- [order of the flipped points] (default: {None})
    """
    if pairs is None:
        pairs = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 27, 28, 29, 30, 35,
                 34, 33, 32, 31, 45, 44, 43, 42, 47, 46, 39, 38, 37, 36, 41,
                 40, 54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55, 64, 63,
                 62, 61, 60, 67, 66, 65]
    if parts.ndimension() == 3:
        parts = parts[pairs, ...]
    else:
        parts = parts[:, pairs, ...]

    return parts


def flip(tensor, is_label=False):
    """Flip an image or a set of heatmaps left-right

    Arguments:
        tensor {numpy.array or torch.tensor} -- [the input image or heatmaps]

    Keyword Arguments:
        is_label {bool} -- [denote wherever the input is an image or a set of heatmaps ] (default: {False})
    """
    if not torch.is_tensor(tensor):
        tensor = torch.from_numpy(tensor)

    if is_label:
        tensor = shuffle_lr(tensor).flip(tensor.ndimension() - 1)
    else:
        tensor = tensor.flip(tensor.ndimension() - 1)

    return tensor
