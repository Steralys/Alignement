import torch
from tqdm import tqdm
from .utils import crop_csr, paste_csr, crop_with_centers_scales, get_preds_fromhm
from .detection.retina.pytorch_retinaface import Pytorch_RetinaFace
import time
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from typing import Optional
from pathlib import Path

default_model_urls = {
    '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip',
    '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-4a694010b9.zip',
    'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-6c4283c0e0.zip',
}

models_urls = {
    '1.6': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip',
    },
    '1.5': {
        '2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.5-a60332318a.zip',
        '3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.5-176570af4d.zip',
        'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth_1.5-bc10f98e39.zip',
    },
}

def fill_none_with_precedent(boxes, n):
    filled = []
    last_valid = None
    none_count = 0
    for i, box in enumerate(boxes):
        if box is not None:
            filled.append(box)
            last_valid = box
            none_count = 0
        else:
            none_count += 1
            if last_valid is not None and none_count <= n:
                filled.append(last_valid)
            else:
                raise RuntimeError(f"More than {n} consecutive None values at index {i}")
    return filled

def resize_video(video: torch.Tensor, max_resolution: int = 1280, batch_size: int = 8, device="cuda"):
    """ 
    Resizes the video tensor to fit within the max resolution while maintaining aspect ratio.
    Args:
        video (torch.Tensor): Input video tensor of shape (B, C, H, W).
        max_resolution (int): Maximum resolution for the longest side.
    Returns:
        tuple:
            resized (torch.Tensor): Resized video tensor.
            scale_factor (float): Scale factor used for resizing.
    """
    B, C, H, W = video.shape
    scale_factor = max_resolution / max(H, W)

    resized = []
    for start in tqdm(range(0, B, batch_size), desc=f"Resizing to {max_resolution} max"):
        end = min(start + batch_size, B)
        batch = video[start:end]
        resized_chunk = torch.nn.functional.interpolate(
            batch.to(device=device, dtype=torch.float32),
            scale_factor=scale_factor, 
            mode='bilinear', align_corners=False
        ).to(video.dtype)
        resized.append(resized_chunk.cpu())
    resized = torch.cat(resized, dim=0)
    return resized, scale_factor


class FaceAlignment:
    def __init__(self, device='cuda'):
        self.device = device

        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        self.face_detector = Pytorch_RetinaFace(top_k=20, keep_top_k=10, device=device, confidence_threshold=0.5)

        # Initialise the face alignemnt networks
        # network_name = '2DFAN-4' https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip
        self.face_alignment_net = torch.jit.load(str(Path(__file__).parent.parent/"ckpt"/"2DFAN4-cd938726ad.zip"))

        self.face_alignment_net.to(device, dtype=torch.float32)
        self.face_alignment_net.eval()

    @torch.no_grad()
    def detect_faces(self, image_batch: torch.Tensor, batch_size=8, mode="v1"):
        """
        Keeps the biggest face in each frame
        Args:
            image_batch: TCHW uint8
            batch_size: int, 32 too much for 8gb
            mode: v1 batch_nms, v2 for loop nms needs benchmark
        Returns:
            bbox_batch: torch.Tensor, shape [N, 4] with x1, y1, x2, y2
        """
        return self.face_detector.detect_face_v2(image_batch, batch_size, mode)

    def sample_crop(
            self,
            image_batch: torch.Tensor,
            bbox: torch.Tensor,
            size: tuple[int, int]=(256, 256),
            scale_factor=None,
            batch_size=8
        ):
        """
        Keeps the biggest face in each frame
        Args:
            image_batch: TCHW uint8
            batch_size: int, 16 too much for 8gb
            size: height, width of cropped images
            scale_factor: default(auto) => proportional to size
        Returns:
            cropped: torch.Tensor, shape [N, 3, 256, 256] with cropped faces
        """
        start_crop = time.time()

        if not scale_factor:
            scale_factor = size[0]
        assert image_batch.shape[0] == bbox.shape[0], "Image batch and bbox batch must be the same size"
        x1, y1, x2, y2 = bbox[:, 0], bbox[:, 1], bbox[:, 2], bbox[:, 3]
        centers = torch.stack([x2 - (x2 - x1) * 0.5, y2 - (y2 - y1) * 0.62]).permute(1, 0)
        scales = (x2 - x1 + y2 - y1) / scale_factor

        cropped = []
        for i in tqdm(range(0, image_batch.shape[0], batch_size), desc="Crop-Sampling"):
            if i + batch_size > image_batch.shape[0]:
                batch_slice = image_batch[i:]
                center = centers[i:]
                scale = scales[i:]
            else:
                batch_slice = image_batch[i:i + batch_size]
                center = centers[i:i + batch_size]
                scale = scales[i:i + batch_size]
            
            result = crop_with_centers_scales(
                batch_slice.to(device=self.device, dtype=torch.float32),
                center,
                scale,
                size,
            )
            cropped.append(result)

        cropped = torch.cat(cropped, dim=0)
        print(f"[INFO] Crop-Sampling time: {time.time() - start_crop:.4f} seconds")
        return cropped

    @torch.inference_mode()
    def sample_crop_csr(
        self,
        image_batch: torch.Tensor,
        batch_size=8
    ):
        """
        Keeps the biggest face in each frame
        Args:
            image_batch: TCHW uint8, full hd max
            batch_size: int
        Returns:
            tuple:
                cropped: torch.Tensor, shape [N, 3, 512, 512] with cropped faces
                landmarks: torch.Tensor, shape [N, 68, 2]
        """
        B, C, H, W = image_batch.shape

        # Detect faces; retina face doesn't work well in res > hd
        MAX_RESOLUTION = 1280
        if max(H, W) > MAX_RESOLUTION:
            resized, resize_ratio = resize_video(image_batch, MAX_RESOLUTION, device=self.device)
            boxes, eyes = self.detect_faces(resized)
            boxes = boxes / resize_ratio
            eyes = eyes / resize_ratio
            del resized
        else:
            boxes, eyes = self.detect_faces(image_batch)

        # Compute face tilt angle
        eye_vector = eyes[:, 1, :] - eyes[:, 0, :]
        angles_rad = torch.arctan2(eye_vector[:, 1], eye_vector[:, 0]) * -1

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        # centers = torch.stack([x2 - (x2 - x1) * 0.5, y2 - (y2 - y1) * 0.62]).permute(1, 0)
        centers = eyes[:, 0, :] + eye_vector / 2
        scales = (x2 - x1 + y2 - y1) / 300

        # Detect landmarks and prepare 512x512 lipsync inputs
        landmarks = []
        cropped = []
        paste_infos = []
        for start in tqdm(range(0, B, batch_size), desc="Crop-Sampling CSR"):
            end = min(start + batch_size, B)
            batch = image_batch[start:end].to(self.device, dtype=torch.float32)

            cropped_256, _ = crop_csr(
                batch,
                centers[start:end],
                scales[start:end],
                angles_rad[start:end],
                out_size=256
            )

            out = self.face_alignment_net(cropped_256 / 255.0)
            pts = get_preds_fromhm(out)
            landmarks.append(pts)

            cropped_512, infos = crop_csr(
                batch,
                centers[start:end],
                scales[start:end] / 2,
                angles_rad[start:end],
                out_size=512
            )
            cropped.append(cropped_512.to(device="cpu", dtype=torch.uint8))
            paste_infos.append(infos)
            torch.cuda.empty_cache()
        landmarks = torch.cat(landmarks) * 2
        cropped = torch.cat(cropped)
        paste_infos = torch.cat(paste_infos)
        return cropped, landmarks, paste_infos

    @torch.inference_mode()
    def paste_back(self):
        raise NotImplementedError()
    
    @torch.inference_mode()
    def paste_back_csr(self, original: torch.Tensor, cropped: torch.Tensor, infos: torch.Tensor, batch_size=8):
        """
        Paste the cropped frames back on the original video
        Args:
            original: TCHW
            cropped: TCHW
            infos: T4 => T, (x1, y1, original_crop_size, angle)
            batch_size: int
        Returns:
            pasted: torch.Tensor, shape [N, C, H, W]
        """

        B, C, H, W = original.shape
        assert cropped.shape[0] == B, f"Got different frame numbers. original: {B}, cropped: {cropped.shape[0]}"
        assert infos.shape[0] == B, f"Crop infos length and frame number don't match. frames: {B}, infos: {infos.shape[0]}"

        pasted = []
        for start in tqdm(range(0, B, batch_size), desc="Paste-Back CSR"):
            end = min(start + batch_size, B)

            original_batch = original[start:end].to(self.device, dtype=torch.float32)
            cropped_batch = cropped[start:end].to(self.device, dtype=torch.float32)
            infos_batch = infos[start:end].to(self.device, dtype=torch.float32)

            x1y1 = infos_batch[:, :2]
            size = infos_batch[:, 2]
            rot = infos_batch[:, 3]

            pasta, mask = paste_csr(
                cropped_batch,
                x1y1,
                size,
                rot,
                (H, W)
            )
            composite = pasta * mask + original_batch * (~ mask)
            pasted.append(composite.to(device="cpu", dtype=torch.uint8))
            torch.cuda.empty_cache()
        pasted = torch.cat(pasted)
        return pasted


    @torch.no_grad()
    def detect_landmarks(self, image_batch: torch.Tensor, batch_size=8):
        """
        Detects landmarks in the given image batch. Must be cropped to faces.
        Args:
            image_batch: TCHW float32 256x256
            batch_size: int
        Returns:
            landmarks: torch.Tensor, shape [N, 68, 2] with x, y coordinates of landmarks
        """
        start_ld = time.time()
        image_batch = image_batch.div(255.0)

        landmarks = []
        for i in range(0, image_batch.shape[0], batch_size):
            if i + batch_size > image_batch.shape[0]:
                batch_slice = image_batch[i:]
            else:
                batch_slice = image_batch[i:i + batch_size]
            out = self.face_alignment_net(batch_slice.to(device=self.device, dtype=torch.float32))
            pts = get_preds_fromhm(out)
            landmarks.append(pts)

        landmarks = torch.cat(landmarks, dim=0)
        print(f"[INFO] Landmark Detect time: {time.time() - start_ld:.4f} seconds")
        torch.cuda.empty_cache()
        return landmarks

    @torch.no_grad()
    def process(self, image_batch: torch.Tensor):
        """
        Full pipeline
        Args:
            image_batch: TCHW uint8
        Returns:
            landmarks: torch.Tensor, shape [N, 68, 2] with x, y coordinates of landmarks
        """
        start = time.time()
        bboxs = self.detect_faces(image_batch)
        cropped = self.sample_crop(image_batch, bboxs)
        landmarks = self.detect_landmarks(cropped)
        print(f"[INFO] Total Face Alignement Time: {time.time() - start:.4f} seconds")
        return landmarks

    def show_frame(
        self,
        image_batch: torch.Tensor,
        idx: int,
        bboxs: Optional[torch.Tensor]=None,
        landmarks: Optional[torch.Tensor]=None
    ):
        """
        Select a frame from the provided image batch and display it using matplotlib,
        optional display of bboxs / landmarks
        Args:
            image_batch: TCHW uint8
            idx: index of the frame
            bboxs: [B, 4]=>(idx)
            landmarks: [B, 68, 2]=>(idx)
        """
        B = image_batch.shape[0]
        assert idx < B, f"frame index out of range, max: {B}"

        frame = image_batch[idx]
        bbox = bboxs[idx] if bboxs is not None else None
        lds = landmarks[idx] if landmarks is not None else None

        plt.imshow(frame.permute(1, 2 ,0).cpu().numpy() / 255)
        if bbox is not None:
            x1, y1, x2, y2 = bbox.cpu().numpy()
            w, h = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='lime', facecolor='none')
            plt.gca().add_patch(rect)
        if lds is not None:
            lds = lds.cpu().numpy()
            x, y = lds[:, 0], lds[:, 1]
            plt.scatter(x, y, s=5, c='red', marker='.')
