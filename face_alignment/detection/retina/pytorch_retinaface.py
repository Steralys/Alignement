# import os
# import math
import torch
from torchvision.ops import batched_nms, nms
# import numpy as np
from tqdm import tqdm

from .data import cfg_mnet, cfg_re50
from .layers.functions.prior_box import PriorBox
# from .utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from .utils.box_utils import batch_decode, batch_decode_eyes #, decode, 


models_urls = {
    "pretrained_path": "./detection/retina/weights/mobilenet0.25_Final.pth",
    "weights_path": "./detection/retina/weights/mobilenetV1X0.25_pretrain.tar",
    # "pretrained_path": "./face_alignment/detection/retina/weights/mobilenet0.25_Final.pth",
    # "weights_path": "./face_alignment/detection/retina/weights/mobilenetV1X0.25_pretrain.tar",
}



class Pytorch_RetinaFace:
    def __init__(
            self,
            cfg="mobile0.25",
            pretrained_path=models_urls["pretrained_path"],
            weights_path=models_urls["weights_path"],
            device="cuda",
            vis_thres=0.6,
            top_k=5000,
            keep_top_k=750,
            nms_threshold=0.4,
            confidence_threshold=0.02
        ):
        self.vis_thres = vis_thres
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.confidence_threshold = confidence_threshold
        self.cfg = cfg_mnet if cfg=="mobile0.25" else cfg_re50
        
        self.device = device

        # print("Using device:", self.device)

        self.net = RetinaFace(cfg=self.cfg, weights_path=weights_path, phase='test', device=self.device).to(self.device)
        self.load_model_weights(pretrained_path)
        self.net.eval()

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True


    def remove_prefix(self, state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}


    def load_model_weights(self, pretrained_path):
        # pretrained_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), pretrained_path)
        print('Loading pretrained model from {}'.format(pretrained_path))
        pretrained_dict = torch.load(pretrained_path, map_location=self.device)
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(self.net, pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.to(self.device)
        return self.net

    # def center_and_crop_rescale(self, image, dets, scale_factor=4, shift_factor=0.35, aspect_ratio=1.0):
    #     cropped_imgs = []
    #     bbox_infos = []
    #     for index, bbox in enumerate(dets):
    #         if bbox[4] < self.vis_thres:
    #             continue

    #         x1, y1, x2, y2 = map(int, bbox[:4])
    #         face_width = x2 - x1
    #         face_height = y2 - y1

    #         default_area = face_width * face_height
    #         default_area *= scale_factor
    #         default_side = math.sqrt(default_area)

    #         # New height and width based on aspect_ratio
    #         new_face_width = int(default_side * math.sqrt(aspect_ratio))
    #         new_face_height = int(default_side / math.sqrt(aspect_ratio))

    #         # Center coordinates of the detected face
    #         center_x = x1 + face_width // 2
    #         center_y = y1 + face_height // 2 + int(new_face_height * (0.5 - shift_factor))

    #         original_crop_x1 = center_x - new_face_width // 2
    #         original_crop_x2 = center_x + new_face_width // 2
    #         original_crop_y1 = center_y - new_face_height // 2
    #         original_crop_y2 = center_y + new_face_height // 2
    #         # Crop coordinates, adjusted to the image boundaries
    #         crop_x1 = max(0, original_crop_x1)
    #         crop_x2 = min(image.shape[1], original_crop_x2)
    #         crop_y1 = max(0, original_crop_y1)
    #         crop_y2 = min(image.shape[0], original_crop_y2)

    #         # Crop the region and add padding to form a square
    #         cropped_imgs.append(image[crop_y1:crop_y2, crop_x1:crop_x2])
    #         bbox_infos.append(((original_crop_x2-original_crop_x1, original_crop_y2-original_crop_y1),(original_crop_x1, original_crop_y1, original_crop_x2, original_crop_y2)))
    #     return cropped_imgs, bbox_infos

    # def detect_faces(self, img):
    #     resize = 1

    #     if len(img.shape) == 4 and img.shape[0] == 1:
    #         img = torch.squeeze(img, 0)
    #     if isinstance(img, np.ndarray):
    #         img = torch.from_numpy(img)
    #     img = img.to(self.device)

    #     im_height, im_width, _ = img.shape
    #     scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    #     mean_values = torch.tensor([104, 117, 123], dtype=torch.float32).to(self.device)
    #     mean_values = mean_values.view(1, 1, 3)
    #     img -= mean_values
    #     img = img.permute(2, 0, 1)
    #     img = img.unsqueeze(0)
    #     scale = scale.to(self.device)

    #     # tic = time.time()
    #     with torch.no_grad():
    #         loc, conf, landms = self.net(img)  # forward pass
    #     # print('net forward time: {:.4f}'.format(time.time() - tic))

    #     priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
    #     priors = priorbox.forward()
    #     priors = priors.to(self.device)
    #     prior_data = priors.data
    #     boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
    #     boxes = boxes * scale / resize
    #     boxes = boxes.cpu().numpy()
    #     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    #     landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
    #     scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                             img.shape[3], img.shape[2], img.shape[3], img.shape[2],
    #                             img.shape[3], img.shape[2]])
    #     scale1 = scale1.to(self.device)
    #     landms = landms * scale1 / resize
    #     landms = landms.cpu().numpy()

    #     # Ignore low scores
    #     inds = np.where(scores > self.confidence_threshold)[0]
    #     boxes = boxes[inds]
    #     landms = landms[inds]
    #     scores = scores[inds]

    #     # Keep top-K before NMS
    #     order = scores.argsort()[::-1][:self.top_k]
    #     boxes = boxes[order]
    #     landms = landms[order]
    #     scores = scores[order]

    #     # Perform NMS
    #     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #     keep = py_cpu_nms(dets, self.nms_threshold)
    #     dets = dets[keep, :]
    #     landms = landms[keep]

    #     # Keep top-K faster NMS
    #     dets = dets[:self.keep_top_k, :]
    #     landms = landms[:self.keep_top_k, :]

    #     dets = np.concatenate((dets, landms), axis=1)
    #     return dets
    
    # def detect_face_bbox(self, img: torch.Tensor) -> list[int] | None:
    #     """
    #     HWC input
    #     Returns list[int], x1, y1, x2, y2
    #     Returns none if no face detected
    #     """
    #     resize = 1
    #     img = img.to(self.device)

    #     im_height, im_width, _ = img.shape
    #     scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    #     mean_values = torch.tensor([104, 117, 123], dtype=torch.float32).to(self.device)
    #     mean_values = mean_values.view(1, 1, 3)
    #     img -= mean_values
    #     img = img.permute(2, 0, 1)
    #     img = img.unsqueeze(0)
    #     scale = scale.to(self.device)

    #     # tic = time.time()
    #     with torch.no_grad():
    #         loc, conf, landms = self.net(img)  # forward pass
    #     # print('net forward time: {:.4f}'.format(time.time() - tic))

    #     priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
    #     priors = priorbox.forward()
    #     priors = priors.to(self.device)
    #     prior_data = priors.data
    #     boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
    #     boxes = boxes * scale / resize
    #     boxes = boxes.cpu().numpy()
    #     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

    #     # Ignore low scores
    #     inds = np.where(scores > self.confidence_threshold)[0]
    #     boxes = boxes[inds]
    #     scores = scores[inds]

    #     # Keep top-K before NMS
    #     order = scores.argsort()[::-1][:self.top_k]
    #     boxes = boxes[order]
    #     scores = scores[order]

    #     # Perform NMS
    #     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
    #     keep = py_cpu_nms(dets, self.nms_threshold)
    #     dets = dets[keep, :]

    #     # Just keep the biggest bbox, discard the others
    #     surface = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
    #     if surface.size == 0:
    #         return None
    #     max_idx = np.argmax(surface)
    #     return dets[max_idx, :-1].astype(np.int16).tolist() # Score not needed
    
    def detect_face_v2(self, image_batch: torch.Tensor, batch_size = 16, mode="v1"):
        """
        TCHW input
        Returns list[int], x1, y1, x2, y2
        Returns none if no face detected
        """

        if mode == "v1":
            process_batch = self._process_batch
        elif mode == "v2":
            process_batch = self._process_batch_v2
        else:
            raise ValueError("Invalid mode. Use 'v1' or 'v2'.")
        
        B, C, H, W = image_batch.shape
        scale = torch.Tensor([W, H, W, H]).to(self.device)
        mean_values = torch.tensor([104, 117, 123], device=self.device).view(1, 3, 1, 1)

        priorbox = PriorBox(self.cfg, image_size=(H, W))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data

        boxes = []
        no_face_frame_idx = []
        landmarks = []
        for i in tqdm(range(0, B, batch_size), desc="Detecting faces"):
            if i + batch_size > B:
                # Last batch may be smaller than batch_size
                batch_slice = image_batch[i:]
                # repeat the last frame to fill the batch for compile purpose
                # batch_slice = torch.cat([
                #     batch_slice, 
                #     image_batch[-1:].repeat(last_batch_padding, 1, 1, 1)
                # ], dim=0)
            else:
                batch_slice = image_batch[i:i+batch_size] 

            out, nfidx, lds = process_batch(
                batch_slice.to(dtype=torch.float32, device=self.device) - mean_values,
                prior_data=prior_data,
                scale=scale,
                resize=1
            )
            boxes.append(out)
            no_face_frame_idx.extend(nfidx)
            landmarks.append(lds)
            
        torch.cuda.empty_cache()

        boxes = torch.cat(boxes, dim=0)
        landmarks = torch.cat(landmarks, dim=0)
        landmarks = landmarks.reshape(landmarks.shape[0], 2, 2) # [B, (xa, ya, xb, yb)] => [B, 2, 2]
        return boxes, landmarks
        # return boxes[:-last_batch_padding]

    @torch.no_grad()
    def _process_batch(self, image_batch: torch.Tensor, prior_data, scale, resize):
        loc, conf, landms = self.net(image_batch)  # forward pass B, C, H, W, not tested with batch size > 1

        boxes = batch_decode(loc, prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        landms = batch_decode_eyes(landms, prior_data, self.cfg['variance'])
        landms = landms * scale / resize
        scores = conf[:, :, 1]

        # Ignore low scores
        inds = torch.nonzero(scores > self.confidence_threshold, as_tuple=False)
        # inds shape: [N, 2], where inds[:,0] is batch index, inds[:,1] is box index
        selected_boxes = boxes[inds[:, 0], inds[:, 1]]
        selected_landms = landms[inds[:, 0], inds[:, 1]]
        selected_scores = scores[inds[:, 0], inds[:, 1]]

        keep = batched_nms(
            selected_boxes,
            selected_scores,
            inds[:, 0],  # Use batch index for NMS
            self.nms_threshold
        )

        frame_inds = inds[keep, 0]
        selected_boxes = selected_boxes[keep]
        selected_scores = selected_scores[keep]
        selected_landms = selected_landms[keep]
        surfaces = (selected_boxes[:, 2] - selected_boxes[:, 0]) * (selected_boxes[:, 3] - selected_boxes[:, 1])
    
        _, sorted_idx = torch.sort(frame_inds * 10**7 + surfaces, descending=True)
        sorted = frame_inds[sorted_idx]
        _, counts = torch.unique_consecutive(
            sorted.float(),
            return_inverse=False,
            return_counts=True,
        )
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=self.device), cum_sum[:-1]))
        max_indices = sorted_idx[cum_sum]

        # Need to implement no face detected case
        return selected_boxes[max_indices].flip(0), [], selected_landms[max_indices].flip(0)
    
    @torch.no_grad()
    def _process_batch_v2(self, image_batch: torch.Tensor, prior_data, scale, resize):
        loc, conf, landms = self.net(image_batch)

        boxes = batch_decode(loc, prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize # B, N, 4
        scores = conf[:, :, 1] # B, N

        new_boxes = []
        no_face_frame_idx = []
        # Looping is faster because it removed the complexity of the batch_nms post processing
        for i in range(image_batch.shape[0]):
            inds = torch.nonzero(scores[i] > self.confidence_threshold, as_tuple=False)
            if inds.shape[0] == 0:
                no_face_frame_idx.append(i)
                continue
            selected_boxes = boxes[i, inds].squeeze(1)
            selected_scores = scores[i, inds].squeeze(1)
            keep = nms(selected_boxes, selected_scores, self.nms_threshold)
            selected = selected_boxes[keep]

            # Only append the largest box per image
            surface = (selected[:, 2] - selected[:, 0]) * (selected[:, 3] - selected[:, 1])
            max_surface_idx = torch.argmax(surface)
            new_boxes.append(selected[max_surface_idx])

        result = torch.stack(new_boxes)
        # assert result.shape[0] == 7 and result.shape[1] == 4, f"Expected 7 boxes, got {result}"
        return result, no_face_frame_idx

    @property
    def reference_scale(self):
        return 195.0