import torch
from enum import IntEnum
import numpy as np
from tqdm import tqdm
from .utils import *
from face_alignment.detection.retina.pytorch_retinaface import Pytorch_RetinaFace


class LandmarksType(IntEnum):
    """Enum class defining the type of landmarks to detect.

    ``TWO_D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
    ``TWO_HALF_D`` - this points represent the projection of the 3D points into 3D
    ``THREE_D`` - detect the points ``(x,y,z)``` in a 3D space

    """
    TWO_D = 1
    TWO_HALF_D = 2
    THREE_D = 3


class NetworkSize(IntEnum):
    # TINY = 1
    # SMALL = 2
    # MEDIUM = 3
    LARGE = 4


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

class FaceAlignment:
    def __init__(self, device='cuda', dtype=torch.float32, flip_input=False, face_detector='retina', face_detector_kwargs=None, verbose=False):
        self.device = device
        self.flip_input = flip_input
        self.verbose = verbose
        self.dtype = dtype

        network_size = 4
        pytorch_version = torch.__version__
        if 'dev' in pytorch_version:
            pytorch_version = pytorch_version.rsplit('.', 2)[0]
        else:
            pytorch_version = pytorch_version.rsplit('.', 1)[0]

        if 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        # Get the face detector
        if face_detector == "retina":
            self.face_detector = Pytorch_RetinaFace(top_k=50, keep_top_k=10, device=device, confidence_threshold=0.5)
        else:
            face_detector_module = __import__('face_alignment.detection.' + face_detector,
                                            globals(), locals(), [face_detector], 0)
            face_detector_kwargs = face_detector_kwargs or {}
            self.face_detector = face_detector_module.FaceDetector(device=device, verbose=verbose, **face_detector_kwargs)

        # Initialise the face alignemnt networks
        network_name = '2DFAN-' + str(network_size)
        self.face_alignment_net = torch.jit.load(
            load_file_from_url(models_urls.get(pytorch_version, default_model_urls)[network_name]))

        self.face_alignment_net.to(device, dtype=dtype)
        self.face_alignment_net.eval()


    # def get_landmarks(self, image_or_path, detected_faces=None, return_bboxes=False, return_landmark_score=False):
    #     """Deprecated, please use get_landmarks_from_image

    #     Arguments:
    #         image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it

    #     Keyword Arguments:
    #         detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
    #         in the image (default: {None})
    #         return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
    #         return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
    #     """
    #     return self.get_landmarks_from_image(image_or_path, detected_faces, return_bboxes, return_landmark_score)

    # @torch.no_grad()
    # def get_landmarks_from_image(self, image_or_path, detected_faces=None, return_bboxes=False,
    #                              return_landmark_score=False):
    #     """Predict the landmarks for each face present in the image.

    #     This function predicts a set of 68 2D or 3D images, one for each image present.
    #     If detect_faces is None the method will also run a face detector.

    #      Arguments:
    #         image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

    #     Keyword Arguments:
    #         detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
    #         in the image (default: {None})
    #         return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
    #         return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

    #     Return:
    #         result:
    #             1. if both return_bboxes and return_landmark_score are False, result will be:
    #                 landmark
    #             2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
    #                 (landmark, landmark_score, detected_face)
    #                 (landmark, None,           detected_face)
    #                 (landmark, landmark_score, None         )
    #     """
    #     image = get_image(image_or_path)
    #     image = image_or_path

    #     if detected_faces is None:
    #         detected_faces = self.face_detector.detect_from_image(image.copy())

    #     if len(detected_faces) == 0:
    #         warnings.warn("No faces were detected.")
    #         if return_bboxes or return_landmark_score:
    #             return None, None, None
    #         else:
    #             return None

    #     landmarks = []
    #     landmarks_scores = []
    #     for _, d in enumerate(detected_faces):
    #         center = np.array([d[2] - (d[2] - d[0]) / 2.0, d[3] - (d[3] - d[1]) / 2.0])
    #         center[1] = center[1] - (d[3] - d[1]) * 0.12
    #         scale = (d[2] - d[0] + d[3] - d[1]) / self.face_detector.reference_scale

    #         inp = crop(image, center, scale)
    #         inp = torch.from_numpy(inp.transpose(
    #             (2, 0, 1))).float()

    #         inp = inp.to(self.device, dtype=self.dtype)
    #         inp.div_(255.0).unsqueeze_(0)

    #         # S1 = time.time()
    #         out = self.face_alignment_net(inp)
    #         # S2 = time.time()
    #         # print("Face alignement:", S2 - S1)

    #         if self.flip_input:
    #             out += flip(self.face_alignment_net(flip(inp)).detach(), is_label=True)
    #         out = out.to(device='cpu', dtype=torch.float32).numpy()

    #         pts, pts_img, scores = get_preds_fromhm(out, center, scale)
    #         pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
    #         pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
    #         scores = scores.squeeze(0)

    #         landmarks.append(pts_img.numpy())
    #         landmarks_scores.append(scores)

    #     if not return_bboxes:
    #         detected_faces = None
    #     if not return_landmark_score:
    #         landmarks_scores = None
    #     if return_bboxes or return_landmark_score:
    #         return landmarks, landmarks_scores, detected_faces
    #     else:
    #         return landmarks

    # @torch.no_grad()
    # def get_landmarks_from_batch(self, image_batch, detected_faces=None, return_bboxes=False,
    #                              return_landmark_score=False):
    #     """Predict the landmarks for each face present in the image.

    #     This function predicts a set of 68 2D or 3D images, one for each image in a batch in parallel.
    #     If detect_faces is None the method will also run a face detector.

    #      Arguments:
    #         image_batch {torch.tensor} -- The input images batch

    #     Keyword Arguments:
    #         detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
    #         in the image (default: {None})
    #         return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
    #         return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

    #     Return:
    #         result:
    #             1. if both return_bboxes and return_landmark_score are False, result will be:
    #                 landmarks
    #             2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
    #                 (landmark, landmark_score, detected_face)
    #                 (landmark, None,           detected_face)
    #                 (landmark, landmark_score, None         )
    #     """

    #     if detected_faces is None:
    #         start_time = time.time()
    #         detected_faces = self.face_detector.detect_from_batch(image_batch)
    #         print("Face detector time:", time.time() - start_time)

    #     if len(detected_faces) == 0:
    #         warnings.warn("No faces were detected.")
    #         if return_bboxes or return_landmark_score:
    #             return None, None, None
    #         else:
    #             return None

    #     landmarks = []
    #     landmarks_scores_list = []
    #     # A batch for each frame
    #     for i, faces in enumerate(detected_faces):
    #         res = self.get_landmarks_from_image(
    #             image_batch[i].cpu().numpy().transpose(1, 2, 0),
    #             detected_faces=faces,
    #             return_landmark_score=return_landmark_score,
    #         )
    #         if return_landmark_score:
    #             landmark_set, landmarks_scores, _ = res
    #             landmarks_scores_list.append(landmarks_scores)
    #         else:
    #             landmark_set = res
    #         # Bacward compatibility
    #         if landmark_set is not None:
    #             landmark_set = np.concatenate(landmark_set, axis=0)
    #         else:
    #             landmark_set = []
    #         landmarks.append(landmark_set)

    #     if not return_bboxes:
    #         detected_faces = None
    #     if not return_landmark_score:
    #         landmarks_scores_list = None
    #     if return_bboxes or return_landmark_score:
    #         return landmarks, landmarks_scores_list, detected_faces
    #     else:
    #         return landmarks

    # def get_landmarks_from_directory(self, path, extensions=['.jpg', '.png'], recursive=True, show_progress_bar=True,
    #                                  return_bboxes=False, return_landmark_score=False):
    #     """Scan a directory for images with a given extension type(s) and predict the landmarks for each
    #         face present in the images found.

    #      Arguments:
    #         path {str} -- path to the target directory containing the images

    #     Keyword Arguments:
    #         extensions {list of str} -- list containing the image extensions considered (default: ['.jpg', '.png'])
    #         recursive {boolean} -- If True, scans for images recursively (default: True)
    #         show_progress_bar {boolean} -- If True displays a progress bar (default: True)
    #         return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
    #         return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.
    #     """
    #     dataset = FolderData(path, self.face_detector.tensor_or_path_to_ndarray, extensions, recursive, self.verbose)
    #     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=4)
      
    #     predictions = {}
    #     for (image_path, image) in tqdm(dataloader, disable=not show_progress_bar):
    #         image_path, image = image_path[0], image[0]
    #         bounding_boxes = self.face_detector.detect_from_image(image)
    #         if return_bboxes or return_landmark_score:
    #             preds, bbox, score = self.get_landmarks_from_image(
    #                 image, bounding_boxes, return_bboxes=return_bboxes, return_landmark_score=return_landmark_score)
    #             predictions[image_path] = (preds, bbox, score)
    #         else:
    #             preds = self.get_landmarks_from_image(image, bounding_boxes)
    #             predictions[image_path] = preds

    #     return predictions



    @torch.no_grad()
    def fast_get_landmarks_from_batch(self, image_batch: torch.Tensor):
        """Biggest face in each frame"""

        image_batch = image_batch.permute(0, 2, 3, 1).to(dtype=torch.float32)
        detected_faces: list[list[int] | None] = []

        # bbox detection (retina)
        for i in tqdm(range(image_batch.shape[0]), desc="Detecting faces"):
            dets = self.face_detector.detect_face_bbox(image_batch[i])
            detected_faces.append(dets)

        landmarks = []
        # landmarks detection (FAN)
        for i, bbox in tqdm(enumerate(detected_faces), total=len(detected_faces), desc="Detecting landmarks"):
            if bbox is None:
                # print("Skipping index n°", i)
                landmarks.append(None)
                continue
            pt_img = self.fast_get_landmarks_from_image(
                image_batch[i],
                bbox=bbox,
                return_landmark_score=True,
            )

            landmarks.append(pt_img)
        return landmarks, detected_faces

    @torch.no_grad()
    def fast_get_landmarks_from_image(self, image_or_path: torch.Tensor, bbox: list[int], return_landmark_score=False):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

         Arguments:
            image_or_path {string or numpy.array or torch.tensor} -- The input image or path to it.

        Keyword Arguments:
            detected_faces {list of numpy.array} -- list of bounding boxes, one for each face found
            in the image (default: {None})
            return_bboxes {boolean} -- If True, return the face bounding boxes in addition to the keypoints.
            return_landmark_score {boolean} -- If True, return the keypoint scores along with the keypoints.

        Return:
            result:
                1. if both return_bboxes and return_landmark_score are False, result will be:
                    landmark
                2. Otherwise, result will be one of the following, depending on the actual value of return_* arguments.
                    (landmark, landmark_score, detected_face)
                    (landmark, None,           detected_face)
                    (landmark, landmark_score, None         )
        """
        image = image_or_path.numpy()
        
        x1, y1, x2, y2 = bbox

        center = np.array([x2 - (x2 - x1) / 2.0, y2 - (y2 - y1) / 2.0])
        center[1] = center[1] - (y2 - y1) * 0.12
        scale = (x2 - x1 + y2 - y1) / self.face_detector.reference_scale

        inp = crop(image, center, scale)
        inp = torch.from_numpy(inp.transpose(
            (2, 0, 1))).float()

        inp = inp.to(self.device)
        inp.div_(255.0).unsqueeze_(0)

        # net_start = time.time()
        out = self.face_alignment_net(inp)
        # tqdm.write(f"FAN éxecuté en {time.time() - net_start:.4f} secondes")
        out = out.to(device='cpu', dtype=torch.float32).numpy()

        pts, pts_img, scores = get_preds_fromhm(out, center, scale)
        pts, pts_img = torch.from_numpy(pts), torch.from_numpy(pts_img)
        pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
        scores = scores.squeeze(0)

        return pts_img.numpy()
    
    def box_rescale(self, bbox_batch: list[list[int]], scale_factor=3, shift_factor=0.5, aspect_ratio=1.0):
        # cropped_imgs = []
        bbox_infos: list[tuple[int, int, int, int]] = []

        for _, bbox in enumerate(bbox_batch):
            x1, y1, x2, y2 = map(int, bbox)
            face_width = x2 - x1
            face_height = y2 - y1

            default_area = face_width * face_height
            default_area *= scale_factor
            default_side = math.sqrt(default_area)

            # New height and width based on aspect_ratio
            new_face_width = int(default_side * math.sqrt(aspect_ratio))
            new_face_height = int(default_side / math.sqrt(aspect_ratio))

            # Center coordinates of the detected face
            center_x = x1 + face_width // 2
            center_y = y1 + face_height // 2 + int(new_face_height * (0.5 - shift_factor))

            original_crop_x1 = center_x - new_face_width // 2
            original_crop_x2 = center_x + new_face_width // 2
            original_crop_y1 = center_y - new_face_height // 2
            original_crop_y2 = center_y + new_face_height // 2
            # # Crop coordinates, adjusted to the image boundaries
            # crop_x1 = max(0, original_crop_x1)
            # crop_x2 = min(image.shape[1], original_crop_x2)
            # crop_y1 = max(0, original_crop_y1)
            # crop_y2 = min(image.shape[0], original_crop_y2)

            # # Crop the region and add padding to form a square
            # cropped_imgs.append(image[crop_y1:crop_y2, crop_x1:crop_x2])
            bbox_infos.append((original_crop_x1, original_crop_y1, original_crop_x2, original_crop_y2))
        return bbox_infos
    
    def crop_pad_interpolate(self, bbox_batch: list[tuple[int, int, int, int]], image_batch: torch.Tensor):
        """
        Crops faces and pad negative coordinates, interpolates to 512x512
        Returns:
            tuple:
                - cropped: torch.Tensor of shape (N, 3, 512, 512)
                - frames_padding: list of tuples (pad_left, pad_right, pad_top, pad_bottom)
        """
        assert len(bbox_batch) == len(image_batch)
        frames: list[torch.Tensor] = []
        frames_padding: list[tuple[int, int, int, int]] = []
        height, width = image_batch.shape[2], image_batch.shape[3]
        for i, bbox in tqdm(enumerate(bbox_batch), total=len(bbox_batch)):
            x1, y1, x2, y2 = bbox
            # side = x2 - x1

            crop_x1 = max(0, x1)
            crop_x2 = min(width, x2)
            crop_y1 = max(0, y1)
            crop_y2 = min(height, y2)

            pad_left = crop_x1 - x1
            pad_right = x2 - crop_x2
            pad_bottom = crop_y1 - y1
            pad_top = y2 - crop_y2

            cropped = image_batch[i][:, crop_y1:crop_y2, crop_x1:crop_x2]

            if pad_left or pad_bottom or pad_right or pad_top:
                padded = torch.nn.functional.pad(
                    cropped, 
                    (pad_left, pad_right, pad_top, pad_bottom), 
                    value=0
                ).unsqueeze(0)
            else:
                padded = cropped.unsqueeze(0)

            interpolated = torch.nn.functional.interpolate(padded, size=(512, 512), mode="bilinear")
            
            # if side > 512:
            #     # => Downscaling, best algo is area
            #     interpolated = torch.nn.functional.interpolate(padded, size=(512, 512), mode="area")
            # elif side < 512:
            #     # => Upscaling, best algo that is not bicubic(bugs?) is bilinear
            #     interpolated = torch.nn.functional.interpolate(padded, size=(512, 512), mode="bilinear")
            # else:
            #     # No interpolation needed
            #     interpolated = padded.unsqueeze(0)

            frames.append(interpolated)
            frames_padding.append((pad_left, pad_right, pad_top, pad_bottom))
        
        return torch.cat(frames, dim=0), frames_padding

    def __call__(self, video: torch.Tensor):
        """
        Args:
            - video: TCHW uint8 Tensor
        Returns:
            tuple:
            - landmarks: list[ NDArray(68, 2) ]
            - cropped_frames: torch.Tensor of shape (N, 3, 512, 512)
            - boxes: bboxs list[ (x1, y1, x2, y2) ]
            - frames_padding: list[ (pad_left, pad_right, pad_top, pad_bottom) ]
        """
        start = time.time()

        landmarks, boxes = self.fast_get_landmarks_from_batch(video) # List[ NDArray(68, 2) ], List[ List[ int ] ]
        boxes = fill_none_with_precedent(boxes, n=2) # !!! Temporaire: Replace up to 2 consecutive None values
        boxes = self.box_rescale(boxes)
        cropped_frames, frames_padding = self.crop_pad_interpolate(boxes, video)

        tqdm.write(f"TOTAL DETECTION TIME: {time.time() - start:.4f}")

        return landmarks, cropped_frames, boxes, frames_padding