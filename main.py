from torchvision.io import read_video
import face_alignment
import time
from tqdm import tqdm

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='retina', device='cuda')

loaf_n_frame = 5
vid = read_video("./s42.mp4", output_format="TCHW")[0][:loaf_n_frame]
print("Video shape:", vid.shape)

# h, w = vid.shape[2], vid.shape[3]
# vid.to(dtype=torch.float32)
# vid = torch.nn.functional.interpolate(vid, (h//2, w//2), mode="bilinear")
# vid.to(dtype=torch.uint8)

# black_frame = torch.zeros((1, 3, 512, 512), dtype=torch.uint8)
# vid = torch.cat((vid, black_frame))

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

start = time.time()

points, boxes = fa.fast_get_landmarks_from_batch(vid) # List[ NDArray(68, 2) ], List[ List[ int ] ]
boxes = fill_none_with_precedent(boxes, n=2) # !!! Temporaire: Replace up to 2 consecutive None values
boxes = fa.box_rescale(boxes)
cropped, frames_padding = fa.crop_pad_interpolate(boxes, vid)

tqdm.write(f"TOTAL DETECTION TIME: {time.time() - start:.4f}")