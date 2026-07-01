from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, check_image
from ultralytics.data.base import BaseDataset
from ultralytics.data.dataset import YOLODataset
from pathlib import Path
import numpy as np
import cv2

IMG_FORMATS.add("npy")

_orig_check_image = check_image


def _patched_check_image(im_file):
    if Path(im_file).suffix.lower() == ".npy":
        data = np.load(im_file)
        h, w = data.shape[-2:]  # (C, H, W) or (H, W)
        return "", (h, w)
    return _orig_check_image(im_file)


import ultralytics.data.utils

ultralytics.data.utils.check_image = _patched_check_image

_orig_load_image = BaseDataset.load_image


def _patched_load_image(self, i, rect_mode=True, resize_short=False):
    im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
    if im is None:
        if fn.exists():
            try:
                data = np.load(fn)
                if data.ndim == 3 and data.shape[0] < data.shape[-1]:
                    data = np.transpose(data, (1, 2, 0))
                if data.dtype != np.uint8:
                    lo, hi = data.min(), data.max()
                    if hi > lo:
                        data = ((data - lo) / (hi - lo) * 255).astype(np.uint8)
                    else:
                        data = np.zeros(data.shape, dtype=np.uint8)
                im = data
            except Exception:
                pass
        if im is None:
            return _orig_load_image(self, i, rect_mode, resize_short)
        h0, w0 = im.shape[:2]
        if rect_mode and not resize_short and not (h0 == w0 == self.imgsz):
            r = self.imgsz / max(h0, w0)
            if r != 1:
                w, h = min(int(w0 * r), self.imgsz), min(int(h0 * r), self.imgsz)
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):
            im = cv2.resize(
                im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR
            )
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]
            self.buffer.append(i)
            if 1 < len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                if self.cache != "ram":
                    self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None
        return im, (h0, w0), im.shape[:2]
    return _orig_load_image(self, i, rect_mode, resize_short)


BaseDataset.load_image = _patched_load_image

model = YOLO("yolo11n_64ch.pt")
model.train(
    data="data.yaml",
    epochs=100,
    imgsz=256,
    batch=16,
    device="cpu",
    mosaic=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    scale=0.0,
)  # fmt: skip
