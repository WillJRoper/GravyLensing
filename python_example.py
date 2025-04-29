# gravity_lens.py
# Half-resolution FFT-based gravitational lens for speed

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import (
    DeepLabV3_MobileNet_V3_Large_Weights,
    deeplabv3_mobilenet_v3_large,
)

# -----------------------
# Configuration & Dependencies
# -----------------------
# Requires:
#   torch >= 1.12
#   torchvision >= 0.13
#   opencv-python
#   numpy
# Installation:
#   pip install torch torchvision opencv-python numpy

# -----------------------
# Model Loading
# -----------------------


def load_deeplab_model(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.to(device)
    model.eval()
    return model, device


# -----------------------
# Preprocessing
# -----------------------


def build_preprocess(input_size=(320, 320)):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


# -----------------------
# Person Mask Extraction
# -----------------------


def get_person_mask(frame, model, device, preprocess):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)["out"][0]
    pred = out.argmax(0).byte().cpu().numpy()
    mask = (pred == 15).astype("float32")
    # resize back to frame
    mask = cv2.resize(
        mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    return mask


# -----------------------
# FFT-based Kernel Precomputation
# -----------------------


def build_lens_kernels(h, w, softening=30.0, pad_factor=2):
    ph = pad_factor * h
    pw = pad_factor * w
    y = np.arange(ph) - ph // 2
    x = np.arange(pw) - pw // 2
    X, Y = np.meshgrid(x, y)
    r2 = X**2 + Y**2 + softening**2
    Kx = X / (r2 * np.pi)
    Ky = Y / (r2 * np.pi)
    Kx = np.fft.ifftshift(Kx)
    Ky = np.fft.ifftshift(Ky)
    Kx_ft = np.fft.fft2(Kx)
    Ky_ft = np.fft.fft2(Ky)
    return Kx_ft, Ky_ft, ph, pw


# -----------------------
# Half-Res FFT Convolution Lensing
# -----------------------


def apply_gravity_lens_fft_halfres(
    bg_full,
    mask_full,
    Kx_ft,
    Ky_ft,
    strength=1.0,
    softening=30.0,
    pad_h=0,
    pad_w=0,
    scale=0.5,
):
    # downsample
    h_f, w_f = mask_full.shape
    h, w = int(h_f * scale), int(w_f * scale)
    mask = cv2.resize(mask_full, (w, h), interpolation=cv2.INTER_NEAREST)
    bg = cv2.resize(bg_full, (w, h), interpolation=cv2.INTER_AREA)

    # blur full-res mask for blending later
    mask_blur = cv2.GaussianBlur(mask_full, (31, 31), sigmaX=15)[..., None]
    mask_blur = np.clip(mask_blur, 0, 1)

    # pad small mask
    pad_y = (pad_h - h) // 2
    pad_x = (pad_w - w) // 2
    mask_pad = np.pad(mask, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")

    # FFT conv
    conv_ft = np.fft.fft2(mask_pad)
    def_x_pad = np.real(np.fft.ifft2(conv_ft * Kx_ft)) * strength
    def_y_pad = np.real(np.fft.ifft2(conv_ft * Ky_ft)) * strength
    # crop
    def_x = def_x_pad[pad_y : pad_y + h, pad_x : pad_x + w]
    def_y = def_y_pad[pad_y : pad_y + h, pad_x : pad_x + w]
    # zero inside mask
    def_x[mask > 0.5] = 0
    def_y[mask > 0.5] = 0

    # remap small
    gy, gx = np.indices((h, w), dtype=np.float32)
    map_x = np.clip(gx - def_x, 0, w - 1).astype("float32")
    map_y = np.clip(gy - def_y, 0, h - 1).astype("float32")
    lensed_small = cv2.remap(bg, map_x, map_y, interpolation=cv2.INTER_LINEAR)

    # upsample back to full
    lensed_bg = cv2.resize(lensed_small, (w_f, h_f), interpolation=cv2.INTER_LINEAR)

    return lensed_bg, mask_blur


# -----------------------
# Main Live Loop
# -----------------------
if __name__ == "__main__":
    model, device = load_deeplab_model()
    preprocess = build_preprocess((320, 320))  # half input size
    scale = 0.5  # downscale factor for lensing

    cap = cv2.VideoCapture(0)
    bg_path = "/Users/willroper/Downloads/Euclid_s_extragalactic_view_in_Southern_Sky_patch.tif"
    bg_full = cv2.imread(bg_path)
    ret, frame = cap.read()
    h_f, w_f = frame.shape[:2]
    bg_full = cv2.resize(bg_full, (w_f, h_f), interpolation=cv2.INTER_AREA)

    # kernels at half res
    Kx_ft, Ky_ft, ph, pw = build_lens_kernels(
        int(h_f * scale), int(w_f * scale), softening=50, pad_factor=2
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        mask_full = get_person_mask(frame, model, device, preprocess)
        lensed_bg, mask_blur = apply_gravity_lens_fft_halfres(
            bg_full,
            mask_full,
            Kx_ft,
            Ky_ft,
            strength=1,
            pad_h=ph,
            pad_w=pw,
            scale=scale,
        )
        # out = ((1 - mask_blur) * lensed_bg + mask_blur * frame).astype("uint8")
        out = lensed_bg.astype("uint8")
        cv2.imshow("Gravitational Lens", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
