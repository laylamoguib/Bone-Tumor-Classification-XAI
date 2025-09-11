import cv2
import timm
import torch
import numpy
from torch import nn
torch.use_deterministic_algorithms(True)

#preprocessing
def padding(img, size: int = 640):
    height, width = img.shape[:2]
    scale = min(size/width, size/height)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(img, (new_width, new_height))
    canvas = numpy.zeros((size, size, 3), dtype=numpy.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas

def prepare_display_and_tensor(image_path: str):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    rgb_sq = padding(rgb, 640)
    img_f = (rgb_sq.astype(numpy.float32) / 255.0 - numpy.array([0.485, 0.456, 0.406], numpy.float32)) / numpy.array([0.229, 0.224, 0.225], numpy.float32)
    tensor = torch.from_numpy(img_f).permute(2, 0, 1).unsqueeze(0).to(torch.device("cpu"))
    return rgb_sq, tensor

def is_tumor(class_name: str):
    name = class_name.lower().replace("_", "").replace(" ", "")
    return any(k in name for k in ["tumor", "benign", "malignant"])

def overlay_cam_on_image(img_rgb: numpy.ndarray, cam01: numpy.ndarray, alpha=0.5):
    heat = cv2.applyColorMap((cam01 * 255).astype(numpy.uint8), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb, 1 - alpha, heat_rgb, alpha, 0)

def norm01(a: numpy.ndarray):
    a = a.astype(numpy.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)

# Model
class EfficientNetB3(nn.Module):
    def __init__(self, model_name='efficientnet_b3', ctx_dim=5):
        super().__init__()
        self.encoder = timm.create_model(model_name, pretrained=False, features_only=True)
        ch = self.encoder.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Linear(ch, 1)
        self.ctx_mlp = nn.Sequential(nn.Linear(ctx_dim, 32), nn.ReLU(True), nn.BatchNorm1d(32),nn.Dropout(0.10), nn.Linear(32, 16), nn.ReLU(True))
        self.fusion_head = nn.Linear(ch + 16, 1)

    def forward(self, image_tensor, context_tensor=None):
        feat = self.encoder(image_tensor)[-1]
        pooled = self.pool(feat).flatten(1)
        img_only = self.cls_head(pooled)
        if context_tensor is not None:
            ctx = self.ctx_mlp(context_tensor)
            fused = self.fusion_head(torch.cat([pooled, ctx], dim=1))
        else:
            fused = img_only
        return fused, None, img_only

class _ImgOnlyWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        index, index, img_only = self.m(x, None)
        return img_only

def get_target_conv(model: nn.Module):
    last = None
    for m in model.encoder.modules():
        if isinstance(m, nn.Conv2d): last = m
    return last