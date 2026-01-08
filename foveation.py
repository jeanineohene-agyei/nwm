import torch.nn.functional as F

def foveate_image(x, fovea_frac=0.5):
    """Divide an image into high resolution crop and low resolution full image

    Args:
        x: shape (N, C, H, W) tensor of input images
        fovea_frac: fraction of image to crop for fovea (0.5 == half-width)

    Returns:
        x_global: (N, C, H_out, W_out)
        x_fovea:  (N, C, H_out, W_out)
    """
    
    _, _, H, W = x.shape
    
    crop_h, crop_w = int((H * fovea_frac)), int((W * fovea_frac))
    
    cy, cx = H // 2, W // 2
    y1 = cy - crop_h // 2
    y2 = cy + crop_h // 2
    x1 = cx - crop_w // 2
    x2 = cx + crop_w // 2

    x_fovea = x[:, :, y1:y2, x1:x2]
    x_fovea = F.interpolate(x_fovea, size=(H, W), mode='bilinear', align_corners=False)
    
    x_global_small = F.interpolate(x, scale_factor=fovea_frac, mode='bilinear', align_corners=False)
    x_global = F.interpolate(x_global_small, size=(H, W), mode='bilinear', align_corners=False)
    
    return x_global, x_fovea