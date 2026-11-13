import os
import cv2
import random
import numpy as np
import pyzjr

def correct_cloud_color(
        cloud_bgra,
        sigma_s=0.15,
        sigma_v=0.85,
        color_mix=0.70,
        atmosphere_strength=None,
        atmospheric_bgr=(255.0, 255.0, 255.0),
        correction_mode="hsv",
    ):
    # 分离 BGR 和 Alpha 通道
    has_alpha = cloud_bgra.shape[2] == 4
    if has_alpha:
        bgr = cloud_bgra[..., :3]
        alpha = cloud_bgra[..., 3:]
    else:
        bgr = cloud_bgra
        alpha = None

    if atmosphere_strength is None:
        atmosphere_strength = 1.0 - sigma_v
    if correction_mode == "hsv":
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= sigma_s  # Reduce saturation to sigma_s
        hsv[..., 2] *= sigma_v  # Reduce the brightness to sigma_v
        hsv_bgr = cv2.cvtColor(
            np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_bgr = cv2.cvtColor(
            gray.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
        corrected_bgr = color_mix * hsv_bgr + (1.0 - color_mix) * gray_bgr
    elif correction_mode == "luma":
        bgr_float = bgr.astype(np.float32)
        weights_bgr = np.array([0.0722, 0.7152, 0.2126], dtype=np.float32)
        luma = np.sum(bgr_float * weights_bgr, axis=2, keepdims=True)

        atm_bgr = np.array(atmospheric_bgr, dtype=np.float32).reshape(1, 1, 3)
        atm_luma = np.sum(atm_bgr * weights_bgr, axis=2, keepdims=True)
        corrected_luma = luma + atmosphere_strength * (atm_luma - luma)
        corrected_bgr = corrected_luma + (1 - atmosphere_strength) * (bgr_float - luma)
        corrected_bgr += atmosphere_strength * (atm_bgr - atm_luma)
    else:
        raise ValueError("mode must be 'hsv' or 'luma'")
    corrected_cloud = np.clip(corrected_bgr, 0, 255).astype(np.uint8)
    if has_alpha:
        return np.concatenate([corrected_cloud, alpha.astype(np.uint8)], axis=2)
    return corrected_cloud


def random_clouds_fuse(
    background,
    clouds_path,
    a: float = 0.75,
    b: float = 1.2,
    sigma_s=0.15,
    sigma_v=0.85,
    color_mix=0.70,
    atmosphere_strength=None,
    atmospheric_bgr=(255.0, 255.0, 255.0),
    correction_mode="hsv",
    layers=1,
    prob=.5,
    seed: int | None = None,
):
    fused_image = background.copy()
    image_path_list = pyzjr.get_image_path(clouds_path, return_abs_path=True)

    fog_groups = {}
    for path in image_path_list:
        base_name = os.path.basename(path)
        fog_class = base_name.split('_')[0]  # Extract category prefix (e.g., fog1)
        if fog_class not in fog_groups:
            fog_groups[fog_class] = []
        fog_groups[fog_class].append(path)

    if seed is not None:
        random.seed(seed)
    # Randomly select cloud images from different categories
    selected_classes = random.sample(list(fog_groups.keys()), min(layers, len(fog_groups)))

    for fog_class in selected_classes:
        rand_path = random.choice(fog_groups[fog_class])
        rand_float = random.uniform(a, b)

        overlay = cv2.imread(rand_path, cv2.IMREAD_UNCHANGED)
        overlay = correct_cloud_color(
            overlay, sigma_s, sigma_v, color_mix,
            atmospheric_bgr=atmospheric_bgr, atmosphere_strength=atmosphere_strength,
            correction_mode=correction_mode
        )
        # overlay = np.flip(overlay, 0)
        if random.random() > prob:
            overlay = np.flip(overlay, 1)
        # else:
        #     overlay = pyzjr.random_rot90(overlay)
        fused_image = pyzjr.OverlayPng(fused_image, overlay, alpha_gain=rand_float)
        print(f"Use transparency gain {rand_float:.2f} combined {os.path.basename(rand_path)} with mode: {correction_mode}")
    return fused_image


if __name__=="__main__":
    background_path = r'E:\PythonProject\DehazeLab\data\RSHD\thick\test\GT'
    overlay_cloudmist_path = r'E:\PythonProject\DehazeLab\data\cloud_512'
    save_path = r'./result'
    # background = cv2.imread(background_path)
    # fused_image = random_clouds_fuse(background, overlay_cloudmist_path,
    #                                  correction_mode='hsv',
    #                                  layers=2)
    # fused_image2 = random_clouds_fuse(background, overlay_cloudmist_path,
    #                                   correction_mode='luma')
    #
    # pyzjr.show_image([fused_image, fused_image2])

    os.makedirs(save_path, exist_ok=True)
    bg_image_path_list = pyzjr.get_image_path(background_path, return_abs_path=True)
    for i in range(len(bg_image_path_list)):
        background_image = cv2.imread(bg_image_path_list[i])
        fused_image = random_clouds_fuse(
            background_image,
            overlay_cloudmist_path,
            a=0.75, b=1.2,
            sigma_s=0.15,
            sigma_v=0.85,
            color_mix=0.70,
            atmospheric_bgr=(255.0, 255.0, 255.0),
            atmosphere_strength=None,
            correction_mode="hsv",
            layers=3,
            prob=.5,
            seed=None
        )
        new_path = os.path.join(save_path, os.path.basename(bg_image_path_list[i]))
        print(f"{i + 1}: {bg_image_path_list[i]} -> {new_path}")
        cv2.imwrite(new_path, fused_image)
