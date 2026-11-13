import math
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from PIL import Image


def read_rgb(path):
    rgb = Image.open(path).convert('RGB')
    return np.asarray(rgb, np.float64) / 255


def save_rgb(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)


def minimum_filter(image, size):
    """Separable minimum filter with replicated borders; odd square kernel."""
    if not isinstance(size, (int, np.integer)) or size < 1 or size % 2 == 0:
        raise ValueError('airlight window must be a positive odd integer')
    p = size//2
    v = np.lib.stride_tricks.sliding_window_view(
        np.pad(image, ((0, 0), (p, p)), mode='edge'), size, axis=1)
    horizontal = v.min(axis=-1)
    v = np.lib.stride_tricks.sliding_window_view(
        np.pad(horizontal, ((p, p), (0, 0)), mode='edge'), size, axis=0)
    return v.min(axis=-1)


def estimate_airlight(image, size=15, method='boundary'):
    """per-channel maximum of local minima."""
    if method == 'boundary':
        A = np.array([minimum_filter(image[..., c], size).max() for c in range(3)])
    elif method == 'dark_channel':
        dark = minimum_filter(image.min(2), size)
        k = max(1, math.ceil(0.001*dark.size))
        indices = np.argpartition(dark.ravel(), -k)[-k:]
        pixels = image.reshape(-1, 3)[indices]
        A = pixels[np.argmax(pixels.mean(1))]
    else:
        raise ValueError('airlight method must be boundary or dark_channel')
    return np.clip(A, 1e-6, 1.0)


def box_sum(image, radius):
    """O(HW) integral-image box sum; windows truncated at image boundaries."""
    h, w = image.shape
    yy, xx = np.arange(h), np.arange(w)
    y0, y1 = np.maximum(yy-radius, 0), np.minimum(yy+radius+1, h)
    x0, x1 = np.maximum(xx-radius, 0), np.minimum(xx+radius+1, w)
    s = np.pad(np.asarray(image, np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return s[y1[:, None], x1]-s[y0[:, None], x1]-s[y1[:, None], x0]+s[y0[:, None], x0]


def guided_filter(guide, src, radius=60, eps=1e-3):
    """grayscale GIF with truncated windows and variable pixel count."""
    if not isinstance(radius, (int, np.integer)) or radius<0:
        raise ValueError('guided radius must be a nonnegative integer')
    if not np.isfinite(eps) or eps<=0:
        raise ValueError('guided eps must be positive and finite')
    count = box_sum(np.ones_like(src), radius)
    def mean(x):
        return box_sum(x, radius)/count
    mg, mp = mean(guide), mean(src)
    # var: E[X^2] - (E[X])^2)
    var = np.maximum(mean(guide*guide)-mg*mg, 0)
    # cov: E[XY] - E[X]E[Y]
    cov = mean(guide*src)-mg*mp
    a = cov/(var+eps)
    b = mp-a*mg
    return mean(a)*guide+mean(b)


def gamma_taylor(theta, q, eta=1.8):
    """Eq.(18)-(20), stable small root: 2c/(b+sqrt(b*b-4ac)).

    Eq.(19): c=1-(1-q)/(1-q**eta). Use expm1 near q=1.
    At theta=0, gamma=0 yields the exact limiting transmission t=1.
    """
    if not np.isfinite(eta) or eta<1:
        raise ValueError('eta must be finite and >=1; paper default 1.8')
    r_mean = q**eta                                 # Eq.(24)
    one_minus_r = -np.expm1(eta*np.log(q))
    a = theta**2/2
    b = a+theta
    c = 1-(1-q)/one_minus_r                         # Eq.(19)
    discriminant = b*b-4*a*c
    den = b+np.sqrt(np.maximum(discriminant, 0))
    raw = np.divide(2*c, den, out=np.zeros_like(c), where=den>1e-14)
    return np.clip(raw, 0, 1), r_mean, raw, discriminant


def gamma_pade(theta, q, eta=1.8):
    """
    基于帕德逼近（Padé Approximant）替代原版泰勒展开的二次方程
    """
    if not np.isfinite(eta) or eta < 1:
        raise ValueError('eta must be finite and >=1; paper default 1.8')
    r_mean = q ** eta
    # 根据推导公式: gamma = [ (q - R_bar) * (2 - theta) ] / [ theta * (2 - q - R_bar) ]
    numerator = (q - r_mean) * (2.0 - theta)
    denominator = theta * (2.0 - q - r_mean)
    denominator = np.where(np.abs(denominator) < 1e-8, 1e-8, denominator)

    raw = numerator / denominator
    return np.clip(raw, 0, 1), r_mean, raw, denominator


def IHDCP(image, eta=1.8, guided_radius=60, guided_eps=1e-3,
           airlight_window=15, airlight_method='boundary', approximation='t'):
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[2] != 3 or min(source.shape[:2]) < 1:
        raise ValueError('expected nonempty H x W x 3 RGB')
    rgb = source.astype(np.float64)
    if source.dtype == np.uint8:
        rgb /= 255
    if not np.isfinite(rgb).all() or rgb.min()<0 or rgb.max()>1:
        raise ValueError('use uint8 RGB or finite float RGB in [0,1]')

    A = estimate_airlight(rgb, airlight_window, airlight_method)
    A_mean = float(A.mean())
    theta_raw = rgb.min(2)/A_mean
    q_raw = rgb.mean(2)/A_mean
    delta = 1e-6
    # Real bright/colored pixels can violate the paper's domain assumptions.
    theta = np.clip(theta_raw, 0, 1-delta)
    q = np.clip(q_raw, delta, 1-delta)

    # gamma, r_mean, gamma_raw, disc = gamma_taylor(theta, q, eta)
    # gamma, r_mean, gamma_raw, disc = gamma_pade(theta, q, eta)
    if approximation == 't':
        gamma, r_mean, gamma_raw, disc = gamma_taylor(theta, q, eta)
    elif approximation == 'p':
        gamma, r_mean, gamma_raw, disc = gamma_pade(theta, q, eta)
    else:
        raise ValueError("approximation must be 't' or 'p'")
    transmission = (1-theta)**gamma                 # Eq.(7)
    k = max(1, math.ceil(0.05*transmission.size))
    tmin = float(np.partition(transmission.ravel(), k-1)[:k].mean())
    t_clamped = np.maximum(transmission, tmin)       # Eq.(25): before GIF
    guide = (0.298936021293776*rgb[..., 0]+0.587043074451121*rgb[..., 1]
             +0.114020904255103*rgb[..., 2])
    t_gif = guided_filter(guide, t_clamped, guided_radius, guided_eps)
    t_used = np.clip(t_gif, delta, 1)
    recovered_raw = 1-(1-rgb/A)/t_used[..., None]     # Eq.(25), albedo R
    result = np.clip(recovered_raw, 0, 1)

    maps = dict(theta=theta, inverted_density=1-theta, albedo_mean=r_mean,
                gamma=gamma, gamma_unprojected=gamma_raw,
                transmission=transmission, transmission_clamped=t_clamped,
                transmission_refined=t_used)
    metadata = dict(atmospheric_light_rgb=A.tolist(), eta=float(eta),
                    airlight_method=airlight_method,
                    airlight_window=int(airlight_window), guided_radius=int(guided_radius),
                    guided_eps=float(guided_eps), tmin=tmin,
                    theta_outside_assumption_fraction=float(np.mean(theta_raw>=1)),
                    q_outside_assumption_fraction=float(np.mean(q_raw>=1)),
                    negative_discriminant_fraction=float(np.mean(disc<0)),
                    gamma_projected_fraction=float(np.mean((gamma_raw<0)|(gamma_raw>1))),
                    recovered_clipped_fraction=float(np.mean((recovered_raw<0)|(recovered_raw>1))))
    return result, dict(maps=maps, metadata=metadata)


if __name__=="__main__":
    import pyzjr
    input_dir = r'E:\PythonProject\DehazeLab\models\prior\files\input'
    output_dir = r'E:\PythonProject\DehazeLab\models\prior\files\output2'
    image_name = pyzjr.get_image_path(input_dir)
    for name in image_name:
        start = perf_counter()
        im_path = os.path.join(input_dir, name)
        save_path = os.path.join(output_dir, name)
        source = read_rgb(im_path)
        result, info = IHDCP(
            source, eta=1.8,
            guided_radius=60,
            guided_eps=1e-3,
            airlight_window=15,
            airlight_method='boundary',
            approximation='p'
        )
        info['metadata']['elapsed_seconds_with_read'] = perf_counter() - start
        info['metadata']['input'] = os.path.abspath(im_path)
        info['metadata']['shape'] = list(source.shape)
        save_rgb(save_path, result)
        pyzjr.printc(info['metadata'])

