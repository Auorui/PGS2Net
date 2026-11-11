import cv2
import numpy as np
from scipy.ndimage import minimum_filter

class CAP:
    """
    Color Attenuation Prior Dehazing
    """
    def __init__(
            self,
            beta=1.0,
            radius=60,
            epsilon=1e-3,
            depth_radius=15,
            top_percent=0.001,
            t_min=0.05,
            seed=0,
    ):
        self.radius = radius
        self.epsilon = epsilon
        self.depth_radius = depth_radius
        self.beta = beta
        self.top_percent = top_percent
        self.t_min = t_min
        self.seed = seed

    @staticmethod
    def _to_float(img):
        if img.dtype == np.float32:
            return img
        return img.astype(np.float32) / 255.

    def _guided_filter(self, I, p):

        r = 2 * self.radius + 1
        eps = self.epsilon

        I = self._to_float(I)
        p = p.astype(np.float32)

        Ir, Ig, Ib = I[:, :, 0], I[:, :, 1], I[:, :, 2]

        Ir_mean = cv2.blur(Ir, (r, r))
        Ig_mean = cv2.blur(Ig, (r, r))
        Ib_mean = cv2.blur(Ib, (r, r))

        Irr_var = cv2.blur(Ir * Ir, (r, r)) - Ir_mean * Ir_mean + eps
        Irg_var = cv2.blur(Ir * Ig, (r, r)) - Ir_mean * Ig_mean
        Irb_var = cv2.blur(Ir * Ib, (r, r)) - Ir_mean * Ib_mean
        Igg_var = cv2.blur(Ig * Ig, (r, r)) - Ig_mean * Ig_mean + eps
        Igb_var = cv2.blur(Ig * Ib, (r, r)) - Ig_mean * Ib_mean
        Ibb_var = cv2.blur(Ib * Ib, (r, r)) - Ib_mean * Ib_mean + eps

        Irr_inv = Igg_var * Ibb_var - Igb_var * Igb_var
        Irg_inv = Igb_var * Irb_var - Irg_var * Ibb_var
        Irb_inv = Irg_var * Igb_var - Igg_var * Irb_var
        Igg_inv = Irr_var * Ibb_var - Irb_var * Irb_var
        Igb_inv = Irb_var * Irg_var - Irr_var * Igb_var
        Ibb_inv = Irr_var * Igg_var - Irg_var * Irg_var

        cov = Irr_inv * Irr_var + Irg_inv * Irg_var + Irb_inv * Irb_var

        Irr_inv /= cov
        Irg_inv /= cov
        Irb_inv /= cov
        Igg_inv /= cov
        Igb_inv /= cov
        Ibb_inv /= cov

        p_mean = cv2.blur(p, (r, r))

        Ipr_mean = cv2.blur(Ir * p, (r, r))
        Ipg_mean = cv2.blur(Ig * p, (r, r))
        Ipb_mean = cv2.blur(Ib * p, (r, r))

        Ipr_cov = Ipr_mean - Ir_mean * p_mean
        Ipg_cov = Ipg_mean - Ig_mean * p_mean
        Ipb_cov = Ipb_mean - Ib_mean * p_mean

        ar = Irr_inv * Ipr_cov + Irg_inv * Ipg_cov + Irb_inv * Ipb_cov
        ag = Irg_inv * Ipr_cov + Igg_inv * Ipg_cov + Igb_inv * Ipb_cov
        ab = Irb_inv * Ipr_cov + Igb_inv * Ipg_cov + Ibb_inv * Ipb_cov

        b = p_mean - ar * Ir_mean - ag * Ig_mean - ab * Ib_mean

        ar = cv2.blur(ar, (r, r))
        ag = cv2.blur(ag, (r, r))
        ab = cv2.blur(ab, (r, r))
        b = cv2.blur(b, (r, r))

        return ar * Ir + ag * Ig + ab * Ib + b

    def _cal_depth_map(self, img):

        np.random.seed(self.seed)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        s = hsv[:, :, 1].astype(np.float32) / 255.
        v = hsv[:, :, 2].astype(np.float32) / 255.

        sigma = 0.041337
        noise = np.random.normal(
            0,
            sigma,
            (img.shape[0], img.shape[1])
        )

        depth = (
                0.121779 +
                0.959710 * v -
                0.780245 * s +
                noise
        )

        depth_refine = minimum_filter(
            depth,
            (self.depth_radius, self.depth_radius)
        )

        return depth_refine, depth

    def _estimate_airlight(self, img, depth):
        img = self._to_float(img)
        h, w = depth.shape
        n = int(np.ceil(self.top_percent * h * w))
        idx = np.argsort(depth.reshape(-1))
        pixels = img.reshape(-1, 3)
        candidates = pixels[idx[-n:]]
        mag = np.linalg.norm(candidates, axis=1)
        A = candidates[np.argmax(mag)]
        return A

    def recover(self, img):
        depth_refine, depth_pixel = self._cal_depth_map(img)
        depth_refine = self._guided_filter(img, depth_refine)
        transmission = np.exp(-self.beta * depth_refine)
        transmission = np.clip(transmission, self.t_min, 1)

        A = self._estimate_airlight(img, depth_refine)
        I = self._to_float(img)
        J = (I - A) / transmission[..., None] + A
        return np.clip(J, 0, 1)

    def __call__(self, img):
        return self.recover(img)

if __name__=="__main__":
    cap = CAP(
        radius=60,
        epsilon=1e-3,
        depth_radius=15,
        beta=1.0
    )

    I = cv2.imread("input.png")

    J = cap(I)

    cv2.imwrite(
        "output.png",
        (J * 255).astype(np.uint8)
    )