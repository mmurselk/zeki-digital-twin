#!/usr/bin/env python3
"""
Shared geometry for the 6-camera surround merge.

The model in one paragraph:
  Every camera has a full 3-DOF orientation (yaw, pitch, roll) and its own
  intrinsics. A panorama pixel is turned into a world ray, rotated into the
  camera's frame by R^T, and projected through that camera's pinhole model.
  So the calibration parameters ARE the physical mounting angles -- there is
  no second "correction" stage that only approximates pitch and roll.

  This replaces the earlier yaw-only map plus a 2D shift/rotate fixup. That
  fixup could not express roll (a rotation about the optical axis is a
  bearing-dependent shear on a cylinder, not a rigid rotation of the slice)
  or pitch (a pitched camera's horizon is a sine curve on the cylinder, not
  a vertical offset). Both errors are smallest at the slice centre and worst
  at the seams, which is exactly where they show.

Panorama convention (unchanged):
    column u -> bearing theta = (u - W/2) * 2*pi / W    (+ = toward left)
    row    v -> height  h     = (v - pano_cy) / pano_f
    world ray d = (sin theta, h, cos theta)
    axes: x right, y DOWN, z forward.

Angle convention:
    yaw   about y (down):    + turns the camera toward +theta
    pitch about x (right):   + is nose up
    roll  about z (forward): + rolls the image clockwise
    R = Ry(yaw) @ Rx(pitch) @ Rz(roll),  world_ray = R @ camera_ray
"""
import numpy as np
import cv2


# --------------------------------------------------------------------------
# orientation
# --------------------------------------------------------------------------

def rotation_matrix(yaw_deg, pitch_deg, roll_deg):
    y, p, r = np.deg2rad([float(yaw_deg), float(pitch_deg), float(roll_deg)])
    Ry = np.array([[np.cos(y), 0.0, np.sin(y)],
                   [0.0, 1.0, 0.0],
                   [-np.sin(y), 0.0, np.cos(y)]])
    Rx = np.array([[1.0, 0.0, 0.0],
                   [0.0, np.cos(p), -np.sin(p)],
                   [0.0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0.0],
                   [np.sin(r), np.cos(r), 0.0],
                   [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def angles_from_rotation(R):
    """Inverse of rotation_matrix, for reporting. Returns degrees."""
    pitch = np.arcsin(np.clip(-R[1, 2], -1.0, 1.0))
    yaw = np.arctan2(R[0, 2], R[2, 2])
    roll = np.arctan2(R[1, 0], R[1, 1])
    return tuple(np.degrees([yaw, pitch, roll]))


def pixels_to_rays(pts, f, cx, cy):
    """Undistorted pixel coords -> unit rays in the camera frame."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    r = np.stack([(pts[:, 0] - cx) / f,
                  (pts[:, 1] - cy) / f,
                  np.ones(len(pts))], axis=1)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


# --------------------------------------------------------------------------
# panorama maps
# --------------------------------------------------------------------------

def build_camera_map(src_w, src_h, f, cx, cy, R,
                     pano_w, pano_h, pano_cy, pano_f, fov_limit_deg=85.0):
    """
    Remap tables painting one undistorted frame onto its slice of the
    cylinder. Returns (map_x, map_y) of shape (pano_h, pano_w); pixels this
    camera cannot see are set to -1 so cv2.remap leaves them blank.

    f, cx, cy   intrinsics of the UNDISTORTED frame (may differ per camera)
    R           camera->world rotation from rotation_matrix()
    pano_f      panorama scale; keep pano_f == pano_w / (2*pi) so the
                horizontal and vertical angular scales agree
    """
    u = np.arange(pano_w, dtype=np.float64)[None, :]
    v = np.arange(pano_h, dtype=np.float64)[:, None]

    theta = (u - pano_w / 2.0) * (2.0 * np.pi / pano_w)
    height = (v - pano_cy) / float(pano_f)

    dx = np.broadcast_to(np.sin(theta), (pano_h, pano_w))
    dy = np.broadcast_to(height, (pano_h, pano_w))
    dz = np.broadcast_to(np.cos(theta), (pano_h, pano_w))

    Rt = np.asarray(R, dtype=np.float64).T
    cxr = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz
    cyr = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz
    czr = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz

    norm = np.sqrt(cxr * cxr + cyr * cyr + czr * czr)
    ahead = (czr / norm) > np.cos(np.deg2rad(fov_limit_deg))
    safe_z = np.where(ahead, czr, 1.0)

    x_src = f * cxr / safe_z + cx
    y_src = f * cyr / safe_z + cy

    inside = (ahead
              & (x_src >= 0) & (x_src <= src_w - 1)
              & (y_src >= 0) & (y_src <= src_h - 1))
    x_src = np.where(inside, x_src, -1.0)
    y_src = np.where(inside, y_src, -1.0)
    return x_src.astype(np.float32), y_src.astype(np.float32)


def warp_to_pano(img, map_x, map_y):
    """Remap a source frame onto the panorama. Returns (bgr, coverage_mask)."""
    warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = (map_x >= 0) & (map_y >= 0)
    warped[~mask] = 0
    return warped, mask


def compile_map(map_x, map_y):
    """
    Pack a float map pair into the fixed-point form cv2.remap prefers, and
    precompute the coverage mask.

    The geometry is static once calibration is done, so the runtime node
    should pay for this once at startup instead of re-deriving the mask from
    the float maps on every frame. Returns a dict for warp_compiled().
    """
    m1, m2 = cv2.convertMaps(map_x, map_y, cv2.CV_16SC2)
    mask = (map_x >= 0) & (map_y >= 0)
    return {"m1": m1, "m2": m2, "mask": mask,
            "outside": np.repeat((~mask)[:, :, None], 3, axis=2)}


def warp_compiled(img, cm, out=None):
    """warp_to_pano() using a compile_map() result. Returns (bgr, mask)."""
    warped = cv2.remap(img, cm["m1"], cm["m2"], cv2.INTER_LINEAR,
                       dst=out, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=(0, 0, 0))
    warped[cm["outside"]] = 0
    return warped, cm["mask"]


# --------------------------------------------------------------------------
# seams
# --------------------------------------------------------------------------

def find_vertical_seam(cost):
    """Lowest-cost top-to-bottom path, one column step per row."""
    h, w = cost.shape
    if h == 0 or w == 0:
        return np.zeros(max(h, 0), dtype=np.int32)
    acc = cost.astype(np.float64).copy()
    back = np.zeros((h, w), dtype=np.int32)
    for y in range(1, h):
        prev = acc[y - 1]
        left = np.roll(prev, 1); left[0] = np.inf
        right = np.roll(prev, -1); right[-1] = np.inf
        stack = np.stack([left, prev, right], axis=0)
        best = np.argmin(stack, axis=0)
        acc[y] += stack[best, np.arange(w)]
        back[y] = np.arange(w) + (best - 1)
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = int(np.argmin(acc[-1]))
    for y in range(h - 2, -1, -1):
        seam[y] = int(np.clip(back[y + 1, seam[y + 1]], 0, w - 1))
    return seam


def build_owner_map(masks, warped, camera_order, pano_w, pano_h):
    """
    Decide which camera owns each panorama pixel. In each pairwise overlap the
    boundary follows the lowest-difference seam, so cuts fall where the two
    images already agree rather than through an object.
    """
    owner = np.full((pano_h, pano_w), -1, dtype=np.int16)
    n = len(camera_order)

    for i in range(n):
        j = (i + 1) % n
        a, b = camera_order[i], camera_order[j]
        if a not in masks or b not in masks:
            continue
        overlap = masks[a] & masks[b]
        if not overlap.any():
            continue

        xs = np.where(overlap.any(axis=0))[0]
        # Overlaps can wrap past column 0; split into contiguous runs.
        groups, start = [], 0
        for sp in np.where(np.diff(xs) > 1)[0]:
            groups.append(xs[start:sp + 1])
            start = sp + 1
        groups.append(xs[start:])

        for grp in groups:
            if len(grp) < 2:
                continue
            x0, x1 = int(grp[0]), int(grp[-1]) + 1
            band_a = warped[a][:, x0:x1].astype(np.float32)
            band_b = warped[b][:, x0:x1].astype(np.float32)
            diff = np.abs(band_a - band_b).sum(axis=2)
            ov = overlap[:, x0:x1]
            diff[~ov] = 0.0
            seam = find_vertical_seam(diff)

            cols = np.arange(x0, x1)[None, :]
            is_left = cols < (x0 + seam)[:, None]
            sub = owner[:, x0:x1]
            free = (sub == -1) & ov
            sub[free & is_left] = i
            sub[free & ~is_left] = j

    # Anything covered by exactly one camera goes to that camera.
    for idx, name in enumerate(camera_order):
        if name not in masks:
            continue
        owner[masks[name] & (owner == -1)] = idx
    return owner


def _dist_to_set(S, pad):
    """Distance to the nearest True pixel of S, wrapping horizontally."""
    src = np.hstack([S[:, -pad:], S, S[:, :pad]])
    d = cv2.distanceTransform((~src).astype(np.uint8), cv2.DIST_L2, 3)
    return d[:, pad:pad + S.shape[1]]


def _seam_bands(owner, masks, camera_order, feather):
    """
    Feather band around each ownership boundary.

    Returns a list of (flat_index, i, j, alpha) where alpha is the weight of
    camera i. Only pixels both cameras actually see are included, so the hard
    outer edge of the panorama is never softened against nothing.
    """
    if feather <= 0:
        return []
    n = len(camera_order)
    pad = int(feather) + 4
    bands = []
    for i in range(n):
        j = (i + 1) % n
        a, b = camera_order[i], camera_order[j]
        if a not in masks or b not in masks:
            continue
        region_i, region_j = (owner == i), (owner == j)
        if not region_i.any() or not region_j.any():
            continue
        s = _dist_to_set(region_j, pad) - _dist_to_set(region_i, pad)
        band = (masks[a] & masks[b] & (np.abs(s) < feather)
                & (region_i | region_j))
        if not band.any():
            continue
        alpha = np.clip(0.5 + 0.5 * s / float(feather), 0.0, 1.0)
        idx = np.flatnonzero(band.ravel())
        bands.append((idx, i, j, alpha.ravel()[idx].astype(np.float32)))
    return bands


def build_composite_plan(owner, masks, camera_order, feather=24):
    """
    Turn the fixed seam layout into something cheap to execute per frame.

    Ownership is run-length encoded per row, so compositing becomes a few
    thousand contiguous row-slice copies instead of six full-panorama boolean
    masks. At 6016x700 that is ~3 ms per frame instead of ~280 ms.
    """
    h, w = owner.shape
    runs = []
    for y in range(h):
        row = owner[y]
        cuts = np.flatnonzero(np.diff(row)) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [w]))
        for s, e in zip(starts, ends):
            i = int(row[s])
            if i >= 0:
                runs.append((y, int(s), int(e), i))
    return {"runs": runs,
            "bands": _seam_bands(owner, masks, camera_order, feather),
            "shape": (h, w)}


def composite(warped, camera_order, plan, out=None):
    """Assemble the panorama. Pass `out` to reuse the buffer across frames."""
    h, w = plan["shape"]
    if out is None:
        out = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        out[:] = 0

    imgs = [warped.get(n) for n in camera_order]
    for y, s, e, i in plan["runs"]:
        src = imgs[i]
        if src is not None:
            out[y, s:e] = src[y, s:e]

    flat = out.reshape(-1, 3)
    for idx, i, j, alpha in plan["bands"]:
        A, B = imgs[i], imgs[j]
        if A is None or B is None:
            continue
        a = alpha[:, None]
        blended = (a * A.reshape(-1, 3)[idx].astype(np.float32)
                   + (1.0 - a) * B.reshape(-1, 3)[idx].astype(np.float32))
        flat[idx] = np.clip(blended + 0.5, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------
# exposure
# --------------------------------------------------------------------------

def compute_gains(warped, masks, camera_order, max_samples=4000,
                  sigma_n=10.0, sigma_g=0.1):
    """
    Per-camera, per-channel gain that makes overlapping regions agree
    (Brown & Lowe style). Removes the brightness steps at the seams.
    """
    n = len(camera_order)
    A = np.zeros((3, n, n))
    b = np.zeros((3, n))
    for i in range(n):
        j = (i + 1) % n
        na, nb = camera_order[i], camera_order[j]
        if na not in masks or nb not in masks:
            continue
        ov = masks[na] & masks[nb]
        N = int(ov.sum())
        if N < 200:
            continue
        idx = np.flatnonzero(ov.ravel())
        if len(idx) > max_samples:
            idx = idx[np.linspace(0, len(idx) - 1, max_samples).astype(int)]
        Ia = warped[na].reshape(-1, 3)[idx].astype(np.float64).mean(axis=0)
        Ib = warped[nb].reshape(-1, 3)[idx].astype(np.float64).mean(axis=0)
        for c in range(3):
            A[c, i, i] += N * Ia[c] ** 2 / sigma_n ** 2 + N / sigma_g ** 2
            A[c, j, j] += N * Ib[c] ** 2 / sigma_n ** 2 + N / sigma_g ** 2
            A[c, i, j] -= N * Ia[c] * Ib[c] / sigma_n ** 2
            A[c, j, i] -= N * Ia[c] * Ib[c] / sigma_n ** 2
            b[c, i] += N / sigma_g ** 2
            b[c, j] += N / sigma_g ** 2

    G = np.ones((n, 3))
    for c in range(3):
        try:
            G[:, c] = np.linalg.solve(A[c] + 1e-9 * np.eye(n), b[c])
        except np.linalg.LinAlgError:
            pass
    mean = G.mean()
    if mean > 1e-6:
        G = G / mean
    G = np.clip(G, 0.5, 2.0)
    return {name: G[i] for i, name in enumerate(camera_order)}


def gain_lut(g):
    """(256, 3) uint8 lookup table for one camera's per-channel gain."""
    return np.clip(np.arange(256)[:, None] * np.asarray(g, dtype=float)[None, :],
                   0, 255).astype(np.uint8)


def apply_gain(img, lut):
    """
    In-place gain correction on one BGR image. Cheapest applied to the source
    frame before warping -- it is smaller than the panorama, and the seam
    blend then mixes already-corrected pixels.
    """
    for c in range(3):
        img[:, :, c] = cv2.LUT(img[:, :, c], np.ascontiguousarray(lut[:, c]))
    return img


def apply_gains(images, gains):
    for name, g in gains.items():
        if name in images:
            apply_gain(images[name], gain_lut(g))
    return images
