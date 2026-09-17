#!/usr/bin/env python3
"""
Publishes the merged surround view using the orientations solved by
calibrate_merge.py.

    python3 surround_merge_node.py --ros-args -p config:=/path/to/merge_config.json

Everything expensive -- per-camera undistortion maps, the rotation-based
cylindrical maps, the seam layout and the feather weights -- is built once at
startup. Per frame it is one remap per camera, a masked copy and a narrow
blend band, so the per-frame cost stays roughly flat.

Publishes: /cameras_merged/image/compressed
"""
import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
import message_filters

from pano_geometry import (rotation_matrix, build_camera_map, compile_map,
                           warp_compiled, build_owner_map, build_composite_plan,
                           composite, compute_gains, gain_lut, apply_gain)


class SurroundMergeNode(Node):
    def __init__(self):
        super().__init__("surround_merge_node")

        self.declare_parameter("config", "merge_config.json")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("output_scale", 1.0)
        self.declare_parameter("sync_slop", 0.15)
        self.declare_parameter("gain_period", 30)   # frames between gain updates; 0 = off

        cfg_path = self.get_parameter("config").value
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.output_scale = float(self.get_parameter("output_scale").value)
        slop = float(self.get_parameter("sync_slop").value)
        self.gain_period = int(self.get_parameter("gain_period").value)

        if not os.path.exists(cfg_path):
            raise SystemExit(
                f"config not found: {cfg_path}\n"
                f"Run capture_merge_frames.py then calibrate_merge.py first.")

        with open(cfg_path) as fh:
            cfg = json.load(fh)

        if cfg.get("version", 1) < 2:
            raise SystemExit(
                f"{cfg_path} is an old (v1) config. It only stores a yaw plus "
                f"a 2D nudge, which cannot express pitch or roll. Re-run "
                f"calibrate_merge.py to produce a v2 config.")

        self.camera_order = cfg["camera_order"]
        self.pano_w = cfg["pano_width"]
        self.pano_h = cfg["pano_height"]
        pano_f = cfg["pano_focal"]
        alpha = cfg.get("undistort_alpha", 0.0)
        skip_undistort = cfg.get("frames_pre_undistorted", False)
        self.feather = int(cfg.get("blend_feather", 24))

        self.undist_maps = {}
        self.maps = {}
        for name in self.camera_order:
            c = cfg["cameras"][name]
            src_w, src_h = c["source_size"]
            f = c["focal"]
            cx, cy = c["principal_point"]

            if skip_undistort:
                self.undist_maps[name] = None
            else:
                K = np.array(c["camera_matrix"], dtype=float)
                D = np.array(c["dist_coeffs"], dtype=float)
                cal_w, cal_h = c["cal_size"]
                model = c.get("model", "standard")
                sx, sy = src_w / cal_w, src_h / cal_h
                Ks = K.copy()
                Ks[0, 0] *= sx; Ks[0, 2] *= sx
                Ks[1, 1] *= sy; Ks[1, 2] *= sy
                if model == "fisheye":
                    Dfe = D.reshape(-1, 1)[:4] if D.size >= 4 else np.zeros((4, 1))
                    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                        Ks, Dfe, (src_w, src_h), np.eye(3),
                        balance=alpha, new_size=(src_w, src_h))
                    mx, my = cv2.fisheye.initUndistortRectifyMap(
                        Ks, Dfe, np.eye(3), new_K, (src_w, src_h), cv2.CV_16SC2)
                else:
                    new_K, _ = cv2.getOptimalNewCameraMatrix(
                        Ks, D, (src_w, src_h), alpha=alpha, newImgSize=(src_w, src_h))
                    mx, my = cv2.initUndistortRectifyMap(
                        Ks, D, None, new_K, (src_w, src_h), cv2.CV_16SC2)
                self.undist_maps[name] = (mx, my)

            R = rotation_matrix(c["yaw_deg"], c["pitch_deg"], c["roll_deg"])
            mx, my = build_camera_map(
                src_w, src_h, f, cx, cy, R,
                self.pano_w, self.pano_h, self.pano_h / 2.0, pano_f)
            # Fixed point maps plus a precomputed coverage mask: the geometry
            # never changes after startup, so the float maps and the mask
            # comparison should not be paid for on every frame.
            self.maps[name] = compile_map(mx, my)

            self.get_logger().info(
                f"  {name:12s} yaw={c['yaw_deg']:+7.2f}  "
                f"pitch={c['pitch_deg']:+6.2f}  roll={c['roll_deg']:+6.2f}  "
                f"f={f:.1f}")

        self.plan = None       # seam layout + feather bands, solved once
        self.luts = {}         # per-camera gain lookup tables
        self.buf = None        # reused output buffer
        self.warp_buf = {}     # reused per-camera warp buffers
        self.frame_i = 0

        subs = [message_filters.Subscriber(
                    self, CompressedImage, f"/camera_{n}/image/compressed",
                    qos_profile=qos_profile_sensor_data)
                for n in self.camera_order]
        self.ts = message_filters.ApproximateTimeSynchronizer(
            subs, queue_size=30, slop=slop)
        self.ts.registerCallback(self.on_frames)

        self.pub = self.create_publisher(
            CompressedImage, "/cameras_merged/image/compressed", 10)

        self.get_logger().info(
            f"ready -- panorama {self.pano_w}x{self.pano_h}"
            + (f", published at {self.output_scale:.2f}x scale"
               if self.output_scale != 1.0 else ""))

    def on_frames(self, *msgs):
        try:
            warped, masks = {}, {}
            for name, msg in zip(self.camera_order, msgs):
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                                   cv2.IMREAD_COLOR)
                if img is None:
                    self.get_logger().warning(f"could not decode {name}")
                    return
                um = self.undist_maps.get(name)
                if um is not None:
                    img = cv2.remap(img, um[0], um[1], cv2.INTER_LINEAR)
                # Exposure correction goes on the source frame: it is smaller
                # than the panorama, and the seam blend then mixes pixels that
                # have already been matched.
                if name in self.luts:
                    apply_gain(img, self.luts[name])
                if name not in self.warp_buf:
                    self.warp_buf[name] = np.zeros(
                        (self.pano_h, self.pano_w, 3), np.uint8)
                warped[name], masks[name] = warp_compiled(
                    img, self.maps[name], out=self.warp_buf[name])

            # Seams depend only on the fixed geometry, so solve them once.
            if self.plan is None:
                owner = build_owner_map(masks, warped, self.camera_order,
                                        self.pano_w, self.pano_h)
                self.plan = build_composite_plan(owner, masks,
                                                 self.camera_order, self.feather)
                self.get_logger().info(
                    f"seam layout fixed, {(owner >= 0).mean() * 100:.1f}% of "
                    f"panorama covered, {len(self.plan['runs'])} row runs, "
                    f"{len(self.plan['bands'])} blend bands")

            # Gains are measured now and applied from the next frame on. One
            # frame of lag on exposure is not worth a second warp.
            if self.gain_period > 0 and self.frame_i % self.gain_period == 0:
                self.luts = {n: gain_lut(g) for n, g in
                             compute_gains(warped, masks, self.camera_order).items()}
            self.frame_i += 1

            if self.buf is None:
                self.buf = np.zeros((self.pano_h, self.pano_w, 3), np.uint8)
            pano = composite(warped, self.camera_order, self.plan, out=self.buf)

            if self.output_scale != 1.0:
                pano = cv2.resize(
                    pano, (int(self.pano_w * self.output_scale),
                           int(self.pano_h * self.output_scale)),
                    interpolation=cv2.INTER_AREA)

            ok, enc = cv2.imencode(".jpg", pano,
                                   [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                self.get_logger().error("jpeg encode failed")
                return

            out = CompressedImage()
            out.header = msgs[0].header
            if not out.header.frame_id:
                out.header.frame_id = "base_link"
            out.format = "jpeg"
            out.data = enc.tobytes()
            self.pub.publish(out)

        except Exception as e:
            self.get_logger().error(f"merge failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SurroundMergeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()