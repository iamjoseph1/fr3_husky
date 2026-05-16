#!/usr/bin/env python3

import math
from datetime import datetime
from pathlib import Path
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def _default_output_dir() -> Path:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        repo_scripts_dir = parent / "src" / "fr3_husky" / "fr3_husky_controller" / "scripts"
        if repo_scripts_dir.is_dir():
            return repo_scripts_dir / "pub_image"
    return script_path.parent / "pub_image"


def _depth_to_bgr(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth)
    positive = finite & (depth > 0.0)
    if not np.any(positive):
        normalized = np.zeros(depth.shape, dtype=np.uint8)
    else:
        valid = depth[positive]
        near = np.percentile(valid, 2.0)
        far = np.percentile(valid, 98.0)
        if math.isclose(float(near), float(far)):
            far = near + 1.0
        clipped = np.clip(depth, near, far)
        normalized = ((clipped - near) * 255.0 / (far - near)).astype(np.uint8)
        normalized[~positive] = 0
    return cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)


def _image_to_cv(msg: Image) -> np.ndarray:
    if msg.encoding in ("rgb8", "bgr8"):
        channels = 3
        dtype = np.uint8
    elif msg.encoding in ("mono8", "8UC1"):
        channels = 1
        dtype = np.uint8
    elif msg.encoding in ("32FC1",):
        channels = 1
        dtype = np.float32
    elif msg.encoding in ("16UC1",):
        channels = 1
        dtype = np.uint16
    else:
        raise ValueError(f"Unsupported image encoding: {msg.encoding}")

    height = int(msg.height)
    width = int(msg.width)
    expected_step = width * channels * np.dtype(dtype).itemsize
    row_step = int(msg.step)

    data = np.frombuffer(msg.data, dtype=dtype)
    if row_step == expected_step:
        if channels == 1:
            image = data.reshape((height, width))
        else:
            image = data.reshape((height, width, channels))
    else:
        row_items = row_step // np.dtype(dtype).itemsize
        if channels == 1:
            image = data.reshape((height, row_items))[:, :width]
        else:
            image = data.reshape((height, row_items // channels, channels))[:, :width, :]

    if msg.encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if msg.encoding == "32FC1":
        return _depth_to_bgr(image)
    if msg.encoding == "16UC1":
        return _depth_to_bgr(image.astype(np.float32) * 0.001)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


class SAFrontOverviewImageSaver(Node):
    def __init__(self) -> None:
        super().__init__("sa_front_overview_image_saver")

        default_output_dir = _default_output_dir()

        self.declare_parameter("image_topic", "/sa_front_overview/image_raw")
        self.declare_parameter("output_dir", str(default_output_dir))
        self.declare_parameter("save_period_sec", 1.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.save_period_sec = float(self.get_parameter("save_period_sec").value)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._latest_image = None
        self._latest_stamp_ns = None

        self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_timer(self.save_period_sec, self._save_latest_image)

        self.get_logger().info(f"Subscribing camera topic: {self.image_topic}")
        self.get_logger().info(f"Saving images to: {self.output_dir}")
        self.get_logger().info(f"Save period: {self.save_period_sec:.2f} sec")

    def _image_callback(self, msg: Image) -> None:
        try:
            image = _image_to_cv(msg)
        except ValueError as exc:
            self.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return

        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        with self._lock:
            self._latest_image = image
            self._latest_stamp_ns = stamp_ns

    def _save_latest_image(self) -> None:
        with self._lock:
            if self._latest_image is None:
                return
            image = self._latest_image.copy()
            stamp_ns = self._latest_stamp_ns

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if stamp_ns is None:
            filename = self.output_dir / f"sa_front_overview_{timestamp}.png"
        else:
            filename = self.output_dir / f"sa_front_overview_{timestamp}_{stamp_ns}.png"

        if cv2.imwrite(str(filename), image):
            self.get_logger().info(
                f"Saved image: {filename.name}"
            )
        else:
            self.get_logger().error(f"Failed to save image: {filename}")


def main() -> None:
    rclpy.init()
    node = SAFrontOverviewImageSaver()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
