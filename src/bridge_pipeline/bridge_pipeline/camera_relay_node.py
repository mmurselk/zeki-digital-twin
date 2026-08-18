import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage


class CameraRelayNode(Node):
    def __init__(self):
        super().__init__('camera_relay_node')

        self.declare_parameter('input_topic', '/camera_front_wide/image/compressed')
        self.declare_parameter('output_topic', '/camera_front_wide/image/compressed_relay')
        self.declare_parameter('target_width', 960)   # 0 = no resize, keep original width
        self.declare_parameter('jpeg_quality', 60)    # 0-100
        self.declare_parameter('target_hz', 3.0)       # 0 = no throttle, pass every frame

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_width = self.get_parameter('target_width').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.target_hz = self.get_parameter('target_hz').value

        self.min_interval = (1.0 / self.target_hz) if self.target_hz > 0 else 0.0
        self.last_publish_time = 0.0

        # camera drivers commonly publish best-effort, same lesson as the lidar node -
        # match it here or risk silently receiving nothing.
        self.subscription = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.publisher = self.create_publisher(CompressedImage, self.output_topic, 10)

        self.get_logger().info(
            f"Subscribed to '{self.input_topic}', publishing relayed frames to "
            f"'{self.output_topic}' (target_width={self.target_width}, "
            f"jpeg_quality={self.jpeg_quality}, target_hz={self.target_hz})"
        )

    def on_image(self, msg: CompressedImage):
        now = time.time()

        # throttle first, before spending any CPU on decode/resize/encode -
        # cheapest possible way to skip frames we don't want anyway
        if self.min_interval > 0 and (now - self.last_publish_time) < self.min_interval:
            return
        self.last_publish_time = now

        t0 = time.perf_counter()
        input_size = len(msg.data)

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            self.get_logger().warn(f"Failed to decode frame on '{self.input_topic}'")
            return
        t1 = time.perf_counter()

        if self.target_width and image.shape[1] > self.target_width:
            scale = self.target_width / image.shape[1]
            new_size = (self.target_width, int(image.shape[0] * scale))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        t2 = time.perf_counter()

        success, encoded = cv2.imencode(
            '.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not success:
            self.get_logger().warn(f"Failed to re-encode frame on '{self.input_topic}'")
            return
        t3 = time.perf_counter()

        out_msg = CompressedImage()
        out_msg.header = msg.header
        out_msg.format = 'jpeg'
        out_msg.data = encoded.tobytes()
        self.publisher.publish(out_msg)
        t4 = time.perf_counter()

        output_size = len(out_msg.data)
        self.get_logger().info(
            f"{input_size/1024:.1f}KB -> {output_size/1024:.1f}KB "
            f"({100 * (1 - output_size / input_size):.1f}% reduction) | "
            f"decode={1000*(t1-t0):.1f}ms resize={1000*(t2-t1):.1f}ms "
            f"encode={1000*(t3-t2):.1f}ms publish={1000*(t4-t3):.1f}ms "
            f"total={1000*(t4-t0):.1f}ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = CameraRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()