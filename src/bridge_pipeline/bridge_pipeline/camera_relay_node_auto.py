import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class CameraRelayNode(Node):
    def __init__(self):
        super().__init__('camera_relay_node')

        self.declare_parameter('input_topic', '/camera_front_wide/image/compressed')
        self.declare_parameter('output_topic', '/camera_front_wide/image/compressed_relay')
        self.declare_parameter('target_width', 960)   # 0 = no resize, keep original width
        self.declare_parameter('jpeg_quality', 60)    # 0-100
        self.declare_parameter('target_hz', 3.0)       # 0 = no throttle, pass every frame

        # adaptive feedback settings - target_hz above becomes the *starting point*,
        # actual rate adjusts based on what the frontend reports it can handle
        self.declare_parameter('feedback_topic', '/frontend/perf_feedback')
        self.declare_parameter('min_hz', 1.0)
        self.declare_parameter('max_hz', 8.0)
        self.declare_parameter('feedback_timeout_sec', 5.0)
        self.declare_parameter('smoothing_alpha', 0.3)  # 0-1, higher = react faster, more jitter

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_width = self.get_parameter('target_width').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.target_hz = self.get_parameter('target_hz').value

        self.feedback_topic = self.get_parameter('feedback_topic').value
        self.min_hz = self.get_parameter('min_hz').value
        self.max_hz = self.get_parameter('max_hz').value
        self.feedback_timeout_sec = self.get_parameter('feedback_timeout_sec').value
        self.smoothing_alpha = self.get_parameter('smoothing_alpha').value

        self.min_interval = (1.0 / self.target_hz) if self.target_hz > 0 else 0.0
        self.last_publish_time = 0.0
        self.last_feedback_time = time.time()  # start now, not 0 - avoids an immediate "stale" trigger

        # camera drivers commonly publish best-effort, same lesson as the lidar node -
        # match it here or risk silently receiving nothing.
        self.subscription = self.create_subscription(
            CompressedImage,
            self.input_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.publisher = self.create_publisher(CompressedImage, self.output_topic, 10)

        # frontend reports a suggested safe fps here - see on_feedback for the expected format
        self.feedback_subscription = self.create_subscription(
            Float32,
            self.feedback_topic,
            self.on_feedback,
            10,
        )

        # if feedback stops arriving (tab closed, connection dropped), recover toward
        # max_hz gradually rather than getting stuck at whatever the last value was
        self.recovery_timer = self.create_timer(2.0, self.check_feedback_staleness)

        self.get_logger().info(
            f"Subscribed to '{self.input_topic}', publishing relayed frames to "
            f"'{self.output_topic}' (target_width={self.target_width}, "
            f"jpeg_quality={self.jpeg_quality}, target_hz={self.target_hz}, "
            f"feedback_topic={self.feedback_topic}, range=[{self.min_hz}, {self.max_hz}])"
        )

    def set_target_hz(self, new_hz: float):
        """Apply a new target rate, clamped to [min_hz, max_hz], and recompute the throttle interval."""
        clamped = max(self.min_hz, min(self.max_hz, new_hz))
        if abs(clamped - self.target_hz) > 0.05:  # avoid log spam for tiny changes
            self.get_logger().info(f"Adjusting target_hz: {self.target_hz:.2f} -> {clamped:.2f}")
        self.target_hz = clamped
        self.min_interval = (1.0 / self.target_hz) if self.target_hz > 0 else 0.0

    def on_feedback(self, msg: Float32):
        """Frontend reports the fps it estimates it can currently sustain (e.g. 1000 / avg_decode_ms).
        We smooth toward it rather than jumping instantly, to avoid oscillating on noisy measurements."""
        self.last_feedback_time = time.time()
        reported_hz = msg.data
        smoothed = (self.smoothing_alpha * reported_hz) + ((1 - self.smoothing_alpha) * self.target_hz)
        self.set_target_hz(smoothed)

    def check_feedback_staleness(self):
        """If no feedback has arrived recently, nudge the rate back up toward max_hz -
        assumes silence means 'no known constraint', not 'stay throttled forever'."""
        if (time.time() - self.last_feedback_time) > self.feedback_timeout_sec:
            if self.target_hz < self.max_hz:
                recovery_step = self.target_hz + 0.5 * (self.max_hz - self.target_hz)
                self.set_target_hz(recovery_step)

    def on_image(self, msg: CompressedImage):
        now = time.time()

        # throttle first, before spending any CPU on decode/resize/encode -
        # cheapest possible way to skip frames we don't want anyway
        if self.min_interval > 0 and (now - self.last_publish_time) < self.min_interval:
            return
        self.last_publish_time = now

        t0 = time.perf_counter()
        input_size = len(msg.data)

        # a single corrupt/truncated frame (dropped bytes, mid-stream disconnect)
        # used to throw inside imdecode/imencode and take the whole node down
        # under rclpy.spin() - now we log and skip the frame instead.
        try:
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
        except Exception as exc:
            self.get_logger().warn(f"Error processing frame on '{self.input_topic}': {exc}")
            return

        out_msg = CompressedImage()
        out_msg.header = msg.header
        out_msg.format = 'jpeg'
        out_msg.data = encoded.tobytes()
        self.publisher.publish(out_msg)
        t4 = time.perf_counter()

        output_size = len(out_msg.data)
        # debug, not info - this runs every published frame and the string
        # formatting/logging overhead itself was adding jitter at higher rates
        self.get_logger().debug(
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