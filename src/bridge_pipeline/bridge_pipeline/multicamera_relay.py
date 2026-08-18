import re
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

# Matches raw (non-relayed) compressed camera topics like
# /camera_front_wide/image/compressed - captures "front_wide" as the name.
# Deliberately anchored with $ so it does NOT match our own
# .../image/compressed_relay output topics.
DEFAULT_DISCOVERY_PATTERN = r'^/camera_(?P<name>[^/]+)/image/compressed$'


class CameraChannel:
    """All state and pub/sub wiring for a single relayed camera stream.

    One of these is created per entry in the node's `camera_names` parameter,
    each fully independent - its own subscription, publisher, throttle state,
    and adaptive target_hz driven by its own feedback topic.
    """

    def __init__(self, node: Node, name: str, discovered_input_topic: str = None):
        self.node = node
        self.name = name
        log = node.get_logger()

        # Namespaced params: cameras.<name>.<field>. Defaults fall back to the
        # /camera_<name>/... convention (or the exact topic discovery found,
        # if this channel was created by auto-discovery) so a camera "works"
        # with zero config, but every field is still overridable per-camera
        # via the usual cameras.<name>.<field> param.
        prefix = f'cameras.{name}.'
        default_input = discovered_input_topic or f'/camera_{name}/image/compressed'

        def declare(field, default):
            node.declare_parameter(prefix + field, default)
            return node.get_parameter(prefix + field).value

        self.input_topic = declare('input_topic', default_input)
        self.output_topic = declare('output_topic', f'/camera_{name}/image/compressed_relay')
        self.target_width = declare('target_width', 960)   # 0 = no resize, keep original width
        self.jpeg_quality = declare('jpeg_quality', 60)     # 0-100
        self.target_hz = declare('target_hz', 3.0)          # 0 = no throttle, pass every frame

        # each camera gets its own feedback topic by default so throttling one
        # stream doesn't accidentally throttle every other camera sharing the
        # node - override explicitly if you actually want cameras to move in
        # lockstep off a single shared feedback signal.
        self.feedback_topic = declare('feedback_topic', f'/frontend/perf_feedback/{name}')
        self.min_hz = declare('min_hz', 1.0)
        self.max_hz = declare('max_hz', 8.0)
        self.feedback_timeout_sec = declare('feedback_timeout_sec', 5.0)
        self.smoothing_alpha = declare('smoothing_alpha', 0.3)  # 0-1, higher = react faster, more jitter

        # staleness checks now only run for channels we already know are active
        # (see CameraRelayNode.check_all_feedback_staleness) - get_subscription_count()
        # is the authoritative "is anyone watching" signal, not feedback presence.
        # so silence here means "feedback topic isn't wired up for this camera yet",
        # not "nobody's watching" - default to max_hz rather than punishing a
        # confirmed-active camera down to min_hz. set to 'min_hz' to restore the
        # older behavior if you're intentionally not sending per-camera feedback.
        self.recover_toward = declare('recover_toward', 'max_hz')

        # lazy relay: don't subscribe to the raw camera topic at all until
        # something is actually subscribed to our output. This is the expensive
        # part (decode/resize/encode per frame) - a publisher with 0 subscribers
        # costs nothing, so we can safely create that eagerly, but the input
        # subscription is only stood up on demand. See update_active_state().
        self.lazy_relay = declare('lazy_relay', True)
        self.deactivation_grace_sec = declare('deactivation_grace_sec', 3.0)
        self.subscription = None
        self.active = False
        self.last_active_time = 0.0

        self.min_interval = (1.0 / self.target_hz) if self.target_hz > 0 else 0.0
        self.last_publish_time = 0.0
        self.last_feedback_time = time.time()  # start now, not 0 - avoids an immediate "stale" trigger

        self.publisher = node.create_publisher(CompressedImage, self.output_topic, 10)

        if not self.lazy_relay:
            self.activate()

        # frontend reports a suggested safe fps here - see on_feedback for the expected format
        self.feedback_subscription = node.create_subscription(
            Float32,
            self.feedback_topic,
            self.on_feedback,
            10,
        )

        log.info(
            f"[{self.name}] Ready - will relay '{self.input_topic}' -> '{self.output_topic}' "
            f"on demand (lazy_relay={self.lazy_relay}) (target_width={self.target_width}, "
            f"jpeg_quality={self.jpeg_quality}, target_hz={self.target_hz}, "
            f"feedback_topic={self.feedback_topic}, range=[{self.min_hz}, {self.max_hz}])"
        )

    def activate(self):
        """Stand up the raw camera subscription - this is what actually costs CPU/bandwidth,
        so it only happens once we know someone wants this camera's output."""
        if self.subscription is not None:
            return
        self.node.get_logger().info(f"[{self.name}] Activating relay - subscriber detected on '{self.output_topic}'")
        # camera drivers commonly publish best-effort, same lesson as the lidar node -
        # match it here or risk silently receiving nothing.
        self.subscription = self.node.create_subscription(
            CompressedImage,
            self.input_topic,
            self.on_image,
            qos_profile_sensor_data,
        )
        self.active = True

    def deactivate(self):
        """Tear down the raw camera subscription - no subscribers means no reason to keep
        pulling and processing frames nobody will see."""
        if self.subscription is None:
            return
        self.node.get_logger().info(f"[{self.name}] Deactivating relay - no subscribers on '{self.output_topic}'")
        self.node.destroy_subscription(self.subscription)
        self.subscription = None
        self.active = False

    def update_active_state(self):
        """Called on a shared poll timer. Activates on first subscriber, deactivates after
        deactivation_grace_sec of having zero subscribers (grace period avoids flapping the
        subscription on quick tab refreshes or brief rosbridge reconnects)."""
        if not self.lazy_relay:
            return
        has_subscribers = self.publisher.get_subscription_count() > 0
        if has_subscribers:
            self.last_active_time = time.time()
            if self.subscription is None:
                self.activate()
        elif self.subscription is not None:
            if (time.time() - self.last_active_time) > self.deactivation_grace_sec:
                self.deactivate()

    def set_target_hz(self, new_hz: float):
        """Apply a new target rate, clamped to [min_hz, max_hz], and recompute the throttle interval."""
        clamped = max(self.min_hz, min(self.max_hz, new_hz))
        if abs(clamped - self.target_hz) > 0.05:  # avoid log spam for tiny changes
            self.node.get_logger().info(
                f"[{self.name}] Adjusting target_hz: {self.target_hz:.2f} -> {clamped:.2f}"
            )
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
        """If no feedback has arrived recently, drift target_hz toward `recover_toward`.
        Default is min_hz: with multiple cameras relayed at once, silence most likely
        means nobody's subscribed to this one, so idle low instead of burning resources
        relaying an unwatched stream at full rate. Set recover_toward='max_hz' to restore
        the single-camera assumption that silence means 'no known constraint'."""
        if (time.time() - self.last_feedback_time) > self.feedback_timeout_sec:
            recovery_target = self.max_hz if self.recover_toward == 'max_hz' else self.min_hz
            if abs(self.target_hz - recovery_target) > 0.05:
                recovery_step = self.target_hz + 0.5 * (recovery_target - self.target_hz)
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
                self.node.get_logger().warn(f"[{self.name}] Failed to decode frame on '{self.input_topic}'")
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
                self.node.get_logger().warn(f"[{self.name}] Failed to re-encode frame on '{self.input_topic}'")
                return
            t3 = time.perf_counter()
        except Exception as exc:
            self.node.get_logger().warn(f"[{self.name}] Error processing frame on '{self.input_topic}': {exc}")
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
        self.node.get_logger().debug(
            f"[{self.name}] {input_size/1024:.1f}KB -> {output_size/1024:.1f}KB "
            f"({100 * (1 - output_size / input_size):.1f}% reduction) | "
            f"decode={1000*(t1-t0):.1f}ms resize={1000*(t2-t1):.1f}ms "
            f"encode={1000*(t3-t2):.1f}ms publish={1000*(t4-t3):.1f}ms "
            f"total={1000*(t4-t0):.1f}ms"
        )


class CameraRelayNode(Node):
    def __init__(self):
        super().__init__('camera_relay_node')

        # Auto-discovery is the default: the node periodically scans the ROS
        # graph for any topic matching discovery_pattern and starts relaying
        # it automatically, with zero config needed per camera. Subscribe to
        # whichever <topic>_relay you want from the frontend at any time -
        # nothing needs to be told which cameras exist ahead of time.
        self.declare_parameter('auto_discover', True)
        self.declare_parameter('discovery_pattern', DEFAULT_DISCOVERY_PATTERN)
        self.declare_parameter('discovery_interval_sec', 5.0)

        # optional allowlist: if non-empty, only these camera names are ever
        # relayed even if others are discovered on the bus. Leave empty to
        # relay every camera matching discovery_pattern.
        self.declare_parameter('camera_names', [])

        self.auto_discover = self.get_parameter('auto_discover').value
        self.discovery_pattern = re.compile(self.get_parameter('discovery_pattern').value)
        self.camera_allowlist = set(self.get_parameter('camera_names').value)

        self.channels = {}  # name -> CameraChannel

        if self.auto_discover:
            # run once immediately so we don't wait discovery_interval_sec
            # before relaying anything on startup, then keep polling for
            # cameras that come online later (e.g. driver nodes starting
            # after this one, or hotplugged sensors).
            self.discover_cameras()
            interval = self.get_parameter('discovery_interval_sec').value
            self.discovery_timer = self.create_timer(interval, self.discover_cameras)
        elif self.camera_allowlist:
            for name in self.camera_allowlist:
                self.channels[name] = CameraChannel(self, name)
        else:
            self.get_logger().warn(
                "auto_discover is False and camera_names is empty - node is up but "
                "relaying nothing. Either enable auto_discover or set camera_names."
            )

        # single shared timer drives staleness recovery for every channel -
        # cheaper than one timer per camera, and each channel tracks its own
        # last_feedback_time so recovery is still independent per camera.
        self.recovery_timer = self.create_timer(2.0, self.check_all_feedback_staleness)

        # shared timer that activates/deactivates each channel's raw camera
        # subscription based on whether anything is currently subscribed to
        # its output - this is what actually saves CPU/bandwidth with lazy_relay.
        self.declare_parameter('activation_poll_interval_sec', 1.0)
        poll_interval = self.get_parameter('activation_poll_interval_sec').value
        self.activation_timer = self.create_timer(poll_interval, self.update_all_active_states)

        self.get_logger().info(
            f"CameraRelayNode up (auto_discover={self.auto_discover}) with "
            f"{len(self.channels)} camera(s) so far: {list(self.channels.keys())}"
        )

    def discover_cameras(self):
        """Scan the ROS graph for new compressed-image topics matching discovery_pattern
        and start relaying any we're not already handling. Existing channels are left
        untouched - this only ever adds new cameras, never removes or restarts running ones."""
        for topic_name, topic_types in self.get_topic_names_and_types():
            if 'sensor_msgs/msg/CompressedImage' not in topic_types:
                continue
            match = self.discovery_pattern.match(topic_name)
            if not match:
                continue
            name = match.group('name')
            if name in self.channels:
                continue
            if self.camera_allowlist and name not in self.camera_allowlist:
                continue
            self.get_logger().info(f"Discovered new camera '{name}' on '{topic_name}' - starting relay")
            self.channels[name] = CameraChannel(self, name, discovered_input_topic=topic_name)

    def check_all_feedback_staleness(self):
        for channel in self.channels.values():
            if channel.active:  # no point tracking staleness for a channel that isn't relaying
                channel.check_feedback_staleness()

    def update_all_active_states(self):
        for channel in self.channels.values():
            channel.update_active_state()


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