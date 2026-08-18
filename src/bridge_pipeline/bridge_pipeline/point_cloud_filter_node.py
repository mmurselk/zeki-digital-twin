import time
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField


def extract_field(raw_bytes: np.ndarray, point_step: int, fields, name: str) -> np.ndarray:
    """Pull one named field out of a raw PointCloud2 byte buffer as a 1D float array."""
    field = next((f for f in fields if f.name == name), None)
    if field is None:
        return None

    if field.datatype == PointField.FLOAT32:
        dtype, itemsize = np.float32, 4
    elif field.datatype == PointField.FLOAT64:
        dtype, itemsize = np.float64, 8
    else:
        raise ValueError(f"Unsupported datatype {field.datatype} for field '{name}'")

    rows = raw_bytes.reshape(-1, point_step)
    return rows[:, field.offset:field.offset + itemsize].copy().view(dtype).reshape(-1)


def build_cloud_msg(header, avg_positions: np.ndarray, avg_intensities) -> PointCloud2:
    """Build a PointCloud2 message in one vectorized pass, avoiding create_cloud's
    per-point struct.pack loop which is the other major bottleneck at this point count."""
    num_points = avg_positions.shape[0]
    has_intensity = avg_intensities is not None

    if has_intensity:
        point_step = 16
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('intensity', '<f4')])
        structured = np.zeros(num_points, dtype=dtype)
        structured['x'] = avg_positions[:, 0]
        structured['y'] = avg_positions[:, 1]
        structured['z'] = avg_positions[:, 2]
        structured['intensity'] = avg_intensities
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
    else:
        point_step = 12
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        structured = np.zeros(num_points, dtype=dtype)
        structured['x'] = avg_positions[:, 0]
        structured['y'] = avg_positions[:, 1]
        structured['z'] = avg_positions[:, 2]
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = num_points
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * num_points
    msg.is_dense = True
    msg.data = structured.tobytes()
    return msg


def voxel_downsample(positions: np.ndarray, intensities, voxel_size: float):
    """Reduce a point cloud to one averaged point per occupied voxel cell."""
    voxel_ids = np.floor(positions / voxel_size).astype(np.int64)

    # collapse (i, j, k) voxel coords into a single key for grouping
    keys = (voxel_ids[:, 0].astype(np.int64) * 73856093
            ^ voxel_ids[:, 1].astype(np.int64) * 19349663
            ^ voxel_ids[:, 2].astype(np.int64) * 83492791)

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    num_voxels = counts.shape[0]

    # np.bincount is a genuinely vectorized C reduction (unlike np.add.at, which
    # falls back to a slow unbuffered loop) - run it once per column instead
    summed_positions = np.column_stack([
        np.bincount(inverse, weights=positions[:, 0], minlength=num_voxels),
        np.bincount(inverse, weights=positions[:, 1], minlength=num_voxels),
        np.bincount(inverse, weights=positions[:, 2], minlength=num_voxels),
    ])
    avg_positions = (summed_positions / counts[:, None]).astype(np.float32)

    avg_intensities = None
    if intensities is not None:
        summed_intensity = np.bincount(inverse, weights=intensities, minlength=num_voxels)
        avg_intensities = (summed_intensity / counts).astype(np.float32)

    return avg_positions, avg_intensities


class PointCloudFilterNode(Node):
    def __init__(self):
        super().__init__('point_cloud_filter_node')

        self.declare_parameter('input_topic', '/lidars/points_fused_throttle')
        self.declare_parameter('output_topic', '/lidars/points_fused_filtered')
        self.declare_parameter('voxel_size', 0.1)  # meters

        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.voxel_size = self.get_parameter('voxel_size').value

        # lidar publishers commonly use best-effort reliability (sensor data QoS) -
        # matching it here avoids the silent zero-messages bug we hit with the
        # default reliable subscription in earlier testing.
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.on_cloud,
            qos_profile_sensor_data,
        )

        self.publisher = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10,
        )

        self.get_logger().info(
            f"Subscribed to '{self.input_topic}', publishing filtered cloud to "
            f"'{self.output_topic}' (voxel_size={self.voxel_size}m)"
        )

    def on_cloud(self, msg: PointCloud2):
        t0 = time.perf_counter()

        raw = np.frombuffer(msg.data, dtype=np.uint8)
        positions_flat = np.column_stack([
            extract_field(raw, msg.point_step, msg.fields, 'x'),
            extract_field(raw, msg.point_step, msg.fields, 'y'),
            extract_field(raw, msg.point_step, msg.fields, 'z'),
        ])
        intensities = extract_field(raw, msg.point_step, msg.fields, 'intensity')
        t1 = time.perf_counter()

        input_count = positions_flat.shape[0]
        if input_count == 0:
            return

        avg_positions, avg_intensities = voxel_downsample(
            positions_flat, intensities, self.voxel_size
        )
        t2 = time.perf_counter()

        out_msg = build_cloud_msg(msg.header, avg_positions, avg_intensities)
        t3 = time.perf_counter()

        self.publisher.publish(out_msg)
        t4 = time.perf_counter()

        output_count = avg_positions.shape[0]
        self.get_logger().info(
            f"{input_count} -> {output_count} pts "
            f"({100 * (1 - output_count / input_count):.1f}% reduction) | "
            f"extract={1000*(t1-t0):.1f}ms voxel={1000*(t2-t1):.1f}ms "
            f"build={1000*(t3-t2):.1f}ms publish={1000*(t4-t3):.1f}ms "
            f"total={1000*(t4-t0):.1f}ms"
        )


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudFilterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()