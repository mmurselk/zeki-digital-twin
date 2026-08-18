import numpy as np
import pytest

from bridge_pipeline.point_cloud_filter_node import voxel_downsample, extract_field
from sensor_msgs.msg import PointField


def test_voxel_downsample_collapses_nearby_points():
    # two points close together (same 0.1m voxel), one point far away
    positions = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],
        [5.0, 5.0, 5.0],
    ], dtype=np.float32)

    avg_positions, avg_intensities = voxel_downsample(positions, None, voxel_size=0.1)

    assert avg_positions.shape[0] == 2  # two voxels occupied, not three points
    # the far point should survive unchanged (only occupant of its voxel)
    assert np.any(np.all(np.isclose(avg_positions, [5.0, 5.0, 5.0], atol=1e-4), axis=1))


def test_voxel_downsample_averages_intensity():
    positions = np.array([
        [0.01, 0.01, 0.01],
        [0.02, 0.02, 0.02],
    ], dtype=np.float32)
    intensities = np.array([10.0, 20.0], dtype=np.float32)

    avg_positions, avg_intensities = voxel_downsample(positions, intensities, voxel_size=0.1)

    assert avg_positions.shape[0] == 1
    assert np.isclose(avg_intensities[0], 15.0)


def test_voxel_downsample_no_reduction_when_points_are_far_apart():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
    ], dtype=np.float32)

    avg_positions, _ = voxel_downsample(positions, None, voxel_size=0.1)

    assert avg_positions.shape[0] == 3  # nothing should collapse


def test_extract_field_reads_correct_offsets():
    # build a fake raw buffer: two points, point_step=8, x at offset 0, y at offset 4
    point_step = 8
    raw = np.zeros(point_step * 2, dtype=np.uint8)
    raw[0:4] = np.frombuffer(np.float32(1.5).tobytes(), dtype=np.uint8)
    raw[4:8] = np.frombuffer(np.float32(2.5).tobytes(), dtype=np.uint8)
    raw[8:12] = np.frombuffer(np.float32(3.5).tobytes(), dtype=np.uint8)
    raw[12:16] = np.frombuffer(np.float32(4.5).tobytes(), dtype=np.uint8)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    ]

    x_vals = extract_field(raw, point_step, fields, 'x')
    y_vals = extract_field(raw, point_step, fields, 'y')

    assert np.allclose(x_vals, [1.5, 3.5])
    assert np.allclose(y_vals, [2.5, 4.5])


def test_extract_field_returns_none_for_missing_field():
    fields = [PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1)]
    raw = np.zeros(4, dtype=np.uint8)

    result = extract_field(raw, 4, fields, 'intensity')

    assert result is None