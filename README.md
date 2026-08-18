# zeki-digital-twin

A containerized ROS 2 pipeline that bridges vehicle sensor topics (camera, LiDAR, GPS/pose, vehicle status) to a web-based live visualization over `rosbridge`.

## Structure

```
├── docker/                  # container entrypoint
├── docs/                    # architecture notes
├── src/bridge_pipeline/     # ROS 2 package: rosbridge launch, camera/pointcloud relay nodes
├── web/                     # browser-based live viewer (Mapbox + deck.gl + roslib.js)
├── Dockerfile
└── docker-compose.yml
```

## Requirements

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with WSL2 backend on Windows)
- A rosbag recording to play back (not included in this repo — see below)
- A free [Mapbox access token](https://account.mapbox.com/access-tokens/) for the web viewer

## Setup

### 1. Get a rosbag

The rosbag used for development (`rosbag2_2024_03_19-15_18_39`) is ~69GB and is **not tracked in git** (see `.gitignore`). Get it from a teammate and place it at the repo root so the path matches what `docker-compose.yml` expects:

```
zeki-digital-twin/rosbag2_2024_03_19-15_18_39/
├── metadata.yaml
└── rosbag2_2024_03_19-15_18_39_0.db3
```

### 2. Build and enter the container

```bash
docker compose run --rm --service-ports bridge_pipeline bash
```

`--service-ports` is required so the container's port 9090 (rosbridge) is actually published to the host — without it, `docker compose run` skips the port mapping defined in `docker-compose.yml`.

### 3. Start rosbridge (inside the container)

```bash
ros2 launch rosbridge_server rosbridge_websocket_launch.xml port:=9090
```

You should see `Rosbridge WebSocket server started on port 9090`. Leave this running.

### 4. Play the rosbag (in a second terminal, inside the container)

```bash
docker compose exec bridge_pipeline bash
ros2 bag play /bags/rosbag2_2024_03_19-15_18_39
```

### 5. Open the web viewer

In a third terminal, on the host machine:

```bash
cd web
python -m http.server 8000
```

Open `http://localhost:8000/mapbox_live_stream.html` in a browser.

Paste your own Mapbox token into the `mapboxgl.accessToken` line near the top of the script (left empty in this repo — do not commit a real token, GitHub's push protection will reject it).

In the **Connection** panel, connect to `localhost:9090`. Then subscribe to position, point cloud, and camera topics from the dropdowns.

## Notes

- `.gitattributes` forces LF line endings on `*.sh` files. Without this, Windows checkouts turn `docker/entrypoint.sh` into CRLF, which breaks it inside the Linux container (`exec /entrypoint.sh: no such file or directory`).
- The camera/point cloud relay nodes in `src/bridge_pipeline/bridge_pipeline/` (`multicamera_relay.py`, `point_cloud_filter_node.py`) publish throttled/filtered topics (`*_relay`, `*_filtered`) used by the web viewer instead of the raw high-rate topics.
