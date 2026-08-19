# zeki-digital-twin

A containerized ROS 2 pipeline that bridges vehicle sensor topics (camera, LiDAR, GPS/pose, vehicle status) to a web-based live visualization over `rosbridge`.

## Usage

Watch the demo below to see the project in action:


<img width="800" height="450" alt="Image" src="https://github.com/user-attachments/assets/20ae38bf-6629-4074-9520-597ee0a1489e" />


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
- **Custom ROS 2 Messages:** The bridge pipeline requires custom message definitions to properly deserialize the vehicle's status data. You need to have the following repositories in your `src/` directory:  - [autoware_auto_msgs](https://github.com/tier4/autoware_auto_msgs)  - [tier4_autoware_msgs](https://github.com/tier4/tier4_autoware_msgs)

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

### 4. Build custom messages and play the rosbag

In a second terminal, open another bash session inside the container. Before playing the bag, you need to build the custom vehicle messages so that `rosbridge` can properly interpret the vehicle status data.

```bash
docker compose exec bridge_pipeline bash

# 1. Build ONLY the required message packages
colcon build --packages-up-to autoware_auto_vehicle_msgs tier4_vehicle_msgs

# 2. Source the newly built workspace
source install/setup.bash

# 3. Play the bag
ros2 bag play /bags/rosbag2_2024_03_19-15_18_39

### 5. Run the camera and point cloud relay nodes

The web viewer subscribes to the throttled/filtered topics published by the relay nodes (`*_relay`, `*_filtered`), not the raw high-rate topics. Open two more bash sessions inside the container (one per node) and run:

```bash
docker compose exec bridge_pipeline bash
source install/setup.bash
ros2 run bridge_pipeline multicamera_relay
```

```bash
docker compose exec bridge_pipeline bash
source install/setup.bash
ros2 run bridge_pipeline point_cloud_filter_node
```

> Executable names above assume they're registered as `console_scripts` entry points matching the script filenames. Check `src/bridge_pipeline/setup.py` if `ros2 run` fails with "No executable found".

Leave both running alongside the bag playback from step 4 — the web viewer won't have data on the relay/filtered topics without them.
```

### 6. Open the web viewer

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
