ros2_ws/
├── src/
│   └── bridge_pipeline/                    # main ROS 2 package
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       │
│       ├── bridge_pipeline/                # python module (nodes live here)
│       │   ├── __init__.py
│       │   ├── camera_relay_node.py        # optimization node - parked for now
│       │   └── pointcloud_filter_node.py   # optimization node - parked for now
│       │
│       ├── launch/
│       │   └── bridge.launch.py            # launches rosbridge_websocket
│       │
│       ├── config/
│       │   └── params.yaml                 # config for optimization nodes
│       │
│       ├── scripts/
│       │   └── test_bridge_client.py       # roslibpy test client (standalone, not a ROS node)
│       │
│       └── test/
│           ├── test_camera_relay_node.py
│           └── test_pointcloud_filter_node.py
│
├── bags/
│   └── recorded_bag/                       # 69GB rosbag2 (raw + metadata.yaml)
│       ├── metadata.yaml
│       └── recorded_bag_0.db3
│
├── docs/
│   └── architecture.md                     # notes / this diagram's source
│
└── README.md