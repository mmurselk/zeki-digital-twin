from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import LogInfo
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
 
 
def generate_launch_description() -> LaunchDescription:
    # TODO: declare launch arguments (port, address, SSL options) so the
    #       websocket endpoint is configurable per environment (dev/staging/partner)
    port_arg = DeclareLaunchArgument(
        'port', default_value='9090', 
        description='The WebSocket port for the bridge.'
    )
    address_arg = DeclareLaunchArgument(
        'address', default_value='', 
        description='The WebSocket address (leave empty to bind to all interfaces).'
    )
    ssl_arg = DeclareLaunchArgument(
        'ssl', default_value='false', 
        description='Enable SSL for secure WebSocket connections (wss://).'
    )
    certfile_arg = DeclareLaunchArgument(
        'certfile', default_value='', 
        description='Path to the SSL certificate file (if ssl=true).'
    )
    keyfile_arg = DeclareLaunchArgument(
        'keyfile', default_value='', 
        description='Path to the SSL key file (if ssl=true).'
    )
 
    
    # TODO: pass through QoS / topic whitelist params if we need to limit
    #       which raw topics (camera/lidar/vehicle) get exposed over the bridge
    topics_glob_arg = DeclareLaunchArgument(
        'topics_glob', 
        default_value="''", # <-- Single quotes inside double quotes
        description='Whitelist of topics to expose (comma-separated, e.g., "/camera/*,/vehicle/odom"). Leave empty for all.'
    )
    
    # TODO: include rosbridge_server's rosbridge_websocket_launch.xml/py
    #       from the rosbridge_suite package instead of reinventing it
    rosbridge_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('rosbridge_server'),
                'launch',
                'rosbridge_websocket_launch.xml'
            ])
        ),
        launch_arguments={
            'port': LaunchConfiguration('port'),
            'address': LaunchConfiguration('address'),
            'ssl': LaunchConfiguration('ssl'),
            'certfile': LaunchConfiguration('certfile'),
            'keyfile': LaunchConfiguration('keyfile'),
            'topics_glob': LaunchConfiguration('topics_glob'),
        }.items()
    )
    
    # TODO: add a log message or event handler confirming the websocket
    #       server started successfully (useful before running test_bridge_client.py)
    log_msg = LogInfo(
        msg=[
            'Initializing rosbridge websocket server. ',
            'Binding to port: ', LaunchConfiguration('port'), 
            ' | SSL enabled: ', LaunchConfiguration('ssl')
        ]
    )
    
    
    
    # TODO: return LaunchDescription with the above actions
    return LaunchDescription([
        port_arg,
        address_arg,
        ssl_arg,
        certfile_arg,
        keyfile_arg,
        topics_glob_arg,   # must come before rosbridge_launch uses it
        log_msg,
        rosbridge_launch,
    ]  )
    
 
 
def get_default_params_path() -> str:
    # TODO: resolve path to config/params.yaml (currently only relevant once
    #       optimization nodes are unparked — leave unused for now, just stubbed)
    pass
 
