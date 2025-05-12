from simple_launch import SimpleLauncher, GazeboBridge

from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

from launch import LaunchDescription


def generate_launch_description():
    
    sl = SimpleLauncher()
    sl.declare_arg('gui', default_value=False)
    sl.declare_arg('spawn', default_value=True)


    # Chargement du monde Gazebo
    with sl.group(if_arg='gui'):
        sl.gz_launch(sl.find('bluerov2_description', 'demo_world.sdf'), "-r")
        
    with sl.group(unless_arg='gui'):
        sl.gz_launch(sl.find('bluerov2_description', 'demo_world.sdf'), "-r -s")

    # Configuration des bridges (topics uniquement)
    bridges = [
        GazeboBridge.clock(),

        GazeboBridge('/ocean_current', '/current', 'geometry_msgs/msg/Vector3',
                     GazeboBridge.ros2gz),

        GazeboBridge('/model/bluerov2/pose', '/bluerov2/current_pose',
                     'geometry_msgs/msg/Pose', GazeboBridge.gz2ros),
    ]
    
    sl.create_gz_bridge(bridges)

    # Ton Node ROS 2 classique pour le bridge service
    service_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='bridge_set_pose_service',
        arguments=[
            '/world/ocean/set_pose@ros_gz_interfaces/srv/SetEntityPose'
        ],
        output='screen'
    )

    # Spawn du robot
    with sl.group(if_arg='spawn'):
        sl.include('bluerov2_description', 
                   'upload_bluerov2_launch.py', 
                   launch_arguments={'sliders': 'false'}.items())

    # Rassembler tout dans une seule LaunchDescription
    return LaunchDescription(
        sl.launch_description().entities + [service_bridge]
    )
