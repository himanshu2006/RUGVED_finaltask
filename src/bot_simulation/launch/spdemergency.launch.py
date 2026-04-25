import os
from ament_index_python.packages import get_package_share_directory


from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, AppendEnvironmentVariable
from launch.substitutions import Command
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # Define package path
    pkg_path = get_package_share_directory('bot_simulation')
    
    # 1. Paths (Ensure these folders exist in your 'src/bot_simulation'!)
    xacro_file = os.path.join(pkg_path, 'model', 'robot_model.xacro')
    # world_file = os.path.join(pkg_path, 'worlds', 'empty.sdf') 
    # world_file= 'empty.sdf'

    world_file = os.path.join(pkg_path, 'worlds', 'world.sdf')
    model_path = os.path.join(pkg_path, 'model')
    # Build resource path including all subdirectories (for textures, meshes, etc.)

    # 2. Convert Xacro to URDF (Uses 'Command' from substitutions)
    robot_description_config = Command(['xacro ', xacro_file])
    
    # 3. Robot State Publisher (Broadcasts the robot's structure)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_config, 'use_sim_time': True}]
    )

    # 4. START GAZEBO (Uses 'IncludeLaunchDescription' and 'PythonLaunchDescriptionSource')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py'
        )]),
        launch_arguments={'gz_args': f'-r {world_file}'}.items(),
    )

    # 5. SPAWN THE ROBOT (The physical 3D model)
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description', '-name', 'my_bot', '-z', '0.5', '-Y', '1.5708'],
        output='screen'
    )

    # 6. THE BRIDGE (Connects LiDAR/Camera to ROS)
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/scan@sensor_msgs/msg/LaserScan[ignition.msgs.LaserScan',
            '/camera/image_raw@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/cmd_vel@geometry_msgs/msg/Twist]ignition.msgs.Twist',
            '/model/my_bot/tf@tf2_msgs/msg/TFMessage[ignition.msgs.Pose_V'
        ],
        output='screen'
    )

    return LaunchDescription([
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', model_path),
        AppendEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', model_path),
        gazebo,
        node_robot_state_publisher,
        spawn_entity,
        bridge
    ])