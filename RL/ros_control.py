import rclpy
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Vector3
from ros_gz_interfaces.srv import SetEntityPose
import signal
import sys
from transforms3d.euler import quat2euler, euler2quat
import numpy as np
import time


class BlueRovROSInterface:
    def __init__(self):
        
        # Initialize ROS 2 node
        rclpy.init()
        self.node = rclpy.create_node('bluerov_rl_agent')

        # Initialize thruster publishers
        self.thruster_topics = [
            '/bluerov2/cmd_thruster1',
            '/bluerov2/cmd_thruster2',
            '/bluerov2/cmd_thruster3',
            '/bluerov2/cmd_thruster4',
            '/bluerov2/cmd_thruster5',
            '/bluerov2/cmd_thruster6'
        ]
        self.thruster_publishers = {topic: self.node.create_publisher(Float64, topic, 10) for topic in self.thruster_topics}

        # Initialize ocean_current publisher
        self.ocean_current_publisher = self.node.create_publisher(Vector3, '/current', 10)
        
        # Initialize pose_gt subscriber
        self.subscription = self.node.create_subscription(Pose, '/bluerov2/pose_gt', self.pose_callback, 1)

        # Initialize service client for setting entity pose
        self.client = self.node.create_client(SetEntityPose, '/world/ocean/set_pose')

        # Signal handler for graceful shutdown
        def signal_handler(sig, frame):
            self.node.get_logger().info("👋 Interruption detected. Stopping thrusters...")
            self.stop_all_thrusters()
            rclpy.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

    def pose_callback(self, msg):
        phi, theta, psi = quat2euler([msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z])
        self.robot_position = np.array([
            msg.position.x, msg.position.y, msg.position.z,
            phi, theta, psi
        ])

        # Add random noise to the robot's position
        sensor_noise = np.random.uniform(0.05, 0.1, size=6)
        self.robot_position += sensor_noise

    def publish_thrusters(self, action):
        for i, topic in enumerate(self.thruster_topics):
            msg = Float64()
            msg.data = float(action[i])
            self.thruster_publishers[topic].publish(msg)

        rclpy.spin_once(self.node)

    def set_pose(self, yaw):
        # Call SetEntityPose service
        req = SetEntityPose.Request()
        req.entity.name = 'bluerov2'
        req.entity.type = 2  # Type MODEL
        req.pose.position.x = 0.0
        req.pose.position.y = 0.0
        req.pose.position.z = -20.0
        quat = euler2quat(0.0, 0.0, yaw)
        req.pose.orientation.x = quat[1]
        req.pose.orientation.y = quat[2]
        req.pose.orientation.z = quat[3]
        req.pose.orientation.w = quat[0]

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        self.robot_initial_position = np.array([
            req.pose.position.x,
            req.pose.position.y,
            req.pose.position.z,
            0.0, 0.0, yaw
        ])


    def wait_for_reset(self):
        # Active waiting loop until pose_callback detects the reset
        timeout = 2.0  # seconds
        start_time = time.time()
        rclpy.spin_once(self.node, timeout_sec=0.1)
        distance_to_initial = np.linalg.norm(self.robot_position[:3] - self.robot_initial_position[:3])
        while distance_to_initial > 0.2 and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            distance_to_initial = np.linalg.norm(self.robot_position[:3] - self.robot_initial_position[:3])
        print("time for  resetting", time.time() - start_time)


    def stop_all_thrusters(self):
        """Set all thrusters to zero."""
        for topic in self.thruster_topics:
            msg = Float64()
            msg.data = 0.0
            self.thruster_publishers[topic].publish(msg)

    def set_ocean_currents(self, cv, cha, cva):
        """Set ocean currents to a random value."""
        msg = Vector3()
        msg.x = cv*np.cos(cva)*np.cos(cha)
        msg.y = cv*np.cos(cva)*np.sin(cha)
        msg.z = cv*np.sin(cva)
        self.ocean_current_publisher.publish(msg)

    
    def close(self):
        self.stop_all_thrusters()
        rclpy.shutdown()
        self.node.destroy_node()
        self.node.get_logger().info("ROS 2 node destroyed.")
