import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Vector3
from transforms3d.euler import quat2euler, euler2quat
import time
from test_pid import BlueROVPIDController
from ros_gz_interfaces.srv import SetEntityPose
import signal
import sys
from math import inf
import random

class BlueROVEnv(gym.Env):
    def __init__(self, use_goal_file=False, seed=42):
        super(BlueROVEnv, self).__init__()
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.py_random = random.Random(seed)

        # Initialize ROS 2 node
        rclpy.init()
        self.node = rclpy.create_node('bluerov_rl_agent')

        # Observation and action spaces
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))

        # Thruster publishers
        self.thruster_topics = [
            '/bluerov2/cmd_thruster1',
            '/bluerov2/cmd_thruster2',
            '/bluerov2/cmd_thruster3',
            '/bluerov2/cmd_thruster4',
            '/bluerov2/cmd_thruster5',
            '/bluerov2/cmd_thruster6'
        ]
        self.thruster_publishers = {topic: self.node.create_publisher(Float64, topic, 10) for topic in self.thruster_topics}

        # Ocean current publisher
        self.ocean_current_publisher = self.node.create_publisher(Vector3, '/current', 10)

        # Subscription to the robot's ground truth position
        self.subscription = self.node.create_subscription(Pose, '/bluerov2/pose_gt', self.pose_callback, 1)

        # Service client for setting entity pose
        self.client = self.node.create_client(SetEntityPose, '/world/ocean/set_pose')

        # Step counters and flags
        self.num_episodes = 0
        self.total_steps = 0
        self.max_steps = 800

        # Metrics initialization
        self.nb_success = 0
        self.nb_collisions = 0
        self.nb_timeouts = 0
        self.d_delta = 0
        self.norm_u = 0

        self.goal_position_idx = 0
        self.use_goal_file = use_goal_file

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
        sensor_noise = self.np_random.uniform(0.05, 0.1, size=6)
        #self.robot_position += sensor_noise

        self.pose_updated = True


    def step(self, action):
        """Apply forces to thrusters and return the new observation."""

        self.current_step += 1
        self.total_steps += 1

        self.pose_updated = False

        # Add random noise to the action
        action_noise = self.np_random.uniform(0.01, 0.05, size=6)
        #action += action_noise

        scaled_action = np.clip(action, -1.0, 1.0) * 20.0
        for i, topic in enumerate(self.thruster_topics):
            msg = Float64()
            msg.data = float(scaled_action[i])
            self.thruster_publishers[topic].publish(msg)

        rclpy.spin_once(self.node)


        # Observation calculation
        self.yaw_error = self.goal_position[5] - self.robot_position[5]
        self.yaw_error = 2 * np.arctan(np.tan(self.yaw_error / 2))
        self.yaw_error = np.array([self.yaw_error])
        self.pose_error = self.robot_position[:3] - self.goal_position[:3]
        self.prev_action = scaled_action
        observation = np.concatenate([self.yaw_error, self.pose_error, self.prev_action])


        # Metrics calculation
        self.d_delta = distance_point_segment_3d_cross(self.robot_initial_position[:3], self.goal_position[:3], self.robot_position[:3])
        self.norm_u = np.linalg.norm(self.prev_action)
        
        # Reward calculation
        distance_to_goal = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])

        if distance_to_goal >= self.prev_distance:
            reward = -10.0
        else:
            reward = 40 * np.exp(-distance_to_goal / 20)

        self.prev_distance = distance_to_goal

        terminated = distance_to_goal < 3.0

        if terminated:
            self.node.get_logger().info("✅ Goal reached!")
            self.goal_reached = True
            self.nb_success += 1
            reward = 500.0  # Bonus for reaching the goal

        if self.robot_position[2] < -60 or self.robot_position[2] > -1:
            print(f"robot_position[2]: {self.robot_position[2]}")
            self.node.get_logger().info("⚠️ Collision detected!")
            self.collision = True
            self.nb_collisions += 1
            reward = -550.0  # Penalty for collision

        if self.current_step >= self.max_steps:
            self.node.get_logger().info("⏳ Timeout!")
            self.nb_timeouts += 1
            self.timeout = True

        truncated = self.collision or self.timeout

        time.sleep(0.1)  # Simulate a delay for thruster response
    

        info = {
            'nb_success': self.nb_success,
            'nb_collisions': self.nb_collisions,
            'nb_timeouts': self.nb_timeouts,
            'd_delta': self.d_delta,
            'norm_u': self.norm_u,
        }

        return observation, reward, terminated, truncated, info

    def save_goal_positions(self):
        """Save the goal positions to a text file."""
        with open('goal_positions.txt', 'a') as file:
            file.write(f"{self.goal_position}\n")

    def load_goal_positions(self, index, filename='goal_positions.txt'):
        """Charge la position de but à l'index donné depuis le fichier texte."""
        try:
            with open(filename, 'r') as file:
                lines = [line for line in file if line.strip()]
                if not lines:
                    self.node.get_logger().warn(f"⚠️ File {filename} is empty.")
                    return None
                idx = index % len(lines)  # Pour boucler si index > nb de lignes
                line = lines[idx].strip().replace('[', '').replace(']', '')
                values = [float(x) for x in line.split() if x.strip()]
                while len(values) < 6:
                    values.append(0.0)
                return np.array(values[:6])
        except FileNotFoundError:
            self.node.get_logger().warn(f"⚠️ File {filename} not found. No goal positions loaded.")
            return None

    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement et place le robot à la position de départ."""
        self.num_episodes += 1
        print(f"🔄 Resetting environment for episode {self.num_episodes}...")
        super().reset(seed=seed)

        # 1. Arrêt des propulseurs
        self.stop_all_thrusters()

        time.sleep(0.1)

        # Activate the flag to wait for the new position in pose_callback
        self.resetting_pose = True

        # 3. Génération d'un yaw aléatoire
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.node.get_logger().info(f"🔄 Random yaw for reset: {yaw:.2f} radians")

        # 4. Téléportation du robot à la position initiale
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

        # Active waiting loop until pose_callback detects the reset
        timeout = 2.0  # seconds
        start_time = time.time()
        rclpy.spin_once(self.node, timeout_sec=0.1)
        distance_to_initial = np.linalg.norm(self.robot_position[:3] - self.robot_initial_position[:3])
        while distance_to_initial > 0.2 and (time.time() - start_time) < timeout:
            rclpy.spin_once(self.node, timeout_sec=0.1)
            distance_to_initial = np.linalg.norm(self.robot_position[:3] - self.robot_initial_position[:3])
        print("time for  resetting", time.time() - start_time)

        distance_to_goal = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])
        print(f"📏 Distance initiale robot-goal : {distance_to_goal:.2f}")



        # 7. Réinitialisation des états internes
        self.prev_action = np.zeros(6)
        self.current_step = 0
        self.dist_initial = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])
        self.prev_distance = self.dist_initial
        self.collision = False
        self.timeout = False

        # 8. Calcul des erreurs initiales
        self.yaw_error = self.goal_position[5] - self.robot_position[5]
        self.yaw_error = 2 * np.arctan(np.tan(self.yaw_error / 2))
        self.yaw_error = np.array([self.yaw_error])
        self.pose_error = self.robot_position[:3] - self.goal_position[:3]

        # Concatenate yaw error, position error, and previous action
        obs = np.concatenate([self.yaw_error, self.pose_error, self.prev_action])

        return obs, {}

    def stop_all_thrusters(self):
        """Set all thrusters to zero."""
        for topic in self.thruster_topics:
            msg = Float64()
            msg.data = 0.0
            self.thruster_publishers[topic].publish(msg)

    def modify_ocean_currents(self):
        """Modify the ocean currents."""
        ocean_current_velocity = np.random.uniform(0, 1)
        ocean_current_horizontal_angle = np.random.uniform(-0.5, 0.5)
        ocean_current_vertical_angle = np.random.uniform(-0.5, 0.5)

        # Publish new ocean current values
        current_msg = Vector3()
        current_msg.x = ocean_current_velocity * np.cos(ocean_current_horizontal_angle)
        current_msg.y = ocean_current_velocity * np.sin(ocean_current_horizontal_angle)
        current_msg.z = ocean_current_velocity * np.sin(ocean_current_vertical_angle)
        self.ocean_current_publisher.publish(current_msg)

    def close(self):
        self.stop_all_thrusters()
        rclpy.shutdown()
        self.node.destroy_node()
        self.node.get_logger().info("ROS 2 node destroyed.")

def distance_point_segment_3d_cross(A, B, C):
    A = np.asarray(A)
    B = np.asarray(B)
    C = np.asarray(C)

    AB = B - A
    AC = C - A
    AB_len2 = np.dot(AB, AB)

    if AB_len2 == 0:
        return np.linalg.norm(AC)

    t = np.dot(AC, AB)
    if t <= 0:
        return np.linalg.norm(AC)

    BC = C - B
    if np.dot(BC, AB) >= 0:
        return np.linalg.norm(BC)

    return np.linalg.norm(np.cross(AB, AC)) / np.sqrt(AB_len2)
