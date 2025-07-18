import gymnasium as gym
import numpy as np
import time
from math import inf
from ros_control import BlueRovROSInterface
import os

class BlueROVEnv(gym.Env):
    def __init__(self, seed=None, save_dir=None):
        super(BlueROVEnv, self).__init__()

        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.seed_value = seed

        self.ros = BlueRovROSInterface()

        # Observation and action spaces
        self.observation_space = gym.spaces.Box(low=-inf, high=inf, shape=(10,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(6,))

        # Step counters
        self.num_episodes = 0
        self.total_steps = 0
        self.max_steps = 1000

        # Metrics initialization
        self.nb_success = 0
        self.nb_collisions = 0
        self.nb_timeouts = 0
        self.d_delta = 0
        self.norm_u = 0

        # File to log distances over episodes
        if save_dir is None:
            save_dir = "./RL"
        os.makedirs(save_dir, exist_ok=True)
        self.distance_file_path = os.path.join(save_dir, "distances_over_episodes.txt")
        self.distance_file = open(self.distance_file_path, "w")


    def step(self, action):
        """Apply forces to thrusters and return the new observation."""
        self.current_step += 1
        self.total_steps += 1

        # Add random noise to the action
        action_noise = np.random.uniform(0.01, 0.05, size=6)
        action += action_noise
        scaled_action = np.clip(action, -1.0, 1.0) * 20.0

        # Publish thruster commands
        self.ros.publish_thrusters(scaled_action)

        time.sleep(0.1)  # Wait for the thrusters to apply forces

        # Observation calculation
        self.yaw_error = self.goal_position[5] - self.ros.robot_position[5]
        self.yaw_error = 2 * np.arctan(np.tan(self.yaw_error / 2))
        self.yaw_error = np.array([self.yaw_error])
        
        self.pose_error = self.ros.robot_position[:3] - self.goal_position[:3]
        
        self.prev_action = action
        
        observation = np.concatenate([self.yaw_error, self.pose_error, self.prev_action])


        # Metrics calculation
        self.d_delta = distance_point_segment_3d_cross(self.ros.robot_initial_position[:3], self.goal_position[:3], self.ros.robot_position[:3])
        self.norm_u = np.linalg.norm(self.prev_action)
        
        # Reward calculation
        distance_to_goal = np.linalg.norm(self.ros.robot_position[:3] - self.goal_position[:3])

        if distance_to_goal >= self.prev_distance:
            reward = -10.0
        else:
            reward = np.exp(-distance_to_goal / 20)

        self.prev_distance = distance_to_goal

        terminated = distance_to_goal < 3.0
        if terminated:
            self.ros.node.get_logger().info("✅ Goal reached!")
            self.nb_success += 1
            reward = 500.0  # Bonus for reaching the goal

        elif self.ros.robot_position[2] < -60 or self.ros.robot_position[2] > -1:
            print(f"robot_position[2]: {self.ros.robot_position[2]}")
            self.ros.node.get_logger().info("⚠️ Collision detected!")
            self.collision = True
            self.nb_collisions += 1
            reward = -550.0  # Penalty for collision

        elif self.current_step >= self.max_steps:
            self.ros.node.get_logger().info("⏳ Timeout!")
            self.nb_timeouts += 1
            self.timeout = True

        truncated = self.collision or self.timeout

        self.episodes_reward += reward

        if terminated or truncated:
            # Log the distance to the goal at the end of the episode
            self.distance_file.write(f"{distance_to_goal},{self.episodes_reward}\n")
            self.distance_file.flush()

        info = {
            'nb_success': self.nb_success,
            'nb_collisions': self.nb_collisions,
            'nb_timeouts': self.nb_timeouts,
            'd_delta': self.d_delta,
            'norm_u': self.norm_u,
        }

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.num_episodes += 1
        print(f"🔄 Resetting environment for episode {self.num_episodes}...")

        self.episodes_reward = 0

        # Stop all thrusters
        self.ros.stop_all_thrusters()

        time.sleep(0.1)

        # Activate the flag to wait for the new position in pose_callback
        self.resetting_pose = True

        # Generate random yaw angle (ψ)
        yaw = self.np_random.uniform(-np.pi, np.pi)

        # Set the robot's pose
        self.ros.set_pose(yaw)

        self.ros.wait_for_reset()

        # Generate random waypoint
        self.goal_position = np.array([
            self.np_random.uniform(-20.0, 20.0),  # x
            self.np_random.uniform(-20.0, 20.0),  # y
            self.np_random.uniform(-60.0, -1.0),  # z
            0.0, 0.0, 0.0
        ])
        self.ros.node.get_logger().info(f"New goal position: {self.goal_position}")

        # Reset internal states
        self.prev_action = np.zeros(6)
        self.current_step = 0
        self.dist_initial = np.linalg.norm(self.ros.robot_position[:3] - self.goal_position[:3])
        self.prev_distance = self.dist_initial
        self.collision = False
        self.timeout = False

        self.yaw_error = self.goal_position[5] - self.ros.robot_position[5]
        self.yaw_error = 2 * np.arctan(np.tan(self.yaw_error / 2))
        self.yaw_error = np.array([self.yaw_error])
        self.pose_error = self.ros.robot_position[:3] - self.goal_position[:3]

        # Concatenate yaw error, position error, and previous action
        obs = np.concatenate([self.yaw_error, self.pose_error, self.prev_action])

        return obs, {}
    
    def close(self):
        """Stop all thrusters and shutdown the ROS node."""
        if hasattr(self, "distance_file") and self.distance_file:
            self.distance_file.close()
        self.ros.close()
        print("Environment closed.")


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