import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from math import inf
from ros_interface import BlueROVROSInterface

class BlueROVEnv(gym.Env):
    def __init__(self, seed=None):
        super().__init__()
        
        # seeding for reproducibility
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        self.py_random = random.Random(seed)

        # ROS interface for BlueROV
        self.ros = BlueROVROSInterface()

        # Define action and observation spaces
        self.observation_space = spaces.Box(low=-inf, high=inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        
        # Initialize variables
        self.num_episodes = 0
        self.total_steps = 0
        self.max_steps = 1000
        self.nb_success = 0
        self.nb_collisions = 0
        self.nb_timeouts = 0
        self.d_delta = 0
        self.norm_u = 0

        
        # File to log distances over episodes
        self.distance_file = open("distances_over_episodes_test_3.txt", "w")

    def step(self, action):
        self.current_episode_step += 1
        self.total_steps += 1
        
        # Apply action to thrusters
        self.ros.publish_thrusters(action)
        self.ros.wait_for_pose_update()
        
        robot_position_xyz = self.ros.robot_position[:3]
        robot_position_yaw = self.ros.robot_position[5]
        goal_position_xyz = self.goal_position[:3]
        goal_position_yaw = self.goal_position[5]

        # Error calculations
        self.yaw_error = angle_diff(robot_position_yaw, goal_position_yaw)
        self.pose_error = robot_position_xyz - goal_position_xyz

        # Metrics calculation
        robot_initial_position_xyz = self.robot_initial_position[:3]
        self.d_delta = distance_point_segment_3d_cross(robot_position_xyz, goal_position_xyz, robot_initial_position_xyz)
        self.norm_u = np.linalg.norm(action)

        # Observation and reward
        observation = np.concatenate([self.yaw_error, self.pose_error, action])
        distance_to_goal = np.linalg.norm(robot_position_xyz - goal_position_xyz)
        
        if distance_to_goal < self.prev_distance:
            reward = 40 * np.exp(-distance_to_goal / 20)
        else:
            reward = -10.0

        self.prev_distance = distance_to_goal

        terminated = False
        truncated = False

        # Gestion des fins d'épisode
        if distance_to_goal < 3.0:
            print("✅ Goal reached!")
            self.nb_success += 1
            reward = 500.0
            terminated = True
        elif robot_position_xyz[2] < -60 or robot_position_xyz[2] > -1 \
            or robot_position_xyz[0] < -30 or robot_position_xyz[0] > 30 \
            or robot_position_xyz[1] < -30 or robot_position_xyz[1] > 30:
            
            print("🚨 Collision detected!")
            self.nb_collisions += 1
            reward = -550.0
            truncated = True
        elif self.current_episode_step >= self.max_steps:
            print("⏳ Timeout reached!")
            self.nb_timeouts += 1
            truncated = True

        if terminated or truncated:
            print(f"Episode {self.num_episodes} ended: Distance to goal: {distance_to_goal:.2f}, Total Reward: {self.episodes_reward:.2f}, Total Steps: {self.current_episode_step}")
            # Écrit la distance et la récompense totale dans le fichier
            self.distance_file.write(f"{distance_to_goal},{self.episodes_reward}\n")
            self.distance_file.flush()

        self.episodes_reward += reward

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
        print(f"------- Episode {self.num_episodes} started -------")

        self.episodes_reward = 0.0

        # Stop all thrusters and reset the robot's pose
        self.ros.stop_all_thrusters()
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.robot_initial_position = np.array([0.0, 0.0, -20.0, 0.0, 0.0, yaw])
        self.ros.set_pose(self.robot_initial_position)
        
        # Generate random goal position
        self.goal_position = np.array([
            self.np_random.uniform(-20.0, 20.0),
            self.np_random.uniform(-20.0, 20.0),
            self.np_random.uniform(-60.0, -1.0),
            0.0, 0.0, 0.0
        ])
        print(f"🎯 New goal position: {self.goal_position[:3]} with yaw {self.goal_position[5]:.2f} rad")
        distance_to_goal = np.linalg.norm(self.ros.robot_position[:3] - self.goal_position[:3])
        print(f"📏 Distance initiale robot-goal : {distance_to_goal:.2f}")
        
        # Resetting variables
        self.prev_action = np.zeros(6)
        self.current_episode_step = 0
        self.prev_distance = distance_to_goal
        self.yaw_error = angle_diff(self.ros.robot_position[5], self.goal_position[5])
        self.pose_error = self.ros.robot_position[:3] - self.goal_position[:3]
        obs = np.concatenate([self.yaw_error, self.pose_error, self.prev_action])
        return obs, {}

    def close(self):
        self.ros.close()


def angle_diff(a, b):
    diff = np.arctan2(np.sin(a - b), np.cos(a - b))
    return np.atleast_1d(diff)

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


