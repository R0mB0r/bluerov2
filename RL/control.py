import gymnasium as gym
import numpy as np
from gymnasium import spaces
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose
from transforms3d.euler import quat2euler
import time
from test_pid import BlueROVPIDController
from ros_gz_interfaces.srv import SetEntityPose

import signal  
import sys     

class BlueROVEnv(gym.Env):
    def __init__(self):
        super(BlueROVEnv, self).__init__()

        rclpy.init()
        self.node = rclpy.create_node('bluerov_rl_agent')

        # L'espace d'observation = [Position du robot (6D) + Position relative au but (6D)+ action précédentes]
        self.observation_space = spaces.Box(low=-50, high=50, shape=(18,), dtype=np.float32)

        # L'espace d'action = 6 thrusters (forces entre -30 et 30)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,))
        
        # Création des publishers pour les thrusters
        self.thruster_topics = [
            '/bluerov2/cmd_thruster1',
            '/bluerov2/cmd_thruster2',
            '/bluerov2/cmd_thruster3',
            '/bluerov2/cmd_thruster4',
            '/bluerov2/cmd_thruster5',
            '/bluerov2/cmd_thruster6'
        ]
        self.thruster_publishers = {topic: self.node.create_publisher(Float64, topic, 10) for topic in self.thruster_topics}

        # Subscription à la position du robot (ground truth)
        self.subscription = self.node.create_subscription(Pose, '/bluerov2/pose_gt', self.pose_callback, 100)
        self.robot_position = np.zeros(6)  # [x, y, z, roll, pitch, yaw]

        # Position du but
        self.goal_position = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])  

        self.dist_initial = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])

        # Compteur de steps pour le troncage
        self.current_step = 0  
        self.max_steps = 800  # Nombre maximum d'étapes par épisode

        # Distance précédente pour le calcul de la récompense
        self.prev_distance = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])

        self.prev_action = np.zeros(6)

        # Création du client de service SetEntityPose
        self.client = self.node.create_client(SetEntityPose, '/world/ocean/set_pose')

        self.resetting_pose = False

                
        def signal_handler(sig, frame):
            self.node.get_logger().info("👋 Interruption détectée. Arrêt des propulseurs...")
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

        if self.resetting_pose and np.linalg.norm(self.robot_position[:3]) < 0.05:
            self.resetting_pose = False


    def step(self, action):
        """Applique les forces aux thrusters et retourne la nouvelle observation."""
        self.current_step += 1

        scaled_action = np.clip(action, -1.0, 1.0) * 20.0 

        for i, topic in enumerate(self.thruster_topics):
            msg = Float64()
            msg.data = float(scaled_action[i])  
            self.thruster_publishers[topic].publish(msg)

        self.prev_action = scaled_action
        rclpy.spin_once(self.node)

        relative_position = self.goal_position - self.robot_position
        observation = np.concatenate([self.robot_position, relative_position, self.prev_action])

        distance_to_goal = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])
        reward = -distance_to_goal/self.dist_initial  # Normalisation de la distance

        if distance_to_goal < self.prev_distance:
            reward += 1.0

        self.prev_distance = distance_to_goal

        terminated = distance_to_goal < 0.5 
        if terminated:
            self.node.get_logger().info("✅ Objectif atteint !")
            reward += 50.0  # Bonus pour avoir atteint l'objectif 
        truncated = self.current_step >= self.max_steps 

        time.sleep(0.1)

        self.node.get_logger().info(f"Étape {self.current_step}/{self.max_steps} - Position: {self.robot_position[:3]} - Récompense: {reward:.2f} - Action: {scaled_action}")

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Stoppe tous les propulseurs
        for topic in self.thruster_topics:
            msg = Float64()
            msg.data = 0.0
            self.thruster_publishers[topic].publish(msg)

        time.sleep(0.5)

        # Active le flag pour attendre la nouvelle position dans pose_callback
        self.resetting_pose = True

        # Appel au service SetEntityPose
        self.node.get_logger().info("🔄 Appel au service SetEntityPose pour réinitialiser la position...")
        req = SetEntityPose.Request()
        req.entity.name = 'bluerov2'
        req.entity.type = 2  # Type MODEL
        req.pose.position.x = 0.0
        req.pose.position.y = 0.0
        req.pose.position.z = 0.0
        req.pose.orientation.x = 0.0
        req.pose.orientation.y = 0.0
        req.pose.orientation.z = 0.0
        req.pose.orientation.w = 1.0

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future)

        if future.result() is not None and future.result().success:
            self.node.get_logger().info("Position téléportée avec succès.")
        else:
            self.node.get_logger().warn("⚠️ Échec de la réinitialisation de la position.")

        # Boucle d'attente active jusqu'à ce que pose_callback détecte le reset
        
        timeout = 5.0  # secondes
        start_time = time.time()
        while self.resetting_pose and (time.time() - start_time < timeout):
            rclpy.spin_once(self.node, timeout_sec=0.1)

        

        # Reset des états internes
        self.prev_action = np.zeros(6)
        self.current_step = 0
        self.dist_initial = np.linalg.norm(self.robot_position[:3] - self.goal_position[:3])
        self.prev_distance = self.dist_initial

        obs = np.concatenate([self.robot_position, self.goal_position - self.robot_position, self.prev_action])
        self.node.get_logger().info("observation : " + str(obs))
        return obs, {}

    def stop_all_thrusters(self):
        """Met tous les propulseurs à zéro."""
        for topic in self.thruster_topics:
            msg = Float64()
            msg.data = 0.0
            self.thruster_publishers[topic].publish(msg)
        self.node.get_logger().info("🛑 Tous les propulseurs ont été arrêtés.")


    
    def close(self):
        self.stop_all_thrusters()
        rclpy.shutdown()
