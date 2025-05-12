import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from control import BlueROVEnv  # Ton environnement Gym ROS2
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Wrapper pour l’environnement (nécessaire pour stable-baselines3)
def make_env():
    return BlueROVEnv()

def train_model():
    print("🚀 Entraînement PPO du BlueROV en cours...")

    # Environnement d'entraînement monitoré
    train_env = DummyVecEnv([lambda: Monitor(BlueROVEnv())])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="cpu",  # Pour éviter l'avertissement GPU inutile
        ent_coef=0.001
    )

    # Démarrage de l'apprentissage
    model.learn(total_timesteps=200_000)  
    # Sauvegarde du modèle
    model.save("./logs/final_model_bluerov_ppo")
    print("✅ Modèle entraîné et sauvegardé.")

def test_model():
    print("🧪 Test du modèle PPO sur BlueROV...")

    env = DummyVecEnv([make_env])
    model = PPO.load("ppo_bluerov")

    num_episodes = 10

    for episode in range(num_episodes):
        print(f"\n🎯 Épisode {episode + 1} / {num_episodes}")
        obs = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, _ = env.step(action)

            total_reward += reward[0]
            step += 1
            print(f"🔹 Step {step}: Reward={reward[0]:.2f}, Done={done}, Truncated={truncated}")

            time.sleep(0.1)

        print(f"✅ Fin de l’épisode {episode + 1} - Total Reward: {total_reward:.2f}, Steps: {step}")
        time.sleep(1)

    env.close()

if __name__ == "__main__":
    train_model()
    #test_model()
