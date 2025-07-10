import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from control import BlueROVEnv  # Votre environnement Gym ROS2
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm



class ProgressBarCallback(BaseCallback):
    """
    Callback pour mettre à jour la barre de progression tqdm.
    """
    def __init__(self, total_timesteps, verbose=0):
        super(ProgressBarCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.progress_bar = tqdm(total=total_timesteps, desc="Entraînement en cours")

    def _on_step(self) -> bool:
        self.progress_bar.n = self.num_timesteps
        self.progress_bar.refresh()
        return True

    def _on_training_end(self) -> None:
        self.progress_bar.close()

def train_model():
    print("🚀 Entraînement PPO du BlueROV en cours...")

    train_env = make_vec_env(lambda: BlueROVEnv(seed=42), n_envs=1)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="cpu",  # Pour éviter l'avertissement GPU inutile    
    )

    total_timesteps = 1_000_000
    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)

    model.learn(total_timesteps=total_timesteps, callback=progress_callback)

    # Sauvegarde du modèle
    model.save("./logs/final_model_bluerov_ppo")
    train_env.save("logs/vecnormalize_ppo.pkl")
    print("✅ Modèle entraîné et sauvegardé.")
    train_env.close()

def test_model():
    print("🧪 Test du modèle PPO sur BlueROV...")

    test_env = make_vec_env(lambda: BlueROVEnv(seed=42), n_envs=1)
    test_env = VecNormalize.load("logs/vecnormalize_ppo.pkl", venv=test_env)
    model = PPO.load("./logs/final_model_bluerov_ppo.zip", device="cpu")
 
    num_episodes = 500
    distances_over_steps = []
    steps_test = 0
    nb_steps_episode = []
    sum_norm_u_list = []

    list_d_delta = []
    list_norm_u = []

    obs = test_env.reset()
    #print("observation shape:", obs.shape, obs)

    for episode in range(num_episodes):
        print(f"\n🎯 Épisode {episode + 1} / {num_episodes}")
        done = False
        truncated = False
        total_reward = 0
        step = 0
        sum_norm_u = 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = test_env.step(action)

            total_reward += reward[0]
            step += 1
            steps_test += 1

            list_d_delta.append(info[0]['d_delta'])
            list_norm_u.append(info[0]['norm_u'])
            sum_norm_u += info[0]['norm_u']
            

        nb_steps_episode.append(step)
        sum_norm_u_list.append(sum_norm_u)

        print(f"✅ Fin de l’épisode {episode + 1} - Total Reward: {total_reward:.2f}, Steps: {step}")

    test_env.close()
    # Print metrics
    nb_success = info[0]['nb_success']
    nb_collisions = info[0]['nb_collisions']
    nb_timeouts = info[0]['nb_timeouts']

    print(f"success rate (%): {nb_success / num_episodes * 100:.2f}%")
    print(f"collision rate (%): {nb_collisions / num_episodes * 100:.2f}%")
    print(f"timeout rate (%): {nb_timeouts / num_episodes * 100:.2f}%")
    print(f"mean of d_delta: {np.mean(list_d_delta):.2f}")
    print(f"std of d_delta: {np.std(list_d_delta):.2f}")
    print(f"mean of norm_u: {np.mean(list_norm_u):.2f}")
    print(f"mean number of step: {np.mean(nb_steps_episode):.2f}")
    print(f"mean of norm_u: {np.mean(sum_norm_u_list):.2f}")

    #plot_distance(steps_test, distances_over_steps)

def plot_distance(steps, distance):
    steps = np.arange(1, steps + 1)
    distance = np.array(distance)

    plt.plot(steps, distance, label='Distance to Goal')
    plt.axhline(y=3, color='r', linestyle='--', label='Threshold (3)')

    plt.xlabel('Steps')
    plt.ylabel('Distance to Goal')
    plt.title('Evolution de la distance à l\'objectif en fonction des étapes')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model()
    test_model()
