import time
import numpy as np
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from bluerov_env import BlueROVEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm import tqdm
import os
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Train or test SAC on BlueROVEnv")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode: train or test")
    parser.add_argument('--train-seed', type=int, default=42, help="Seed for training")
    parser.add_argument('--test-seed', type=int, default=123, help="Seed for testing")
    parser.add_argument('--timesteps', type=int, default=1_000_000, help="Total timesteps for training")
    parser.add_argument('--episodes', type=int, default=500, help="Number of episodes for testing")
    return parser.parse_args()

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

def get_savedir(mode):
    base = "SAC_savedir_"
    dirs = [d for d in os.listdir(".") if re.match(rf"{base}\d+$", d)]
    pairs = sorted([int(d.split("_")[-1]) for d in dirs if int(d.split("_")[-1]) % 2 == 0])
    if mode == "train":
        next_pair = pairs[-1] + 2 if pairs else 0
        savedir = f"{base}{next_pair}"
        os.makedirs(savedir, exist_ok=True)
        return savedir
    elif mode == "test":
        if not pairs:
            raise FileNotFoundError("Aucun dossier de sauvegarde pair trouvé pour le test.")
        savedir = f"{base}{pairs[-1]}"
        return savedir

def train_model(train_seed, total_timesteps):
    savedir = get_savedir("train")
    print(f"🚀 Entraînement SAC du BlueROV, dossier de sauvegarde : {savedir}")

    train_env = DummyVecEnv([lambda: Monitor(BlueROVEnv(seed=train_seed, save_dir=savedir))])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    model = SAC(
        policy="MlpPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log="./logs/tensorboard/",
        device="cpu",  # Pour éviter l'avertissement GPU inutile    
    )

    progress_callback = ProgressBarCallback(total_timesteps=total_timesteps)

    model.learn(total_timesteps=total_timesteps, callback=progress_callback)

    # Sauvegarde du modèle
    model.save(os.path.join(savedir, "final_model_bluerov_sac"))
    #train_env.save(os.path.join(savedir, "vecnormalize_SAC.pkl"))
    print("✅ Modèle entraîné et sauvegardé.")
    train_env.close()

def test_model(test_seed, num_episodes):
    savedir = get_savedir("test")
    print(f"🧪 Test du modèle SAC sur BlueROV, dossier utilisé : {savedir}")

    test_env = DummyVecEnv([lambda: Monitor(BlueROVEnv(seed=test_seed, save_dir=savedir, mode="test"))])
    #test_env = VecNormalize.load(os.path.join(savedir, "vecnormalize_SAC.pkl"), venv=test_env)
    model = SAC.load(os.path.join(savedir, "final_model_bluerov_sac.zip"), device="cpu")
    print(model.policy)

    steps_test = 0
    nb_steps_episode = []
    sum_norm_u_list = []
    list_d_delta = []
    list_norm_u = []

    obs = test_env.reset()
    progress_bar = tqdm(total=num_episodes, desc="Test en cours")

    #positions_path = os.path.join(savedir, "positions_log.txt")

    for episode in range(num_episodes):
        print(f"\n🎯 Épisode {episode + 1} / {num_episodes}")
        done = False
        truncated = False
        total_reward = 0
        step = 0
        sum_norm_u = 0
        initial_pos = None
        goal_pos = None

        # Ouverture du fichier pour l’en-tête de l’épisode
        #with open(positions_path, "a") as pos_file:
        #    pos_file.write(f"=== Episode {episode + 1} ===\n")

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            #action = np.zeros(6)  # Action nulle pour le test
            obs, reward, done, info = test_env.step(action)
            total_reward += reward[0]
            step += 1
            steps_test += 1
            d_delta = info[0]['d_delta']
            norm_u = info[0]['norm_u']
            robot_pos = info[0]['robot_position'][:3]
            sum_norm_u += norm_u

            # Mémoriser initial et goal uniquement au premier step
            #if step == 1:
            #    initial_pos = info[0]['robot_initial_position'][:3]
            #    goal_pos = info[0]['goal_position'][:3]
            #    with open(positions_path, "a") as pos_file:
            #        pos_file.write(f"Initial: {initial_pos}\n")
            #        pos_file.write(f"Goal: {goal_pos}\n")
            #        pos_file.write("Positions:\n")

            list_d_delta.append(d_delta)
            list_norm_u.append(norm_u)

            # with open(positions_path, "a") as pos_file:
            #     pos_file.write(f"{robot_pos}\n")

        nb_steps_episode.append(step)
        sum_norm_u_list.append(sum_norm_u)

        print(f"✅ Fin de l’épisode {episode + 1} - Total Reward: {total_reward:.2f}, Steps: {step}")
        progress_bar.update(1)

    progress_bar.close()
    test_env.close()

    # Metrics
    nb_success = info[0]['nb_success']
    nb_collisions = info[0]['nb_collisions']
    nb_timeouts = info[0]['nb_timeouts']

    print(f"success rate (%): {nb_success / num_episodes * 100:.2f}%")
    print(f"collision rate (%): {nb_collisions / num_episodes * 100:.2f}%")
    print(f"timeout rate (%): {nb_timeouts / num_episodes * 100:.2f}%")
    print(f"mean of d_delta: {np.mean(list_d_delta):.2f}")
    print(f"std of d_delta: {np.std(list_d_delta):.2f}")
    print(f"mean of norm_u: {np.mean(list_norm_u):.2f}")
    print(f"mean number of steps: {np.mean(nb_steps_episode):.2f}")
    print(f"mean of sum of norm_u: {np.mean(sum_norm_u_list):.2f}")

    # Enregistrement des métriques globales
    metrics_path = os.path.join(savedir, "metrics.txt")
    with open(metrics_path, "a") as f:
        f.write("===== Résultats du test =====\n")
        f.write(f"Nombre d'épisodes : {num_episodes}\n")
        f.write(f"Seed utilisée : {test_seed}\n")
        f.write(f"success rate (%): {nb_success / num_episodes * 100:.2f}%\n")
        f.write(f"collision rate (%): {nb_collisions / num_episodes * 100:.2f}%\n")
        f.write(f"timeout rate (%): {nb_timeouts / num_episodes * 100:.2f}%\n")
        f.write(f"mean of d_delta: {np.mean(list_d_delta):.2f}\n")
        f.write(f"std of d_delta: {np.std(list_d_delta):.2f}\n")
        f.write(f"mean of norm_u: {np.mean(list_norm_u):.2f}\n")
        f.write(f"mean number of step: {np.mean(nb_steps_episode):.2f}\n")
        f.write(f"mean of sum of norm_u: {np.mean(sum_norm_u_list):.2f}\n")
        f.write("Note: Test des courants marins\n")
        f.write("--------------------------------------------------\n\n")



if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train_model(args.train_seed, args.timesteps)
    elif args.mode == "test":
        test_model(args.test_seed, args.episodes)
