import numpy as np
from control import BlueROVEnv
import time

def test_bluerov_env():
    env = BlueROVEnv()  # Initialisation de l'environnement

    num_episodes = 10  # Nombre d'épisodes de test

    for episode in range(num_episodes):
        print(f"\n🎯 Épisode {episode + 1} / {num_episodes}")
        
        observation = env.reset()  # Réinitialisation de l'épisode
        done = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not (done or truncated):  
            action = np.array([5.0, 5.0, 5.0, 5.0, 0.0, 0.0])  # Action nulle (pas de mouvement)
            observation, reward, done, truncated, _ = env.step(action)  
            
            total_reward += reward
            step_count += 1

            print(f"🔹 Step {step_count}: Reward={reward:.2f}, Done={done}, Truncated={truncated}")
            time.sleep(0.1)

        print(f"✅ Fin de l'épisode {episode + 1} - Total Reward: {total_reward:.2f}, Steps: {step_count}")

        time.sleep(1)

    env.close()  # Fermeture de l'environnement

if __name__ == "__main__":
    test_bluerov_env()
