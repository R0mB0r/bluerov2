import matplotlib.pyplot as plt
import numpy as np
import os
import re

def get_latest_savedir():
    #base = "PPO_savedir_"
    base = "SAC_savedir_"
    dirs = [d for d in os.listdir(".") if re.match(rf"{base}\d+$", d)]
    print(f"Found directories: {dirs}")
    pairs = sorted([int(d.split("_")[-1]) for d in dirs if int(d.split("_")[-1]) % 2 == 0])
    odd = sorted([int(d.split("_")[-1]) for d in dirs if int(d.split("_")[-1]) % 2 != 0])
    #if not pairs:
    #    raise FileNotFoundError("Aucun dossier de sauvegarde pair trouvé.")
    #savedir = f"{base}{pairs[-1]}"
    savedir = f"{base}{odd[-1]}" 
    return savedir

savedir = get_latest_savedir()
#savedir = "SAC_savedir_13"
print(f"📂 Dossier de sauvegarde utilisé : {savedir}")
filepath = os.path.join(savedir, "distances_over_episodes_test.txt")

# Charger les distances et rewards depuis le fichier texte (format: distance,reward)
distances = []
rewards = []
nb_steps = []
with open(filepath, "r") as f:
    for line in f:
        try:
            parts = line.strip().split(',')
            if len(parts) == 3:
                distances.append(float(parts[0]))
                rewards.append(float(parts[1]))
                nb_steps.append(int(parts[2]))
        except ValueError:
            continue  # Ignore les lignes non numériques

distances = np.array(distances)
rewards = np.array(rewards)
nb_steps = np.array(nb_steps)
episodes = np.arange(len(distances))

# Points rouges pour y <= 3, bleus sinon
below = distances <= 3
above = distances > 3

# Calcul de la moyenne cumulative
mean_distances = np.cumsum(distances) / (np.arange(1, len(distances) + 1))
mean_rewards = np.cumsum(rewards) / (np.arange(1, len(rewards) + 1))
mean_nb_steps = np.cumsum(nb_steps) / (np.arange(1, len(nb_steps) + 1))

# Calcul du pourcentage cumulé de succès (distance <= 3)
cumulative_success = np.cumsum(below)
cumulative_success_percent = 100 * cumulative_success / (np.arange(1, len(distances) + 1))

# Moyenne cumulative des rewards pour les succès uniquement
success_rewards = np.where(below, rewards, 0)
success_counts = np.cumsum(below)
cumulative_success_rewards = np.cumsum(success_rewards)
mean_success_rewards = np.zeros_like(rewards)
valid = success_counts > 0
mean_success_rewards[valid] = cumulative_success_rewards[valid] / success_counts[valid]

# Pourcentage glissant de succès sur 100 épisodes
window = 10
sliding_success_percent = np.convolve(below, np.ones(window, dtype=int), 'valid') / window * 100
sliding_episodes = np.arange(window - 1, len(distances))

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# --- Graphe 1 : Distances ---
ax1.plot(distances, marker='o', linestyle='-', color='gray', alpha=0.3, label='Distance')
ax1.scatter(episodes[below], distances[below], color='green', label='Succès (<=3)')
ax1.plot(episodes, mean_distances, color='green', linewidth=2, label='Moyenne cumulative')
ax1.axhline(y=3, color='r', linestyle='--', label='Seuil (3)')
# Ajout du pourcentage cumulé de succès (axe secondaire)
ax1b = ax1.twinx()
ax1b.plot(episodes, cumulative_success_percent, color='purple', linestyle='-', linewidth=2, label='% Succès cumulé (<=3)')
ax1b.plot(sliding_episodes, sliding_success_percent, color='orange', linestyle='-', linewidth=2, label='% Succès glissant (100)')
ax1b.set_ylabel("Pourcentage de succès (%)", color='purple')
ax1b.tick_params(axis='y', labelcolor='purple')

ax1.set_ylabel("Distance à la fin de l'épisode")
ax1.set_title("Distance to goal à la fin de chaque épisode")
ax1.grid(True)

# Gestion des légendes pour les deux axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_1b, labels_1b = ax1b.get_legend_handles_labels()
ax1.legend(lines_1 + lines_1b, labels_1 + labels_1b, loc='upper right')

# --- Graphe 2 : Rewards ---
ax2.plot(rewards, marker='o', linestyle='-', color='gray', alpha=0.3, label='Reward')
ax2.plot(episodes, mean_rewards, color='brown', linewidth=2, label='Moyenne cumulative reward')
ax2.plot(episodes, mean_success_rewards, color='green', linestyle='--', linewidth=2, label='Moyenne cumulative reward (succès)')
ax2.scatter(episodes[below], rewards[below], color='green', alpha=0.5, label='Succès (<=3)')
ax2.set_xlabel("Épisode")
ax2.set_ylabel("Reward total de l'épisode")
ax2.set_title("Reward total à la fin de chaque épisode")
ax2.grid(True)
ax2.legend(loc='upper right')

# --- Graphe 3 : Nombre de pas par épisode ---
ax3.plot(nb_steps, marker='o', linestyle='-', color='gray', alpha=0.3, label='Nombre de pas')
ax3.plot(episodes, mean_nb_steps, color='brown', linewidth=2, label='Moyenne cumulative des pas')
ax3.scatter(episodes[below], nb_steps[below], color='green', alpha=0.5, label='Succès (<=3)')
ax3.set_xlabel("Épisode")
ax3.set_ylabel("Nombre de pas")
ax3.set_title("Nombre de pas par épisode")
ax3.grid(True)
ax3.legend(loc='upper right')

plt.tight_layout()
plt.show()