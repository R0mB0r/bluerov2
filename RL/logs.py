import matplotlib.pyplot as plt
import numpy as np

# Charger les distances et rewards depuis le fichier texte (format: distance,reward)
distances = []
rewards = []
with open("RL/distances_over_episodes_test_3.txt", "r") as f:
    for line in f:
        try:
            parts = line.strip().split(',')
            if len(parts) == 2:
                distances.append(float(parts[0]))
                rewards.append(float(parts[1]))
        except ValueError:
            continue  # Ignore les lignes non numériques

distances = np.array(distances)
rewards = np.array(rewards)
episodes = np.arange(len(distances))

# Points rouges pour y <= 3, bleus sinon
below = distances <= 3
above = distances > 3

# Calcul de la moyenne cumulative
mean_distances = np.cumsum(distances) / (np.arange(1, len(distances) + 1))
mean_rewards = np.cumsum(rewards) / (np.arange(1, len(rewards) + 1))

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
window = 100
sliding_success_percent = np.convolve(below, np.ones(window, dtype=int), 'valid') / window * 100
sliding_episodes = np.arange(window - 1, len(distances))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# --- Graphe 1 : Distances ---
ax1.plot(distances, marker='o', linestyle='-', color='gray', alpha=0.3, label='Distance')
ax1.scatter(episodes[below], distances[below], color='green', label='Succès (<=3)')
ax1.plot(episodes, mean_distances, color='green', linewidth=2, label='Moyenne cumulative')
ax1.axhline(y=3, color='r', linestyle='--', label='Seuil (3)')
# Ajout du pourcentage cumulé de succès (axe secondaire)
ax1b = ax1.twinx()
ax1b.plot(episodes, cumulative_success_percent, color='purple', linestyle='-', linewidth=2, label='% Succès cumulé (<=3)')
ax1b.plot(sliding_episodes, sliding_success_percent, color='orange', linestyle='-', linewidth=2, label='% Succès glissant (10)')
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

plt.tight_layout()
plt.show()