import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def clean_position(line):
    text = line.strip().split(":")[1].strip()
    text = text.replace("[", "").replace("]", "")
    text = text.replace("  ", " ")  # Nettoyage double espace
    nums = text.split()
    return [float(x) for x in nums]

def set_axes_equal(ax):
    """Force les axes X, Y, Z à avoir la même échelle"""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range/2, x_middle + max_range/2])
    ax.set_ylim3d([y_middle - max_range/2, y_middle + max_range/2])
    ax.set_zlim3d([z_middle - max_range/2, z_middle + max_range/2])


def parse_episode_positions(file_path, episode_number):
    with open(file_path, "r") as f:
        lines = f.readlines()

    episode_tag = f"=== Episode {episode_number} ==="
    positions = []
    initial = None
    goal = None
    inside_episode = False
    inside_positions = False

    for line in lines:
        line = line.strip()

        if line.startswith("=== Episode"):
            inside_episode = (line == episode_tag)
            inside_positions = False
            continue

        if inside_episode:
            if line.startswith("Initial:"):
                initial = clean_position(line)
            elif line.startswith("Goal:"):
                goal = clean_position(line)
            elif line == "Positions:":
                inside_positions = True
            elif inside_positions and line.startswith("["):
                pos = line.replace("[", "").replace("]", "")
                coords = [float(x) for x in pos.split()]
                positions.append(coords)

    return initial, goal, positions

def plot_sphere(ax, center, radius=3.0, color='blue', alpha=0.2):
    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:15j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

def plot_episode_3d(initial, goal, positions, episode_number):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Trajectoire réelle
    traj = list(zip(*positions))
    ax.plot(traj[0], traj[1], traj[2], label="Trajectoire réelle", color="red", linewidth=2)

    # Ligne droite idéale
    line = list(zip(*[initial, goal]))
    ax.plot(line[0], line[1], line[2], label="Trajectoire idéale", linestyle='--', color="blue")

    # Points
    ax.scatter(*initial, color="green", s=80, label="Position initiale", marker='o')
    ax.scatter(*goal, color="purple", s=80, label="But", marker='^')

    # Sphère autour du goal
    plot_sphere(ax, center=goal, radius=3.0, color='blue', alpha=0.2)

    ax.set_title(f"Trajectoire 3D - Épisode {episode_number}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(True)
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()

# === Exemple d'utilisation ===

#file_path = "PPO_savedir_0/positions_log.txt"  # adapte si besoin
file_path = "SAC_savedir_9/positions_log.txt"  # adapte si besoin
episode_number = 1  # numéro de l’épisode à tracer

initial, goal, positions = parse_episode_positions(file_path, episode_number)

if initial and goal and positions:
    plot_episode_3d(initial, goal, positions, episode_number)
else:
    print(f"❌ Épisode {episode_number} introuvable ou données incomplètes.")
