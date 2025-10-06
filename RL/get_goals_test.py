import numpy as np

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

def clean_position(line):
    """Extrait les coordonnées numériques d'une ligne 'Initial:', 'Goal:' ou 'Position'."""
    text = line.strip().split(":")[1].strip()
    text = text.replace("[", "").replace("]", "")
    nums = text.split()
    return [float(x) for x in nums]

def parse_episode_avg_distance(file_path, episode_number):
    """Retourne la moyenne des distances d'un épisode donné (à partir de positions_log.txt)."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    episode_tag = f"=== Episode {episode_number} ==="
    inside_episode = False
    inside_positions = False
    initial, goal = None, None
    positions = []

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

    if initial is None or goal is None or not positions:
        return None

    # Calcul distances à la droite
    distances = [distance_point_segment_3d_cross(initial, goal, p) for p in positions]
    return float(np.mean(distances))

def enrich_test_file(test_file, positions_file, output_file):
    """Ajoute la moyenne distance droite Initial–Goal au fichier test existant
       + calcule la moyenne générale des distances sur tous les épisodes."""
    with open(test_file, "r") as f:
        lines = f.readlines()

    header = lines[1]
    data_lines = lines[2:]

    # Si l'entête n'a pas encore la colonne, on l'ajoute
    if "AvgDistanceToLine" not in header:
        header = header.strip() + ",AvgDistanceToLine\n"

    new_lines = [lines[0], header]

    episode_number = 1
    all_distances = []  # ⬅️ On stocke toutes les moyennes d'épisodes

    for line in data_lines:
        line = line.strip()
        if not line or line.startswith("=="):
            new_lines.append(line + "\n")
            continue

        parts = line.split(",")
        if len(parts) < 3:
            continue

        avg_dist = parse_episode_avg_distance(positions_file, episode_number)
        if avg_dist is None:
            avg_dist = ""
        else:
            all_distances.append(avg_dist)

        new_line = line + f",{avg_dist}\n"
        new_lines.append(new_line)

        episode_number += 1

    # Calcul global : moyenne sur tous les épisodes
    if all_distances:
        global_mean = np.mean(all_distances)
        new_lines.append(f"\n# Global mean AvgDistanceToLine: {global_mean:.6f}\n")

    # Sauvegarde
    with open(output_file, "w") as f:
        f.writelines(new_lines)

    print(f"✅ Fichier enrichi écrit dans {output_file}")
    if all_distances:
        print(f"📊 Moyenne générale des distances = {global_mean:.6f}")
        print(f"std = {np.std(all_distances):.6f}")

# Exemple d'utilisation
#for i in range(8):
#    test_file = f"SAC_savedir_{i}/distances_over_episodes_test.txt"
#    positions_file = f"SAC_savedir_{i}/positions_log.txt"
#    output_file = f"SAC_savedir_{i}/distances_over_episodes_test_enriched.txt"
#    
#    enrich_test_file(test_file, positions_file, output_file)

test_file = f"SAC_savedir_6/distances_over_episodes_test_mili.txt"
positions_file = f"SAC_savedir_6/positions_log.txt"
output_file = f"SAC_savedir_6/distances_over_episodes_test_enriched.txt"

enrich_test_file(test_file, positions_file, output_file)
