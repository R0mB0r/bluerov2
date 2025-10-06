import matplotlib.pyplot as plt
import numpy as np

file_paths = ["SAC_savedir_4/metrics.txt",
               "SAC_savedir_6/metrics.txt",
               "SAC_savedir_7/metrics.txt",
               "PPO_savedir_4/metrics.txt",
               "PPO_savedir_6/metrics.txt",
               "PPO_savedir_7/metrics.txt"]
              


#file_paths =["SAC_savedir_0/metrics.txt",
#"SAC_savedir_1/metrics.txt",
#"SAC_savedir_2/metrics.txt",
#"SAC_savedir_3/metrics.txt",
#"SAC_savedir_4/metrics.txt",
#"SAC_savedir_5/metrics.txt",
#"SAC_savedir_6/metrics.txt",
#"SAC_savedir_7/metrics.txt"]


def clean_key(key):
    return key.strip().lower()

ordered_metrics = [
    "success rate (%)",
    "collision rate (%)",
    "timeout rate (%)",
    "mean number of steps",
    "mean of d_delta",
    "std of d_delta",
]

# List of metrics where a smaller number is better
invert_metrics = [
    "collision rate (%)",
    "timeout rate (%)",
    "mean of d_delta",
    "std of d_delta",
    "mean of norm_u",
    "mean of sum of norm_u",
    "mean number of steps"
]

# --- Read the data ---
data_dict = {}
for file_path in file_paths:
    metrics = {}
    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("====") or line.startswith("---"):
                    continue
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = clean_key(key)
                    value = value.strip()
                    try:
                        value = float(value.replace("%", ""))
                    except ValueError:
                        pass
                    metrics[key] = value
        data_dict[file_path] = metrics
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        data_dict[file_path] = {}

# --- Prepare the table ---
cell_text = []
cell_colors = []

for metric in ordered_metrics:
    row = [metric.title()]
    colors = ['lightblue']
    values = []
    for file_path in file_paths:
        val = data_dict[file_path].get(metric, None)
        print(f"Metric '{metric}' from '{file_path}': {val}")
        values.append(val)
        if isinstance(val, (int, float)):
            row.append(f"{val:.2f}")  # 👉 format à 2 décimales
        else:
            row.append("N/A")

    numeric_values = [v for v in values if isinstance(v, (int, float))]
    if numeric_values:
        if metric in invert_metrics:
            max_val = min(numeric_values)  # inverted for table
            min_val = max(numeric_values)
        else:
            max_val = max(numeric_values)
            min_val = min(numeric_values)
        for v in values:
            if not isinstance(v, (int, float)):
                colors.append("white")
            elif v == max_val:
                colors.append("lightgreen")
            elif v == min_val:
                colors.append("lightcoral")
            else:
                colors.append("lightyellow")
    else:
        colors.extend(["white"]*len(values))

    cell_text.append(row)
    cell_colors.append(colors)

# --- Figure ---
fig = plt.figure(figsize=(18, 18))

# Table on top
ax_table = plt.subplot2grid((1,1), (0,0))
ax_table.axis('off')
table = ax_table.table(
    cellText=cell_text,
    cellColours=cell_colors,
    #colLabels=["Metric"] + [f"SAC {i}" for i in range(len(file_paths))],
    #colLabels=["Metric"] + ["SAC 4", "SAC 6", "SAC 7", "PPO 4", "PPO 6"],
    #colLabels=["Metric (Kr,f_RL)"] + ["(40,25Hz)","(1,55Hz)","(40,55Hz)","(100,55Hz)","(1,25Hz)","(100,25Hz)","(1,10Hz)","(40,10Hz)"],
    colLabels=["Metric (Kr,f_RL)"] + ["SAC(1,25Hz)","SAC(1,10Hz)","SAC(40,10Hz)"] + ["PPO(1,25Hz)","PPO(1,10Hz)","PPO(40,10Hz)"],
    
    cellLoc='center',
    loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(20)
table.scale(1, 4)  # largeur réduite à 80%, hauteur doublée

# --- Ajuster largeur de la première colonne ---
for key, cell in table.get_celld().items():
    row, col = key
    if col == 0:  # Première colonne (Metric)
        cell.set_width(0.15)  # Mets une largeur fixe (ex: 25%)
    else:
        cell.set_width(0.1)   # Ajuste les autres si besoin


ax_table.set_title("Metrics Comparison", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
