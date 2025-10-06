from graphviz import Digraph

# -------------------------------
# Diagramme RL infographique
# -------------------------------
dot = Digraph('BlueROVEnv_RL_Infographic', format='png')
dot.attr(rankdir='TB', size='12,14', splines='ortho', nodesep='0.4', ranksep='0.2')

def add_class(
    name: str, 
    attributes: list[str], 
    methods: list[str], 
    parent: str = None,
    header_color: str = "#85C1E9"
) -> None:
    """Ajoute une classe RL avec blocs colorés et icônes."""
    
    attr_html = []
    for a in attributes:
        if any(k in a for k in ['observation', 'goal', 'robot_position']):
            color = "#ABEBC6"  # vert clair
            icon = "🟢"
        elif any(k in a for k in ['action', 'prev_action']):
            color = "#AED6F1"  # bleu
            icon = "🔵"
        elif any(k in a for k in ['reward', 'coeff_reward', 'episodes_reward', 'nb_success', 'nb_collisions']):
            color = "#FCF3CF"  # jaune
            icon = "🟡"
        elif 'done' in a or 'timeout' in a or 'collision' in a:
            color = "#F5B7B1"  # rouge
            icon = "🔴"
        else:
            color = "#E5E8E8"
            icon = "⚪"
        attr_html.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="{color}">{icon} {a}</TD></TR>')
    
    meth_html = [f'<TR><TD ALIGN="LEFT" BGCOLOR="#D2B4DE">▸ {m}</TD></TR>' for m in methods]

    label = f'''<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" STYLE="ROUNDED">
        <TR><TD BGCOLOR="{header_color}" ALIGN="CENTER" CELLPADDING="8" STYLE="ROUNDED"><B>{name}</B></TD></TR>
        {''.join(attr_html)}
        {''.join(meth_html)}
    </TABLE>>'''

    dot.node(name, label=label, shape='plaintext', style='shadow')

    if parent:
        dot.edge(parent, name, arrowhead="empty", style="bold", color="#555555")

# -------------------------------
# Classe parent
# -------------------------------
add_class("gym.Env", [], [], header_color="#AED6F1")

# -------------------------------
# Classe RL spécifique : BlueROVEnv
# -------------------------------
attributes_rl = [
    "observation_space : gym.spaces.Box",
    "action_space : gym.spaces.Box",
    "prev_action : np.ndarray",
    "goal_position : np.ndarray",
    "robot_position : np.ndarray",
    "episodes_reward : float",
    "coeff_reward : float",
    "step_freq : float",
    "current_step : int",
    "done : bool",
    "collision : bool",
    "timeout : bool",
    "nb_success : int",
    "nb_collisions : int",
]

methods_rl = [
    "__init__(seed=None, save_dir=None, mode='train' or 'test')",
    "step(action) : tuple (obs, reward, done, info)",
    "reset(seed=None, options=None) : tuple (obs, info)",
    "close() : void"
]

add_class("BlueROVEnv", attributes_rl, methods_rl, parent="gym.Env", header_color="#85C1E9")

# -------------------------------
# Blocs d'infographie pour le flux RL
# -------------------------------
# Etats / Observations
dot.node("State", "🟢 State / Observation", shape="box", style="filled,rounded", fillcolor="#ABEBC6")
# Actions
dot.node("Action", "🔵 Action", shape="box", style="filled,rounded", fillcolor="#AED6F1")
# Reward / Metrics
dot.node("Reward", "🟡 Reward / Metrics", shape="box", style="filled,rounded", fillcolor="#FCF3CF")
# Terminal
dot.node("Terminal", "🔴 Terminal (done/collision/timeout)", shape="box", style="filled,rounded", fillcolor="#F5B7B1")

# Flux RL
dot.edge("BlueROVEnv", "State", style="solid", color="#555555")
dot.edge("State", "Action", style="solid", color="#555555")
dot.edge("Action", "Reward", style="solid", color="#555555")
dot.edge("Action", "Terminal", style="solid", color="#555555")
dot.edge("Reward", "BlueROVEnv", style="dashed", color="#555555")
dot.edge("Terminal", "BlueROVEnv", style="dashed", color="#555555")

# -------------------------------
# Export du diagramme infographique
# -------------------------------
dot.render("BlueROVEnv_RL_Infographic", format="png", cleanup=True)

