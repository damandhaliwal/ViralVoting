import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import get_project_paths

paths = get_project_paths()

dag_adj_matrix = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 1, 0, 0]
])
graph = nx.from_numpy_array(dag_adj_matrix, create_using=nx.DiGraph)

NODE_NAMES = ['Voted(Y)', 'Voting History', 'Treatment(s)', 'Demographics']
labels = {i: name for i, name in enumerate(NODE_NAMES)}

plt.figure(figsize=(12, 12))

pos = nx.spring_layout(graph, seed=123, k=0.2, iterations=100)

# Draw everything at once with correct parameters
nx.draw(
    graph,
    pos,
    labels=labels,
    node_size=12000,  # Smaller nodes
    node_color=['red', 'lightblue', 'lightgreen', 'lightblue'],
    edgecolors='black',
    linewidths=1,
    font_size=14,
    font_color='black',
    edge_color='black',
    width=1,
    arrows=True,
    arrowsize=30,
    arrowstyle='->',
    connectionstyle='arc3,rad=0.1',
    with_labels=True
)

plt.axis('off')
plt.tight_layout()

plots_dir = Path(paths['plots'])
plots_dir.mkdir(parents=True, exist_ok=True)
output_path = plots_dir / 'dag_plot_1.png'
plt.savefig(output_path, dpi=1200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"DAG plot saved to {output_path}")

# Plot 2 - Information Diffusion with Spillover Effects (FIXED LAYOUT)

dag_adj_matrix_2 = np.array([
    [0, 1, 0, 0, 1, 0, 0],  # Treatment (HH A) → Voted A, Information Sharing
    [0, 0, 0, 0, 0, 0, 0],  # Voted A → nothing
    [0, 0, 0, 0, 0, 0, 0],  # Voted B → nothing
    [0, 0, 0, 0, 1, 0, 0],  # Geographic Proximity → Information Sharing
    [0, 0, 1, 0, 0, 0, 0],  # Information Sharing → Voted B (spillover)
    [0, 1, 1, 0, 0, 0, 0],  # Voting History → Voted A, Voted B
    [0, 1, 1, 0, 0, 1, 0]   # Demographics → Voted A, Voted B, Voting History
])

graph_2 = nx.from_numpy_array(dag_adj_matrix_2, create_using=nx.DiGraph)

NODE_NAMES_2 = ['Treatment\n(Household A)', 'Voted(Y)\n(Household A)', 
                'Voted(Y)\n(Household B)', 'Geographic\nProximity',
                'Information\nSharing', 'Voting\nHistory', 'Demographics']
labels_2 = {i: name for i, name in enumerate(NODE_NAMES_2)}

plt.figure(figsize=(14, 10))

# Manual positioning for clearer hierarchical structure
pos_2 = {
    6: (-2, 0),      # Demographics (left, middle)
    5: (-2, 1.5),    # Voting History (left, upper)
    3: (-2, -1.5),   # Geographic Proximity (left, lower)
    0: (3, 0),       # Treatment (center)
    4: (1.5, -1.5),    # Information Sharing (right-center, lower)
    1: (3, 1),       # Voted A (right, upper)
    2: (3, -1)       # Voted B (right, lower)
}

nx.draw(
    graph_2,
    pos_2,
    labels=labels_2,
    node_size=10000,
    node_color=['lightgreen', 'red', 'red', 'lightblue', 'lightyellow', 'lightblue', 'lightblue'],  
    edgecolors='black',
    linewidths=1,
    font_size=11,
    font_color='black',
    edge_color='black',
    width=1,
    arrows=True,
    arrowsize=25,
    arrowstyle='->',
    connectionstyle='arc3,rad=0.1',
    with_labels=True
)

plt.axis('off')
plt.tight_layout()

output_path_2 = plots_dir / 'dag_plot_2.png'
plt.savefig(output_path_2, dpi=1200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Information Diffusion DAG saved to {output_path_2}")

# Plot 3 - Information Diffusion with Unobserved Confounder

dag_adj_matrix_3 = np.array([
    [0, 1, 0, 0, 1, 0, 0, 0],  # Treatment (HH A) → Voted A, Information Sharing
    [0, 0, 0, 0, 0, 0, 0, 0],  # Voted A → nothing
    [0, 0, 0, 0, 0, 0, 0, 0],  # Voted B → nothing
    [0, 0, 0, 0, 1, 0, 0, 0],  # Geographic Proximity → Information Sharing
    [0, 0, 1, 0, 0, 0, 0, 0],  # Information Sharing → Voted B (spillover)
    [0, 1, 1, 0, 0, 0, 0, 0],  # Voting History → Voted A, Voted B
    [0, 1, 1, 0, 0, 1, 0, 0],  # Demographics → Voted A, Voted B, Voting History
    [1, 1, 1, 0, 1, 1, 0, 0]   # Unobserved Civic Engagement → Treatment, Voted A, Voted B, Info Sharing (CONFOUNDER!)
])

graph_3 = nx.from_numpy_array(dag_adj_matrix_3, create_using=nx.DiGraph)

NODE_NAMES_3 = ['Treatment\n(Household A)', 'Voted(Y)\n(Household A)', 
                'Voted(Y)\n(Household B)', 'Geographic\nProximity',
                'Information\nSharing', 'Voting\nHistory', 'Demographics',
                'Unobserved\nCivic Engagement']
labels_3 = {i: name for i, name in enumerate(NODE_NAMES_3)}

plt.figure(figsize=(14, 10))

# Manual positioning with unobserved confounder
pos_3 = {
    6: (-2, 0),      # Demographics (left, middle)
    5: (-2, 1.5),    # Voting History (left, upper)
    3: (-2, -1.5),   # Geographic Proximity (left, lower)
    7: (0, 2),       # Unobserved Civic Engagement (top center) - THE CONFOUNDER
    0: (3, 0),       # Treatment (center)
    4: (1.5, -1.5),    # Information Sharing (right-center, lower)
    1: (3, 1),       # Voted A (right, upper)
    2: (3, -1)       # Voted B (right, lower)
}

nx.draw(
    graph_3,
    pos_3,
    labels=labels_3,
    node_size=10000,
    node_color=['lightgreen', 'red', 'red', 'lightblue', 'lightyellow', 'lightblue', 'lightblue', 'lightgray'],  
    # Treatment, Outcomes, Proximity, Mediator, Covariates, UNOBSERVED CONFOUNDER
    edgecolors='black',
    linewidths=1,
    font_size=11,
    font_color='black',
    edge_color='black',
    width=1,
    arrows=True,
    arrowsize=25,
    arrowstyle='->',
    connectionstyle='arc3,rad=0.1',
    with_labels=True
)

plt.axis('off')
plt.tight_layout()

output_path_3 = plots_dir / 'dag_plot_3.png'
plt.savefig(output_path_3, dpi=1200, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Confounding DAG saved to {output_path_3}")

