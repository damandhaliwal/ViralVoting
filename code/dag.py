import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from utils import get_project_paths


def create_dag_plots():
    paths = get_project_paths()

    # Plot 1 - Basic DAG
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

    nx.draw(
        graph,
        pos,
        labels=labels,
        node_size=12000,
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

    # Plot 2 - Information Diffusion with Spillover Effects
    dag_adj_matrix_2 = np.array([
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0]
    ])

    graph_2 = nx.from_numpy_array(dag_adj_matrix_2, create_using=nx.DiGraph)

    NODE_NAMES_2 = ['Treatment\n(Household A)', 'Voted(Y)\n(Household A)',
                    'Voted(Y)\n(Household B)', 'Geographic\nProximity',
                    'Information\nSharing', 'Voting\nHistory', 'Demographics']
    labels_2 = {i: name for i, name in enumerate(NODE_NAMES_2)}

    plt.figure(figsize=(14, 10))

    pos_2 = {
        6: (-2, 0),
        5: (-2, 1.5),
        3: (-2, -1.5),
        0: (3, 0),
        4: (1.5, -1.5),
        1: (3, 1),
        2: (3, -1)
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
        [0, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 0, 0]
    ])

    graph_3 = nx.from_numpy_array(dag_adj_matrix_3, create_using=nx.DiGraph)

    NODE_NAMES_3 = ['Treatment\n(Household A)', 'Voted(Y)\n(Household A)',
                    'Voted(Y)\n(Household B)', 'Geographic\nProximity',
                    'Information\nSharing', 'Voting\nHistory', 'Demographics',
                    'Unobserved\nCivic Engagement']
    labels_3 = {i: name for i, name in enumerate(NODE_NAMES_3)}

    plt.figure(figsize=(14, 10))

    pos_3 = {
        6: (-2, 0),
        5: (-2, 1.5),
        3: (-2, -1.5),
        7: (0, 2),
        0: (3, 0),
        4: (1.5, -1.5),
        1: (3, 1),
        2: (3, -1)
    }

    nx.draw(
        graph_3,
        pos_3,
        labels=labels_3,
        node_size=10000,
        node_color=['lightgreen', 'red', 'red', 'lightblue', 'lightyellow', 'lightblue', 'lightblue', 'lightgray'],
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

    output_path_3 = plots_dir / 'figure2.png'
    plt.savefig(output_path_3, dpi=1200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Confounding DAG saved to {output_path_3}")

    return {
        'dag_plot_1': str(output_path),
        'dag_plot_2': str(output_path_2),
        'dag_plot_3': str(output_path_3)
    }