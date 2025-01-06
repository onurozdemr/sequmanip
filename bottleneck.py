import robotic as ry
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib import cm
from matplotlib.colors import Normalize
from collections import defaultdict
from math import sqrt

OBJ_NAME = "obj"

def object_vertex(config: ry.Config, step_size: float) -> np.ndarray:

    obj_frame = config.getFrame(OBJ_NAME)

    obj_position_world = obj_frame.getPosition()[:2]

    obj_vertex_grid = obj_position_world / step_size
    
    offset_val = int(2 / step_size)
    grid_offset = np.array([offset_val, offset_val], dtype=int)

    
    obj_vertex = (obj_vertex_grid).astype(int) + grid_offset

    return obj_vertex

def graph_config(config: ry.Config, step_size: float):

    OBJ_NAME = "obj"
    
    all_frames = config.getFrameNames()
    wall_names = [f for f in all_frames if "wall" in f.lower()]

    # -- 2. Determine object size offset --
    obj_frame = config.getFrame(OBJ_NAME)
    obj_width, obj_height = obj_frame.getSize()[:2]
    size_offset = np.array([obj_width, obj_height])

    # -- 3. Prepare empty occupancy grid (4 meters x 4 meters, with step_size spacing) --
    grid_dim = int(4 / step_size)
    occupancy_grid = np.ones((grid_dim, grid_dim), dtype=int)

    # This offset is used to shift everything so that walls at negative coords
    # still end up in a valid array index.
    offset_val = int(2 / step_size)
    grid_offset = np.array([offset_val, offset_val], dtype=int)

    # -- 4. Mark walls in the occupancy grid --
    for wall_name in wall_names:
        wall_frame = config.getFrame(wall_name)

        # Increase wall size by object's dimension (as in original logic)
        wall_size = wall_frame.getSize()[:2] + size_offset
        wall_pos = wall_frame.getPosition()[:2]

        # Determine the anchor and opposite corner
        wall_anchor = wall_pos - wall_size / 2.0
        wall_end = wall_pos + wall_size / 2.0

        # Convert to grid indices (integer)
        wall_indices = np.array([wall_anchor, wall_end]) / step_size + grid_offset
        wall_indices = wall_indices.astype(int)

        # Clip to ensure indices are within the grid
        wall_indices = np.clip(wall_indices, 0, occupancy_grid.shape)

        # Mark those cells as occupied (0)
        (r0, c0), (r1, c1) = wall_indices
        occupancy_grid[r0:r1, c0:c1] = 0

    # -- 5. Convert occupancy grid to graph and get the obj vertex --
    graph = graph_from_grid(occupancy_grid)
    obj_vertex = object_vertex(config, step_size)

    return graph, obj_vertex

def graph_from_grid(grid: np.ndarray) -> dict:

    rows, cols = grid.shape
    graph = defaultdict(list)

    moves_8 = [
        (-1,  0, 1),  ( 1,  0, 1),
        ( 0, -1, 1),  ( 0,  1, 1),
        (-1, -1, sqrt(2)),  (-1, 1, sqrt(2)),
        ( 1, -1, sqrt(2)),  ( 1, 1, sqrt(2))
    ]

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 1:
                # For each valid neighbor, add an edge
                for dx, dy, cost in moves_8:
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        # Append neighbor and weight
                        graph[(r, c)].append(((nr, nc), cost))
    return graph

def bottleneck_via_betweenness_approx(config, inner_radius, outer_radius, step_size, threshold=0.1, sample_size=100):

    graph, obj_vertex = graph_config(config, step_size=step_size)

    G = nx.Graph()
    for node, neighbors in graph.items():
        for neighbor, weight in neighbors:
            G.add_edge(node, tuple(neighbor), weight=weight)

    bc = nx.betweenness_centrality(G, weight='weight', normalized=True, k=sample_size)

    obj_pos = np.array(obj_vertex) * step_size
    def within_region(vertex):
        vertex_pos = np.array(vertex) * step_size
        dist = np.linalg.norm(vertex_pos - obj_pos)
        return inner_radius <= dist <= outer_radius

    bottleneck_nodes = [v for v in G.nodes if bc[v] > threshold and within_region(v)]
    return bottleneck_nodes, bc, obj_vertex


def grid_vertex_to_env(vertex, step_size):
    """
    Convert a grid vertex (i, j) back to environment/world coordinates (x, y).
    """
    offset_val = int(2 / step_size)  
    grid_offset = np.array([offset_val, offset_val], dtype=int)

    env_coords = (np.array(vertex) - grid_offset) * step_size
    return env_coords


def plot_betweenness(graph, bc_values, obj_vertex):

    fig, ax = plt.subplots(figsize=(8, 6))

    for node, neighbors in graph.items():
        for neighbor, _ in neighbors:
            x_vals = [node[1], neighbor[1]]
            y_vals = [node[0], neighbor[0]]
            ax.plot(x_vals, y_vals, c='lightgray', linewidth=0.5, zorder=1)

    all_nodes = list(graph.keys())
    bc_array = np.array([bc_values.get(n, 0.0) for n in all_nodes])
    node_positions = np.array([[n[1], n[0]] for n in all_nodes])

    norm = Normalize(vmin=bc_array.min(), vmax=bc_array.max())
    colors = cm.viridis(norm(bc_array))
    ax.scatter(
        node_positions[:, 0], 
        node_positions[:, 1], 
        c=colors, 
        s=10, 
        alpha=0.8, 
        zorder=2
    )

    ax.plot(obj_vertex[1], obj_vertex[0], 'bo', markersize=8, zorder=3)

    sm = cm.ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label('Betweenness Centrality', fontsize=12)

    ax.invert_yaxis()
    ax.set_aspect('equal')

    ax.set_title("Betweenness Centrality", fontsize=14)
    ax.set_xlabel("Grid Column Index")
    ax.set_ylabel("Grid Row Index")

    plt.tight_layout()
    plt.show()

"""
inner_radius = 1.0
outer_radius = 3.0
step_size = 0.05
threshold = 0.05  
sample_size = 50  

config = ry.Config()
config.addFile("p3-maze.g")


bottleneck_nodes, bc_values, obj_vertex = bottleneck_via_betweenness_approx(
    config, inner_radius, outer_radius, step_size, threshold, sample_size
)

graph, _ = graph_config(config, step_size)

plot_betweenness(graph, bc_values, obj_vertex)

for b_node in bottleneck_nodes:
    env_pos = grid_vertex_to_env(b_node, step_size)
    print("Bottleneck vertex:", b_node, " -> Environment coords:", env_pos)

del config


"""
