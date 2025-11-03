import numpy as np
import networkx as nx
import random
import copy
import itertools
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from IPython.display import HTML, display
import time
from adjustText import adjust_text
from collections import defaultdict

class Graphclass:
    """"Class for studying the behavior of dynamical systems through networks
    Attributes
        From User:
            self.num_nodes- int (Number of nodes in the graph)
            self.full- boolean (Whether or not the BFS should start at every node)
            self.num_colors- int (number of colors randomly chosen from and assigned to edges)
            
        From Constructor:
            self.G- nx graph object (a directed graph randomly constructed based on input method)
                options include random, large (goliath), Watts-Strogatz, Barasi_Albert, and random geometric
            self.weights- np.array (the randomly generated weighted adjacency matrix for G)
            self.subnetworks- list (list of subnetworks found via BFS, with any hanging nodes removed)
            self.subnetworks- list (list of subnetworks found via BFS, complete with hanging nodes)
            self.radii- list (spectral radius of each subnetwork)
            self.predicted- list (nodes predicted to go to zero)
            self.predicted- list (nodes in subnetworks predicted to go to zero)
            self.bfsflop- float (time for the BFS search to terminate)
            self.history- dict (value of each node at each timestep during behavior simulation)
            self.converged- list (the values that each node converged to)
            self.zero_nodes- list (the nodes which went to zero during the simulation)
        if full== True
            self.full_subnetwork- list (list of lists of subnetworks obtained by starting at each node)
            self.full_radii- list (spectral radius of subnetworks)
            self.full_predicted- list (nodes which are predicted to go to zero)
            self.fullflop- float (time for full prediction code to run)
     
    Methods:
        
        """""
        

    def __init__(self, num_nodes, num_edges, num_colors, method, full=False):
        """"Allows for a graph object to be input or the generation of a random graph"""
        # User input attributes
        self.num_nodes = num_nodes
        self.full = full
        self.num_colors = num_colors
        self.method = method
        # Creates desired graph type
        if method == 'random':
             self.G = self.create_graph(num_nodes, num_edges, num_colors)
        if method == 'goliath':
            self.G = self.Goliath()
        if method == 'ws':
            self.G = self.Watts_Strogatz()
        if method == 'ba':
            self.G = self.Barabasi_Albert()
        if method == 'rg':
            self.G = self.random_geometric()
        # Generates random weights
        self.weights = self.create_weights()
        # Process of identifying subnetworks, calculating radii, and making predictions
        start = time.time()
        self.subnetworks = self.breadth_first()
        self.radii = self.spectral_radius(self.subnetworks)
        self.predicted = self.predict(self.radii,self.subnetworks)
        stop = time.time()
        self.bfsflop = stop - start
        # Simulation of behavior
        self.history, self.converged, self.zero_nodes = self.simulate_linear()
        # Repeats the above BFS starting at every node
        if full:
            # Process of identifying subnetworks, calculating radii, and making predictions
            start = time.time()
            self.full_subnetwork = self.full_subnetworks()
            self.full_predicted, self.full_radii = self.predict_full(self.full_subnetwork)
            stop = time.time()
            self.fullflop = stop - start



    """Methods to create Graph objects """
   
    def Watts_Strogatz(self):
        """Creates a Watts Strogatz graph and then gives it colored bidirectional edges"""
        while True:
            num_nodes = self.num_nodes
             # Each node is connected to `k` nearest neighbors
            k = max(1, num_nodes // 5) 
            # Probability of rewiring an edge
            p = 0.3  
            G_ws = nx.watts_strogatz_graph(num_nodes, k, p)
            G = nx.DiGraph()
            G.add_nodes_from(G_ws.nodes())
            for u, v in G_ws.edges():
                # Add edges in both directions
                G.add_edge(u, v)
                G.add_edge(v, u)
            # Assign random colors to edges
            for u, v in G.edges():
                G[u][v]['color'] = random.randint(0, self.num_colors)
            # Check strong connectivity
            if nx.is_strongly_connected(G):
                return G
    
    def Barabasi_Albert(self):
        """Creates a Barabasi Albert graph and then gives it colored bidirectional edges"""
        while True:
            num_nodes = self.num_nodes
            # Each new node attaches to `m` existing nodes
            m = max(1, num_nodes // 5)  
            G_ba = nx.barabasi_albert_graph(num_nodes, m)
            # Convert it to a directed graph with bidirectional edges
            G = nx.DiGraph()
            G.add_nodes_from(G_ba.nodes())
            for u, v in G_ba.edges():
                G.add_edge(u, v)
                G.add_edge(v, u) 
            # Assign random colors to edges
            for u, v in G.edges():
                G[u][v]['color'] = random.randint(0, self.num_colors)
            if nx.is_strongly_connected(G):
                return G
    
    def random_geometric(self):
        """Creates a Radom Geometric graph and then gives it colored bidirectional edges"""
        while True:
            num_nodes = self.num_nodes
            # Each new node attaches to `m` existing nodes
            m = max(3, num_nodes // 5)  
            G_ba = nx.barabasi_albert_graph(num_nodes, m)
            # Convert to a directed graph with bidirectional edges
            G = nx.DiGraph()
            G.add_nodes_from(G_ba.nodes())
            for u, v in G_ba.edges():
                G.add_edge(u, v)
                G.add_edge(v, u)
            # Assign random colors to edges
            for u, v in G.edges():
                G[u][v]['color'] = random.randint(0, self.num_colors)
            if nx.is_strongly_connected(G):
                return G

    
    def Goliath(self):
        """Creates a strongly connected graph with colored edges, good for especially large graph"""
        G = nx.DiGraph()
        # Probability of an edge
        edge_probability = 0.05
        num_nodes = self.num_nodes
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < edge_probability:
                    G.add_edge(i, j)
        while not nx.is_strongly_connected(G):
            components = list(nx.strongly_connected_components(G))
            # If the graph is not strongly connected, connect the components
            for i in range(1, len(components)):
                src_component = components[i - 1]
                dst_component = components[i]
                src_node = random.choice(list(src_component))
                dst_node = random.choice(list(dst_component))
                G.add_edge(src_node, dst_node)
        for u, v in G.edges():
        # Random integer color and weight attributes for each edge
            G[u][v]['color'] = random.randint(0,self.num_colors)
        return G

    def use_premade(self, G, W):
        """Uses a premade graph to overwrite attributes, all other attributes need to be reinitialized"""
        self.G = G
        self.weights = W
        self.num_nodes = len(G.nodes)
        self.subnetworks = self.breadth_first()
        self.radii = self.spectral_radius(self.subnetworks)

    def create_graph(self, num_nodes, num_edges, num_colors):
        """Returns a strongly connected graph with a specified number of nodes, edges, and colored edges"""
        strong = False
        while not strong:
            # Create graph
            G = nx.DiGraph()
            G.add_nodes_from(range(num_nodes))
            # Calculate possible edges-- this does not include self-referential edges
            all_possible_edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
            random.shuffle(all_possible_edges)
            # selects the specified number of edges and adds them to the graph
            edges_to_add = all_possible_edges[:num_edges]
            G.add_edges_from(edges_to_add)
            # Checks if the graph is strongly connected
            strong = nx.is_strongly_connected(G)
        # Assign random colors to edges
        edge_colors = [random.randint(0, num_colors - 1) for _ in range(len(G.edges))]
        edge_color_map = dict(zip(G.edges(), edge_colors))
        nx.set_edge_attributes(G, edge_color_map, 'color')
        return G
    
    def make_custom_graph(self, edges):
        """Older version of use premade which uses inbuilt graphs and weights"""
        #edges = [(0, 2, 1), (0, 1, 1), (1, 3, 0), (2, 0, 1), (2, 3, 0), (3, 2, 0)]
        #edges = [(0, 2, 1),(0, 1, 0),(1, 0, 1),(1, 3, 1),(2,1,1),(3,2,0)]
        #edges = [(0,3,1),(1,2,1),(2,1,1),(2,0,0),(3,0,1),(3,2,1)]
        edges = [(0,1,0),(1,0,1),(1,2,0),(2,1,0)]
        G = nx.DiGraph()
        for u, v, color in edges:
            G.add_edge(u, v, color=color)
        self.G = G
        self.weights = np.array([[0,10,0],[10,0,0.5],[0,0.5,0]])
        self.num_nodes = len(edges)
    
    def create_weights(self, rho=2):
        """"Creates a random weight matrix for a given graph"""
        # limit of the uniform distribution from which the weights are drawn from 
        self.rho = rho
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, j in self.G.edges():
            matrix[i, j] = np.random.uniform(0, rho)
        return matrix


    """Methods to identify subnetworks"""
    
    def breadth_first(self, starting_node=0):
        """Implementation of the BFS algorithm for subnetwork detection """
        completed_subnetworks = []
        working_subnetworks = []
        # [[working levels],[visited nodes],[edge list]]
        working_subnetworks.append([[starting_node], [], []])
        # Loops as long as a subnetwork remains incomplete
        while working_subnetworks:
            # Processes each subnetwork
            for subnetwork in working_subnetworks:
                subnetwork[1] = []
                for entry in subnetwork[2]:
                    node1, node2, color = entry
                    subnetwork[1].append(node1)
                    subnetwork[1].append(node2)
                # Identifies subnetwork and pops it off
                index = working_subnetworks.index(subnetwork)
                working_subnetworks.pop(index)
                # Selects the level to begin searching from and the number of branches
                working_level = subnetwork[0]
                if isinstance(working_level, list):
                    num_branches = len(working_level)
                else:
                    num_branches = 1
                # each branch will be added to branch list, the combinations will be used to create the new subnetworks
                branch_list = []
                edges = {}
                for i in range(num_branches):
                    # this will contain the next level of nodes to be searched for this branch
                    branch = []
                    # Determines which node to evaluate
                    if isinstance(working_level, list):
                        node = working_level[i]
                    else:
                        node = working_level
                    incoming_edges = list(self.G.in_edges(node, data=True))
                    # creates color dictionary
                    color_groups = {}
                    for u, v, data in incoming_edges:
                        color = data.get('color')
                        if color not in color_groups:
                            color_groups[color] = []
                        color_groups[color].append(u)
                    for key in color_groups.keys():
                        colorset = []
                        for value in color_groups[key]:
                            if value in subnetwork[1]:
                                colorset.append((value, node, key))
                            else:
                                colorset.append(value)
                                subnetwork[1].append(value)
                                # add the edge to the edge dict
                                edges[value] = (node, key)
                        branch.append(colorset)
                    branch_list.append(branch)
                cleaned = []
                if len(branch_list) == 1:
                    for i in range(len(branch_list[0])):
                        cleaned.append(branch_list[0][i])
                else:
                    combinations = list(itertools.product(*branch_list))
                    for combo in combinations:
                        combo = self.flatten(combo)
                        cleaned.append(combo)
                for combo in cleaned:
                    new_subnetwork = copy.deepcopy(subnetwork)
                    # new code to add dead edges correctly
                    # make sure to iterate over a copy or the modifications will cause skipped indices
                    for entry in combo[:]:
                        if isinstance(entry, tuple):
                            new_subnetwork[2].append(entry)
                            combo.remove(entry)
                    for node1, (node2, color) in edges.items():
                        if node1 in combo:
                            new_subnetwork[2].append((node1, node2, color))
                    combo = [item for item in combo if isinstance(item, int)]
                    if len(combo) == 0:
                        completed_subnetworks.append(new_subnetwork[2])
                    else:
                        new_subnetwork[0] = combo
                        working_subnetworks.append(new_subnetwork)
        subnetworks = self.remove_dup(completed_subnetworks)
        return subnetworks
    
    def flatten(self, nested_list):
        """Helper function to flatten nested lists"""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self.flatten(item)) 
            else:
                result.append(item) 
        return result
    
    def remove_dup(self,subnetworks):
        """helper function to remove any duplicate subnetworks"""
        seen = set()
        result = []
        for sublist in subnetworks:
            sorted_sublist = tuple(sorted(sublist))
            if sorted_sublist not in seen:
                result.append(sublist) 
                seen.add(sorted_sublist)
        return result
    
  
    

    """Methods to aid in visualization"""

    def visualize_graph(self, G, radius=None):
        """Draws a graph with colored edges"""
        plt.clf()
        colors = [data['color'] for _, _, data in G.edges(data=True)]
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
        edge_colors = [cmap(norm(color)) for color in colors]
        unique_colors = sorted(set(colors))
        patches_list = [patches.Patch(color=cmap(norm(color)), label=f'Color {color}') for color in unique_colors]
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=2, node_size=500, connectionstyle='arc3, rad = 0.05')
        plt.legend(handles=patches_list, title="Edge Colors")
        if radius:
            plt.title(radius)
        plt.show()

    def visualize_subnetworks(self, subnetworks):
        """Calls visualize graph with the radius argument to draw each subnetwork graph"""
        for i, subnetwork in enumerate(subnetworks):
            G = nx.DiGraph()
            for u, v, color in subnetwork:
                G.add_edge(u, v, color=color)
            self.visualize_graph(G, self.radii[i])

    def visualize_behavior(self):
        """Creates a line plot of the flow at each node for each time step"""
        for node, states in self.history.items():
            plt.plot(states, label=f"Node {node}")
        plt.xlabel('Iteration')
        plt.ylabel('State')
        plt.title('Behavior Propagation through Network')
        plt.legend()
        plt.show()

    def visualize_flow(self):
        """Animation of the flow between nodes"""
        fig, ax = plt.subplots(figsize=(7, 7))
        pos = nx.spring_layout(self.G, seed=42, k=0.9, scale=4)  # k adjusts spacing between nodes, scale makes them bigger
        colors = ['black','red', 'orange', 'yellow', 'lightgreen', 'green']
        cmap = LinearSegmentedColormap.from_list("red_orange_yellow_green", colors)
        norm = mcolors.Normalize(vmin=min(min(steps) for steps in self.history.values()),
                                vmax=max(max(steps) for steps in self.history.values()))
        edge_colors = [data['color'] for _, _, data in self.G.edges(data=True)]
        edge_cmap = plt.get_cmap('viridis')
        edge_norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        edge_colors = [edge_cmap(edge_norm(color)) for color in edge_colors]
        nx.draw_networkx_edges(self.G, pos, edge_color=edge_colors, width=0.5, alpha=0.5, ax=ax)
        node_colors = [self.history[node][0] for node in self.G.nodes]
        node_scatter = nx.draw_networkx_nodes(self.G, pos, node_size=20, node_color=node_colors, cmap=cmap, ax=ax)
        cbar = plt.colorbar(node_scatter, ax=ax, orientation='vertical')
        def update_frame(t):
            node_colors = [self.history[node][t] for node in self.G.nodes]
            node_scatter.set_array(np.array(node_colors))  
            return node_scatter,
        ani = animation.FuncAnimation(fig, update_frame, frames=len(next(iter(self.history.values()))),
                                    interval=1000, blit=True)
        plt.title('Node Flow Animation')
        plt.axis('off')
        html = HTML(ani.to_jshtml())
        display(html)

    def visualize_node_flow(self, node, depth):
        """Visualizes the flow around a node to a certain depth"""
        fig, ax = plt.subplots(figsize=(7, 7))
        subgraph_nodes = self.get_nodes_within_depth(node, depth)
        subgraph = self.G.subgraph(subgraph_nodes)
        pos = nx.spring_layout(subgraph, seed=42, k=0.9, scale=4)
        colors = ['black','red', 'orange', 'yellow', 'lightgreen', 'green']
        cmap = LinearSegmentedColormap.from_list("red_orange_yellow_green", colors)
        norm = mcolors.Normalize(vmin=min(min(steps) for steps in self.history.values()),
                                vmax=max(max(steps) for steps in self.history.values()))
        edge_colors = [data['color'] for _, _, data in subgraph.edges(data=True)]
        edge_cmap = plt.get_cmap('viridis')
        edge_norm = mcolors.Normalize(vmin=min(edge_colors), vmax=max(edge_colors))
        edge_colors = [edge_cmap(edge_norm(color)) for color in edge_colors]
        nx.draw_networkx_edges(subgraph, pos, edge_color=edge_colors, width=0.5, alpha=0.5, ax=ax)
        node_colors = [self.history[node][0] for node in subgraph.nodes]
        node_scatter = nx.draw_networkx_nodes(subgraph, pos, node_size=20, node_color=node_colors, cmap=cmap, ax=ax)
        cbar = plt.colorbar(node_scatter, ax=ax, orientation='vertical')

        def update_frame(t):
            """
            Update the node colors for each frame in the animation.
            Only nodes within the specified depth from the central node are updated.
            """
            node_colors = [self.history[node][t] for node in subgraph.nodes]
            node_scatter.set_array(np.array(node_colors))  # Update the node colors for each frame
            return node_scatter,
        ani = animation.FuncAnimation(fig, update_frame, frames=len(next(iter(self.history.values()))),
                                    interval=100, blit=True)
        plt.title(f'Node Flow Animation (Center: Node {node}, Depth: {depth})')
        plt.axis('off')
        html = HTML(ani.to_jshtml())
        display(html)

    def visualize_flow_of_subnetworks(self, method = 'average'):
        """Creates a line plot of the flow in subnetworks, plots using the methods: average, max, and min"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        num_subnetworks = len(self.subnetworks)
        for subnetwork in self.subnetworks:
            nodes = [node for node, _, _ in subnetwork]
            flow = []
            rate_of_change = []
            time_steps = len(next(iter(self.history.values())))
            for t in range(time_steps):
                flows_at_t = []
                for node in nodes:
                    if node in self.history:
                        flows_at_t.append(self.history[node][t])
                if method == 'average':
                    flow_at_t = np.mean(flows_at_t)
                elif method == 'max':
                    flow_at_t = np.max(flows_at_t)
                elif method == 'min':
                    flow_at_t = np.min(flows_at_t)         
                flow.append(flow_at_t)
                if t > 0:
                    rate_of_change_at_t = flow_at_t - flow[t - 1]
                    rate_of_change.append(rate_of_change_at_t)
                else:
                    rate_of_change.append(0)
            ax1.plot(range(time_steps), flow, label=f"Subnetwork {self.subnetworks.index(subnetwork) + 1}")
            ax2.plot(range(1, time_steps), rate_of_change[1:], label=f"Subnetwork {self.subnetworks.index(subnetwork) + 1}")
        self.change = rate_of_change
        self.flow = flow
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel(f'{method.capitalize()} Flow')
        if num_subnetworks <= 5:
            ax1.legend()
        ax1.set_title(f'{method.capitalize()} Flow in Subnetworks Over Time')
        ax2.set_xlabel('Time Steps')
        ax2.set_ylabel('Rate of Change')
        if num_subnetworks <= 5:
            ax2.legend()
        ax2.set_title('Rate of Change of Flow in Subnetworks Over Time')
        plt.tight_layout()
        plt.show()

    def unique_subnetworks_with_radii(self, full_subnetwork, full_radii):
        """Flattens and removes duplicate subnetworks while preserving their spectral radii."""
        unique_subnetworks = []
        unique_radii = []
        seen = set()

        for subnetwork_list, radius_list in zip(full_subnetwork, full_radii):
            for subnetwork, radius in zip(subnetwork_list, radius_list):
                sub_set = frozenset(subnetwork)  # Use frozenset to detect duplicates
                if sub_set not in seen:
                    seen.add(sub_set)
                    unique_subnetworks.append(subnetwork)
                    unique_radii.append(radius)

        return unique_subnetworks, unique_radii

    def visualize_subnetwork_graph(self):
        """Creates a graph where each subnetwork is a node, and edges represent shared nodes.
        Nodes are outlined in red if spectral radius < 1, green otherwise."""
        # Flatten and remove duplicates while keeping spectral radii aligned
        unique_subnetworks, unique_radii = self.unique_subnetworks_with_radii(self.full_subnetwork, self.full_radii)
        G = nx.Graph()
        # Convert each subnetwork into a set of nodes for comparison
        subnetwork_nodes = [set(node for node, _, _ in sub) for sub in unique_subnetworks]
        # Add nodes (subnetworks)
        for i in range(len(subnetwork_nodes)):
            G.add_node(i, size=len(subnetwork_nodes[i]))
        # Add edges if subnetworks share nodes
        edge_colors = []
        for i in range(len(subnetwork_nodes)):
            for j in range(i + 1, len(subnetwork_nodes)):
                shared_nodes = subnetwork_nodes[i] & subnetwork_nodes[j]  # Nodes in both subnetworks
                if shared_nodes:
                    # Check if any shared node is in zero_nodes
                    if any(node in self.zero_nodes for node in shared_nodes):
                        edge_color = "red"  # Critical shared node
                    else:
                        edge_color = "green"  # Normal shared node
                    
                    G.add_edge(i, j, weight=len(shared_nodes), color=edge_color)
                    edge_colors.append(edge_color)
        # Node border colors based on spectral radius
        node_edge_colors = ["red" if radius < 1 else "green" for radius in unique_radii]
        # Visualization
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G, seed=42) 
        sizes = [G.nodes[n]['size'] * 100 for n in G.nodes]  # Scale node sizes
        nx.draw_networkx_edges(G, pos, width=2, edge_color=edge_colors)
        nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color="white",
                            edgecolors=node_edge_colors, linewidths=1.5)
        nx.draw_networkx_labels(G, pos)
        plt.title("Subnetwork Graph with Critical Shared Nodes Highlighted")
        plt.show()

    def visualize_subnetwork_intersection(self):
        """Creates subplots for every pair of subnetworks showing both their intersection and symmetric difference."""
        for i in range(len(self.subnetworks)):
            for j in range(i + 1, len(self.subnetworks)):
                sub_i = set(self.subnetworks[i])
                sub_j = set(self.subnetworks[j])
                intersection = sub_i & sub_j  # Common edges    
                sym_diff = sub_i ^ sub_j  # Symmetric difference (union minus intersection)
                if intersection or sym_diff:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))    
                    for ax, edges, title in zip(axes, [intersection, sym_diff], 
                                                [f"Intersection of Subnetworks {i+1} and {j+1}", 
                                                f"Symmetric Difference of Subnetworks {i+1} and {j+1}"]):
                        G = nx.DiGraph()
                        G.add_edges_from([(u, v, {'color': color}) for u, v, color in edges])
                        pos = nx.spring_layout(G, seed=42)      
                        # Extract edge colors
                        if G.number_of_edges() > 0:
                            colors = [data['color'] for _, _, data in G.edges(data=True)]
                            cmap = plt.get_cmap('viridis')
                            norm = mcolors.Normalize(vmin=min(colors), vmax=max(colors))
                            edge_colors = [cmap(norm(color)) for color in colors]
                        else:
                            edge_colors = []
                            colors = []
                        nx.draw(G, pos, with_labels=True, edge_color=edge_colors, width=2, node_size=500, ax=ax, connectionstyle='arc3, rad = 0.05')
                        # Add legend
                        if colors:
                            unique_colors = sorted(set(colors))
                            patches_list = [patches.Patch(color=cmap(norm(color)), label=f'Color {color}') for color in unique_colors]
                            ax.legend(handles=patches_list, title="Edge Colors")
                        ax.set_title(title)
                    plt.tight_layout()
                    plt.show()

    def visualize_subnetwork_venn(self):
        """Creates a Venn-like diagram where subnetworks are circles, sized by subnetwork size, and overlap based on shared nodes."""
        subnetworks = self.subnetworks
        sub_sizes = [len(sub) for sub in subnetworks]
        sub_sets = [set(sub) for sub in subnetworks]
        num_subs = len(subnetworks)
        # Compute pairwise overlaps
        overlaps = np.zeros((num_subs, num_subs))
        for i in range(num_subs):
            for j in range(i + 1, num_subs):
                shared = len(sub_sets[i] & sub_sets[j])
                overlaps[i, j] = overlaps[j, i] = shared  
        # Scale sizes for visualization
        max_size = max(sub_sizes)
        radii = [np.sqrt(size / max_size) * 2 for size in sub_sizes]  # Normalize radii
        # Initialize positions
        positions = np.random.rand(num_subs, 2) * 10  # Spread randomly in space
        # Optimize placement based on overlaps
        for _ in range(500):  # Adjust positions iteratively
            for i in range(num_subs):
                for j in range(i + 1, num_subs):
                    if overlaps[i, j] > 0:  # If they share elements, bring closer
                        direction = positions[j] - positions[i]
                        distance = np.linalg.norm(direction)
                        desired_distance = (radii[i] + radii[j]) * (1 - overlaps[i, j] / max_size)
                        if distance > 0 and distance > desired_distance:
                            move_vector = direction / distance * (distance - desired_distance) * 0.05
                            positions[i] += move_vector
                            positions[j] -= move_vector
        # Plot circles
        cmap = cm.get_cmap("tab10", num_subs)
        colors = [mcolors.to_rgba(cmap(i)) for i in range(num_subs)]
        fig, ax = plt.subplots(figsize=(8, 8))
        texts = []
        for i in range(num_subs):
            circle = plt.Circle(positions[i], radii[i], alpha=0.4, color=colors[i], label=f'Sub {i+1}')
            ax.add_patch(circle)
            #texts.append(ax.text(*positions[i], f'Sub {i+1}', ha='center', va='center')) 
        adjust_text(texts)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        plt.title('Subnetwork Visualization')
        plt.legend()
        plt.show()
    
    def visualize_node_histogram(self):
        """Creates a histogram showing the number of subnetworks each node appears in."""
        node_counts = defaultdict(int)
        for sub in self.subnetworks:
            for node1, node2, _ in sub:
                node_counts[node1] += 1
                node_counts[node2] +=1 
        plt.figure(figsize=(8, 6))
        plt.bar(node_counts.keys(), node_counts.values(), edgecolor='black')
        plt.xlabel('Node')
        plt.ylabel('Total Appearances')
        plt.title('Node Appearances in Subnetworks')
        plt.xticks(sorted(node_counts.keys()))
        plt.show()


    def ghost_edges(self):
        """Visualizes the graph where dead subnetworks are ghosted out and barrier nodes are marked in green"""
        ghost = self.dead_edges() 
        G = self.G
        # Extract colors from edges
        colors = { (u, v): data['color'] for u, v, data in G.edges(data=True) }
        cmap = plt.get_cmap('viridis')
        norm = mcolors.Normalize(vmin=min(colors.values()), vmax=max(colors.values()))
        # Convert ghost edges into a set for quick lookup
        ghost_edges = {(u, v) for u, v, _ in ghost}
        # Identify nodes that are the origin of ghost edges
        ghost_origin_nodes = {u for u, v, _ in ghost}
        ghost_origin_nodes.update({v for u,v, _ in ghost})
        # Add all edges originating from ghost-origin nodes to the ghost edges set
        for u in ghost_origin_nodes:
            for v in G.neighbors(u):
                ghost_edges.add((u, v))  # Mark these as ghost edges
        # Separate normal and ghost edges
        normal_edges = [(u, v) for u, v in G.edges if (u, v) not in ghost_edges and (v, u) not in ghost_edges]
        ghost_edges = [(u, v) for u, v in ghost_edges if (u, v) in G.edges or (v, u) in G.edges]
        # Identify nodes with excessive same-color incoming edges
        incoming_edges_by_color = {}  # {node: {color: count}}
        for u, v in G.edges:
            color = colors[(u, v)]
            if v not in incoming_edges_by_color:
                incoming_edges_by_color[v] = {}
            if color not in incoming_edges_by_color[v]:
                incoming_edges_by_color[v][color] = 0
            incoming_edges_by_color[v][color] += 1
        # Nodes that should be bordered in green
        green_border_nodes = {node for node, color_counts in incoming_edges_by_color.items() if any(count >= 2 for count in color_counts.values())}
        pos = nx.spring_layout(G, seed=42)
        # Draw normal edges first (excluding ghost edges)
        normal_edge_colors = [cmap(norm(colors[edge])) for edge in normal_edges]
        nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color=normal_edge_colors, width=2, connectionstyle='arc3,rad=0.05')
        # Draw ghost edges as dotted lines
        ghost_edge_colors = [cmap(norm(colors[edge])) for edge in ghost_edges]
        nx.draw_networkx_edges(G, pos, edgelist=ghost_edges, edge_color=ghost_edge_colors, width=2, alpha=0.5, style='dotted', connectionstyle='arc3,rad=0.05')
        # Draw normal nodes
        normal_nodes = [node for node in G.nodes if node not in ghost_origin_nodes and node not in green_border_nodes]
        nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=250)
        # Draw ghost-origin nodes with dashed outline and higher alpha
        nx.draw_networkx_nodes(G, pos, nodelist=ghost_origin_nodes, node_size=250, alpha=0.5)
        green = [node for node in green_border_nodes if node not in ghost_origin_nodes]
        # Draw green-bordered nodes
        nx.draw_networkx_nodes(G, pos, nodelist=green, node_size=250, node_color='green', linewidths=2)
        extraspecial = [node for node in ghost_origin_nodes and green_border_nodes]
        nx.draw_networkx_nodes(G, pos, extraspecial, node_size=250, node_color='green', linewidths=2, alpha=0.5)
        # Draw labels
        nx.draw_networkx_labels(G, pos)
        # Legend for edge colors
        unique_colors = sorted(set(colors.values()))
        patches_list = [patches.Patch(color=cmap(norm(color)), label=f'Color {color}') for color in unique_colors]
        plt.legend(handles=patches_list, title="Edge Colors")
        plt.show()




    """Calculation methods for prediction"""

    def spectral_radius(self, subnetworks):
        """Calculates the spectral radii for each subnetwork in a list of subnetworks"""
        num = self.num_nodes
        weights = self.weights
        radii = []
        for subnetwork in subnetworks:
            matrix = np.zeros((num,num))
            for node1, node2, _ in subnetwork:
                matrix[node1][node2] = weights[node1][node2]
            rho = max(abs(np.linalg.eigvals(matrix)))
            radii.append(rho)
        return radii

    def simulate_linear(self, initial=None, maxiter=1000, epsilon=1e-10):
        """Simulates the behavior of a system, implements a tanh to limit behavior between 0 and 1"""
        G = self.G
        weights = self.weights
        self.T = False
        if initial == None:
            initial = {node: random.uniform(1, 2) for node in G.nodes}
        self.initial=initial
        history = {node: [initial[node]] for node in G.nodes}
        current = initial.copy()
        for i in range(maxiter):
            next = {}
            max_change = 0
            for node in G.nodes:
                incoming_edges = list(G.in_edges(node, data=True))
                color_groups = {}
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color is not None:
                        edge_weight = weights[u][v]
                        state_of_neighbor = current[u]
                        if color not in color_groups:
                            color_groups[color] = 0
                        color_groups[color] += edge_weight * state_of_neighbor
                min_next_state = float('inf')
                for color, total_sum in color_groups.items():
                    if total_sum > 0:
                        min_next_state = min(min_next_state, total_sum)
                if min_next_state == float('inf'):
                    next[node] = 0
                else:
                    next[node] = np.tanh(min_next_state)
                max_change = max(max_change, abs(next[node] - current[node]))
            for node, state in next.items():
                history[node].append(state)
            current = next
            if max_change < epsilon:
                self.T = True
                break
        converged = [current[node] for node in G.nodes]
        zero_nodes = set()
        for i in range(len(converged)):
            if np.abs(converged[i]) <= 1e-2:
                zero_nodes.add(i)
        return history, converged, zero_nodes
    
    def predict(self,radii,subnetworks):
        """Predicts the behavior of a subnetwork based on spectral radius"""
        predicted = set()
        for i in range(len(radii)):
            if radii[i] < 1:
                target = subnetworks[i]
                for node1,node2,_ in target:
                    predicted.add(node1)
                    predicted.add(node2)
        return predicted
    
    def full_subnetworks(self):
        """Calculates the full subnetworks by looping BFS"""
        full_subnetworks = []
        for i in range(self.num_nodes):
            subnetworks = self.breadth_first(i)
            full_subnetworks.append(subnetworks)
        
        return full_subnetworks
    
    def predict_full(self, s):
        """Predicts the behavior of full subnetworks based on spectral radius"""
        full_predicted = set()
        full_radii = []
        for subnetworks in s:
            radii = self.spectral_radius(subnetworks)
            full_radii.append(radii)
            nodes = self.predict(radii, subnetworks)
            for node in nodes:
                full_predicted.add(node)
        return full_predicted, full_radii
    

    """Methods for perturbing weights and calculating condition number"""

    def perturb_weights(self, prange=(-0.5,0.5)):
        """Creates a new Graphcalss instance with perturbed weights """
        new = Graphclass(4,6,1,self.method,self.full)
        new.use_premade(self.G, self.weights)
        p = self.weights.copy()
        perturbation = np.random.uniform(prange[0], prange[1], p.shape)
        p[p != 0] += perturbation[p != 0]
        new.p = p
        new.weights = p
        new.initial = self.initial.copy()
        new.subnetworks = self.subnetworks
        new.radii = new.spectral_radius(new.subnetworks)
        new.history, new.converged, new.zero_nodes = new.simulate_linear(self.initial)
        new.predicted = new.predict(new.radii, new.subnetworks)
        if self.full:
            new.full_predicted, new.full_radii = new.predict_full(new.full_subnetwork)
        return new
    
    


    def perturb_initial(self, prange=(-1e-5,1e-5)):
        """Creates a new Graphcalss instance with perturbed initial conditions """
        new = Graphclass(4,6,1,self.method,self.full)
        new.use_premade(self.G, self.weights)
        p = self.initial.copy()
        for i, value in p.items():
            perturbation = np.random.uniform(prange[0], prange[1])
            p[i] += perturbation    
        new.initial = p 
        new.subnetworks = self.subnetworks
        new.radii = new.spectral_radius(new.subnetworks)
        new.history, new.converged, new.zero_nodes = new.simulate_linear(new.initial)
        new.predicted = new.predict(new.radii, new.subnetworks)
        if self.full:
            new.full_predicted, new.full_radii = new.predict_full(new.full_subnetwork)
        return new
    
    def initial_lyapunov(self, initial=None, maxiter=100, epsilon=1e-10,perturb=1e-5):
        """Calculates the lyapunov coefficients based on initial conditions"""
        if initial is None:
            initial = {node: random.uniform(1, 2) for node in self.G.nodes}
        self.initial = initial
        # Initialize the unperturbed and perturbed states
        current = initial.copy()
        perturbed = {node: current[node] + random.uniform(-perturb, perturb) for node in self.G.nodes} 
        history_current = {node: [current[node]] for node in self.G.nodes}
        history_perturbed = {node: [perturbed[node]] for node in self.G.nodes}
        max_change = 0
        for i in range(maxiter):
            next_current = {}
            next_perturbed = {}
            for node in self.G.nodes:
                incoming_edges = list(self.G.in_edges(node, data=True))
                color_groups_current = {}
                color_groups_perturbed = {}
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color is not None:
                        edge_weight = self.weights[u][v]
                        state_of_neighbor_current = current[u]
                        state_of_neighbor_perturbed = perturbed[u]
                        if color not in color_groups_current:
                            color_groups_current[color] = 0
                        color_groups_current[color] += edge_weight * state_of_neighbor_current
                        if color not in color_groups_perturbed:
                            color_groups_perturbed[color] = 0
                        color_groups_perturbed[color] += edge_weight * state_of_neighbor_perturbed
                min_next_state_current = float('inf')
                min_next_state_perturbed = float('inf')
                for color, total_sum in color_groups_current.items():
                    if total_sum > 0:
                        min_next_state_current = min(min_next_state_current, total_sum)
                for color, total_sum in color_groups_perturbed.items():
                    if total_sum > 0:
                        min_next_state_perturbed = min(min_next_state_perturbed, total_sum)
                next_current[node] = 0 if min_next_state_current == float('inf') else np.tanh(min_next_state_current)
                next_perturbed[node] = 0 if min_next_state_perturbed == float('inf') else np.tanh(min_next_state_perturbed)
            # Calculate the difference between the perturbed and unperturbed states
            perturbation_norm = np.linalg.norm([next_perturbed[node] - next_current[node] for node in self.G.nodes])
            perturbation_growth = np.log(perturbation_norm) if perturbation_norm > 0 else 0
            # Track the states and perturbation growth over time
            for node, state in next_current.items():
                history_current[node].append(state)
            for node, state in next_perturbed.items():
                history_perturbed[node].append(state)
            # Update the system states
            current = next_current
            perturbed = next_perturbed
            # If max_change is smaller than epsilon, break
            max_change = max(abs(next_current[node] - current[node]) for node in self.G.nodes)
            if max_change < epsilon:
                break
        # Compute the Lyapunov exponent
        total_lyapunov = 0
        for i in range(1, len(history_current[next(iter(self.G.nodes))])):
            growth = np.log(np.linalg.norm([history_perturbed[node][i] - history_current[node][i] for node in self.G.nodes]) /
                        np.linalg.norm([history_perturbed[node][0] - history_current[node][0] for node in self.G.nodes]))
            total_lyapunov += growth
        lyapunov_exponent = total_lyapunov / len(history_current[next(iter(self.G.nodes))])
        return lyapunov_exponent
    
    def weight_lyapunov(self, initial=None, maxiter=100, epsilon=1e-10, perturb=1e-5):
        """Calculates the lyapunov coefficients based on weights"""
        if initial is None:
            initial = {node: random.uniform(1, 2) for node in self.G.nodes}
        self.initial = initial
        # Initialize the unperturbed and perturbed states
        current = initial.copy()
        perturbed = current.copy()  # Initially same as current
        history_current = {node: [current[node]] for node in self.G.nodes}
        history_perturbed = {node: [perturbed[node]] for node in self.G.nodes}
        # Initialize edge weights (perturbed edge weights are applied later)
        perturbed_weights = {edge: self.weights[edge] + random.uniform(-perturb, perturb) 
                            if self.weights[edge] != 0 else self.weights[edge] 
                            for edge in self.G.edges}
        max_change = 0
        for i in range(maxiter):
            next_current = {}
            next_perturbed = {}
            for node in self.G.nodes:
                incoming_edges = list(self.G.in_edges(node, data=True))
                color_groups_current = {}
                color_groups_perturbed = {}
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color is not None:
                        # Use perturbed edge weights for the perturbed state
                        edge_weight_current = self.weights[u, v]
                        edge_weight_perturbed = perturbed_weights.get((u, v), self.weights[u, v])
                        
                        state_of_neighbor_current = current[u]
                        state_of_neighbor_perturbed = perturbed[u]
                        
                        if color not in color_groups_current:
                            color_groups_current[color] = 0
                        color_groups_current[color] += edge_weight_current * state_of_neighbor_current
                        
                        if color not in color_groups_perturbed:
                            color_groups_perturbed[color] = 0
                        color_groups_perturbed[color] += edge_weight_perturbed * state_of_neighbor_perturbed
                min_next_state_current = float('inf')
                min_next_state_perturbed = float('inf')
                for color, total_sum in color_groups_current.items():
                    if total_sum > 0:
                        min_next_state_current = min(min_next_state_current, total_sum)
                for color, total_sum in color_groups_perturbed.items():
                    if total_sum > 0:
                        min_next_state_perturbed = min(min_next_state_perturbed, total_sum)
                next_current[node] = 0 if min_next_state_current == float('inf') else np.tanh(min_next_state_current)
                next_perturbed[node] = 0 if min_next_state_perturbed == float('inf') else np.tanh(min_next_state_perturbed)
            # Calculate the difference between the perturbed and unperturbed states
            perturbation_norm = np.linalg.norm([next_perturbed[node] - next_current[node] for node in self.G.nodes])
            perturbation_growth = np.log(perturbation_norm) if perturbation_norm > 0 else 0
            # Track the states and perturbation growth over time
            for node, state in next_current.items():
                history_current[node].append(state)
            for node, state in next_perturbed.items():
                history_perturbed[node].append(state)
            # Update the system states
            current = next_current
            perturbed = next_perturbed
            # If max_change is smaller than epsilon, break
            max_change = max(abs(next_current[node] - current[node]) for node in self.G.nodes)
            if max_change < epsilon:
                break
        # Compute the Lyapunov exponent
        total_lyapunov = 0
        for i in range(1, len(history_current[next(iter(self.G.nodes))])):
            growth = np.log(np.linalg.norm([history_perturbed[node][i] - history_current[node][i] for node in self.G.nodes]) /
                            np.linalg.norm([history_perturbed[node][0] - history_current[node][0] for node in self.G.nodes]))
            total_lyapunov += growth
        lyapunov_exponent = total_lyapunov / len(history_current[next(iter(self.G.nodes))])
        return lyapunov_exponent
    
    def sensitivity_analysis(self, epsilon=1e-6):
        """Perturbs weight matrix and performs a sensitivity analysis"""
        eigenvalues_original = np.linalg.eigvals(self.weights)
        lambda_max_original = np.max(np.real(eigenvalues_original))
        sensitivity_matrix = np.zeros_like(self.weights)
        # Perturb each element of the matrix and calculate the new largest eigenvalue
        for i in range(self.weights.shape[0]):
            for j in range(self.weights.shape[1]):
                if self.weights[i,j] != 0:
                # Perturb element A[i, j] by epsilon
                    A_perturbed = self.weights.copy()
                    A_perturbed[i, j] += epsilon
                    # Compute the largest eigenvalue of the perturbed matrix
                    eigenvalues_perturbed = np.linalg.eigvals(A_perturbed)
                    lambda_max_perturbed = np.max(np.real(eigenvalues_perturbed))
                    # Compute the change in the largest eigenvalue (sensitivity)
                    sensitivity_matrix[i, j] = np.abs(lambda_max_perturbed - lambda_max_original) / epsilon
        return sensitivity_matrix

   
    """Hueristic methods for prediction"""

    def swhueristic_predict(self,iter):
        """Hueristic method which selects the smallest weighted edges to follow"""
        start = time.time()
        zero_nodes = set()
        q = 0
        targets = np.argsort(np.ma.masked_equal(self.weights, 0).min(axis=0).filled(self.rho+1)).tolist()
        while targets and q <= iter:
            target = targets.pop(0)
            subnetwork = self.swDFS(target)
            radius = self.spectral_radius([subnetwork])
            if radius[0] < 1:
                for node1, node2, _ in subnetwork:
                    zero_nodes.add(node1)
                    zero_nodes.add(node2)
                    try:
                        targets.remove(node1)
                    except ValueError:
                        pass
                    try:
                        targets.remove(node2)
                    except ValueError:
                        pass
            q+=1
        self.swhpredicted = zero_nodes
        stop = time.time()
        self.swhflop = (stop-start)
    
    def swDFS(self,target):
        """DFS method which follows the smallest edge weight"""
        subnetwork = []
        seen = set()
        seen.add(target)
        nodes = [target]
        while nodes:
            new_nodes = []
            for node in nodes:
                incoming_edges = self.G.in_edges(node, data=True)
                color_groups = {} 
                color_weights = {}    
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color not in color_groups:
                        color_groups[color] = []
                        color_weights[color] = 0
                    color_groups[color].append((u, v))
                    weight = self.weights[u, v]
                    color_weights[color] += weight
                min_color = min(color_weights, key=color_weights.get)
                for (u, v) in color_groups[min_color]:
                    subnetwork.append((u, v, min_color))
                    if u not in seen:
                        new_nodes.append(u)
            seen.update(new_nodes)
            nodes = new_nodes
        return subnetwork
    
    def greedyhueristic_predict(self,iter):
        """Hueristic which seeks to create the largest subnetworks by collecting nodes with the most edges of the same color"""
        start = time.time()
        zero_nodes = set()
        q = 0
        targets = sorted(self.G.nodes, key=lambda node: self.G.in_degree(node), reverse=True)
        while targets and q <= iter:
            target = targets.pop(0)
            subnetwork = self.greedyDFS(target)
            radius = self.spectral_radius([subnetwork])
            if radius[0] < 1:
                for node1, node2, _ in subnetwork:
                    zero_nodes.add(node1)
                    zero_nodes.add(node2)
                    try:
                        targets.remove(node1)
                    except ValueError:
                        pass
                    try:
                        targets.remove(node2)
                    except ValueError:
                        pass
            q+=1
        self.greedypredicted = zero_nodes
        stop = time.time()
        self.greedyflop = (stop-start)
    
    def greedyDFS(self,target):
        """DFS which follows paths with the most edges of the same color"""
        subnetwork = []
        seen = set()
        seen.add(target)
        nodes = [target]
        while nodes:
            new_nodes = []
            for node in nodes:
                incoming_edges = self.G.in_edges(node, data=True)
                color_groups = {} 
                color_weights = {}    
                for u, v, data in incoming_edges:
                    color = data.get('color')
                    if color not in color_groups:
                        color_groups[color] = []
                        color_weights[color] = 0
                    color_groups[color].append((u, v))
                    weight = self.weights[u, v]
                    color_weights[color] += weight
                max_color = max(color_groups, key=lambda color: len(color_groups[color]))
                for (u, v) in color_groups[max_color]:
                    subnetwork.append((u, v, max_color))
                    if u not in seen:
                        new_nodes.append(u)
            seen.update(new_nodes)
            nodes = new_nodes
        return subnetwork
    



    """Other methods"""

    def count_subnetworks(self,subnetworks):
        """counts number of subnetworks"""
        num = 0
        for sub in subnetworks:
            num += 1
        self.num_subnetworks = num

    def are_subnetworks_identical(self, subnetworks):
        """Check if all subnetworks in the list are identical, ignoring order."""
        example = subnetworks[0]
        correctlen = len(example)
        for network in subnetworks:
            if len(network) != correctlen:
                self.identical =  False
                return
        seen = set()
        for sublist in example:
            sorted_sublist = tuple(sorted(sublist))
            seen.add(sorted_sublist)
        for network in subnetworks[1:]:
            for sublist in network:
                s = tuple(sorted(sublist))
                if s not in seen:
                    self.identical = False
                    return
        self.identical = True

    def get_nodes_within_depth(self, node, depth):
        """
        Returns the list of nodes within the specified depth from the given node.
        This uses a breadth-first search (BFS) to find the nodes within the specified distance.
        """
        visited = set()
        to_visit = [(node, 0)]  # (current_node, current_depth)
        visited.add(node)
        nodes_within_depth = [node]
        while to_visit:
            current_node, current_depth = to_visit.pop(0)
            if current_depth < depth:
                # Add neighbors to the list if they haven't been visited
                for neighbor in self.G.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        nodes_within_depth.append(neighbor)
                        to_visit.append((neighbor, current_depth + 1))
        return nodes_within_depth
    def dead_edges(self):
        """Identifies edges in dying subnetworks"""
        dead = []
        for i in range(len(self.full_subnetwork)):
            for j in range(len(self.full_subnetwork[i])):
                if self.full_radii[i][j] < 1:
                    for q in range(len(self.full_subnetwork[i][j])):
                        dead.append(self.full_subnetwork[i][j][q])
        return dead
