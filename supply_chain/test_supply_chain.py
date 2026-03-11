import random
import networkx as nx
from network import SupplyChainNetwork

def test_subnetworks():
    for nodes in range(3, 9):
        colors = random.randint(2, 6)
        edges = random.randint(nodes, colors * nodes * (nodes - 1))
        sc = SupplyChainNetwork(nodes, edges, colors)

        for subnetwork in sc.subnetworks:
            assert nx.is_weakly_connected(subnetwork)
            for v in subnetwork.nodes:
                colors = {data["color"] for _, _, data in subnetwork.in_edges(v, data=True)}
                assert len(colors) == 1
        