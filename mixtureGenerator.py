import networkx as nx
import numpy as np
import random


class Mixture:
    """
    class for generating mixed graphs
    """

    def __init__(self):
        self.topology = ('small_world', 'scale_free', 'regular', 'random')
        self.max_k = 5
        self.chose = [i for i in range(1, 101)]

    def __call__(self, n: int, nodes: int, file: str, **kwargs: dict):
        """

        :param n: Num of graph
        :param nodes: Num of nodes
        :param file: Path for saving
        :param kwargs: Names of topology and its percentage in the graph
        :return: None
        """
        assert nodes < 10, 'There are too few vertices'
        assert n < 1, 'The number of graphs must be greater than zero'
        assert all(key in kwargs.keys() for key in self.topology), f'Available topologies: {self.topology}'

        with open(file, 'wb') as f:
            for _ in range(n):
                g = nx.Graph()
                g.add_nodes_from(range(nodes))
                vertices = [i for i in range(nodes)]  # list of vertices indexes
                random.shuffle(vertices)
                len_vertices = len(vertices)
                len_kwargs = len(kwargs.items())
                last = 0
                dict_gen = {}  # key - name of the function for generating the topology; value -  set of indexes
                for i, (k, v) in enumerate(kwargs.items()):
                    if int(i) < len_kwargs - 1:
                        now = len_vertices * v
                        dict_gen[k] = {j for j in vertices[last:now]}
                        last = now
                    else:
                        dict_gen[k] = {j for j in vertices[last:len_kwargs]}
                vertices = [i for i in range(nodes)]
                for node_index in vertices:
                    for (k, v) in dict_gen.items():
                        if node_index in v:
                            g = eval('self.' + k + f'({g}, {node_index}, {nodes})')  # generating edges for the node
                    continue
                np.save(f, nx.edges(g))  # сохраняем numpy массив пары вершин
                np.save(f, np.array([0]))

    def small_world(self, g: nx.Graph, node_index, nodes) -> nx.Graph:
        k = min(self.max_k, random.randint(2, self.max_k))
        p = random.randint(1, 100)
        if nodes - node_index - k >= 0:
            g.add_edges_from([(node_index, node_index + i) for i in range(1, k + 1)])
        else:
            g.add_edges_from([(node_index, node_index + i) for i in range(1, -(nodes - node_index - k))])
            g.add_edges_from([(node_index, i) for i in range(0, nodes - node_index)])
        if p < random.choice(self.chose):
            n = [i for i in g.nodes if i not in g.neighbors(node_index)]
            n.remove(node_index)
            g.add_edge(node_index, random.choice(n))

        return g

    def scale_free(self, g: nx.Graph, node_index, nodes) -> nx.Graph:
        degrees = [val for (node, val) in g.degree()]
        k = min(self.max_k, random.randint(2, self.max_k))
        neighbors = random.choices(range(nodes), weights=degrees, k=k)
        for neighbor in neighbors:
            g.add_edge(node_index, neighbor)
        return g

    @staticmethod
    def random(self, g: nx.Graph, node_index, nodes) -> nx.Graph:
        n = [i for i in g.nodes if i not in g.neighbors(node_index)]
        n.remove(node_index)
        g.add_edge(node_index, random.choice(n))
        return g

    def regular(self, g: nx.Graph, node_index, nodes) -> nx.Graph:
        k = min(self.max_k, random.randint(2, self.max_k))
        if nodes - node_index - k >= 0:
            g.add_edges_from([(node_index, node_index + i) for i in range(1, k + 1)])
        else:
            g.add_edges_from([(node_index, node_index + i) for i in range(1, -(nodes - node_index - k))])
            g.add_edges_from([(node_index, i) for i in range(0, nodes - node_index)])
        return g
