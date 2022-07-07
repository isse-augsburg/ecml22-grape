import copy
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Set, Tuple, Union

from preprocessing.node import Node
from preprocessing.component import Component


class Graph:
    """
    Graph object composed of nodes and edges between the nodes.
    Currently, we only represent *undirected* graphs, i.e. graphs with undirected (bidirectional) edges.
    """

    def __init__(self, construction_id: int = None):
        self.__construction_id: int = construction_id  # represents unix timestamp of creation date
        self.__nodes: Set[Node] = set()
        self.__edges: Dict[Node, List[Node]] = {}
        self.__node_counter: int = 0  # internal node id counter
        self.__is_connected: bool = None  # determines if the graph is connected
        self.__contains_cycle: bool = None   # determines if the graph contains non-trivial cycles
        self.__is_bidirectional: bool = None  # determines if all edges are bidirectional

    def __eq__(self, other) -> bool:
        """ Specifies equality of two graph instances. Needed for software tests. """
        if other is None:
            return False
        if not isinstance(other, Graph):
            raise TypeError(f'Can not compare different types ({str(type(self))} and {str(type(other))})')

        # Equality is defined based on the *components*, the node id is not relevant
        # compare number of nodes
        self_nodes_sorted = sorted(self.get_edges().keys())
        other_nodes_sorted = sorted(other.get_edges().keys())
        if len(self_nodes_sorted) != len(other_nodes_sorted):
            return False
        for node_idx in range(len(self_nodes_sorted)):
            # check that components of sorted nodes in both instances are equivalent
            self_node = self_nodes_sorted[node_idx]
            other_node = other_nodes_sorted[node_idx]
            if not self_node.get_component().equivalent(other_node.get_component()):
                return False
            # check that edges are equivalent
            self_node_neighbors = self.get_edges()[self_node]
            other_node_neighbors = other.get_edges()[other_node]
            if len(self_node_neighbors) != len(other_node_neighbors):
                return False
            for neighbor_idx in range(len(self_node_neighbors)):
                if not self_node_neighbors[neighbor_idx].get_component() \
                        .equivalent(other_node_neighbors[neighbor_idx].get_component()):
                    return False
        return True

    def __hash__(self) -> int:
        """ Defines hash of a graph. """
        hash_value = hash("Graph")
        for k, v in self.get_edges().items():
            pair = (k, tuple(v))
            hash_value = hash(hash_value + hash(pair))
        return hash_value

    def __get_edge_pairs(self):
        """
        Returns a list of all edge tuples (sink, source) of the graph, where the entries are components instead of nodes.
        :return: List of edge tuples
        """
        return [(k.get_component(), value_elem.get_component()) for k, value in self.get_edges().items() for value_elem in value]

    def edge_subsumption(self, graph_list: List['Graph']) -> bool:
        """
        Indicates whether a given list of graphs unified represent the current graph.
        :param graph_list:
        :return: boolean that indicates if the given graphs subsume the current graph
        """
        self_edge_tuples = sorted(self.__get_edge_pairs())
        edge_tuples = sorted([pair for g in graph_list for pair in g.__get_edge_pairs()])

        for t in self_edge_tuples:
            found = None
            for ot in edge_tuples:
                if ot[0].equivalent(t[0]) and ot[1].equivalent(t[1]):
                    found = ot
                    break
            if not found:
                return False
            edge_tuples.remove(found)
        return True

    def is_subgraph(self, g: 'Graph') -> bool:
        """
        Determines if the given graph is a subgraph of the current graph. This method does not check for true subset
        relationship, i.e. a graph is always treated as a subgraph of itself.
        :param g: graph to check for subgraph relationship to the current graph
        :return: boolean indicating if the given graph is subgraph of the current graph
        """
        self_nodes_sorted = sorted(self.get_edges().keys())
        other_nodes_sorted = sorted(g.get_edges().keys())
        if len(self_nodes_sorted) < len(other_nodes_sorted):
            return False

        for other_node in other_nodes_sorted:
            has_equivalent = False
            for candidate in filter(lambda n: n.get_component().equivalent(other_node.get_component()), self_nodes_sorted):
                self_node_neighbors = self.get_edges()[candidate]
                other_node_neighbors = g.get_edges()[other_node]
                if len(self_node_neighbors) < len(other_node_neighbors):
                    break
                has_neighbors = True
                for neighbor in other_node_neighbors:
                    has_neighbors &= any([n.get_component().equivalent(neighbor.get_component()) for n in self_node_neighbors])
                has_equivalent |= has_neighbors
                if has_neighbors:
                    break
            if not has_equivalent:
                return False
        return True

    def __get_node_for_component(self, component: Component) -> Node:
        """
        Returns a node of the graph for the given component. If the component is already known in the graph, the
        corresponding node is returned, else a new node is created.
        :param component: component
        :return: corresponding node for the given component
        """
        if component not in [node.get_component() for node in self.get_nodes()]:
            # create new node for component
            node = Node(self.__node_counter, component)
            self.__node_counter += 1
        else:
            node = [node for node in self.get_nodes() if node.get_component() is component][0]

        return node

    def __add_node(self, node):
        """ Adds a node to the internal set of nodes. """
        self.__nodes.add(node)

    def add_undirected_edge(self, component1: Component, component2: Component):
        """
        Adds an undirected edge between component1 and component2. Therefor, the components are transformed to nodes.
        This is equivalent to adding two directed edges, one from component1 to component2 and the second from
        component2 to component1.
        :param component1: one of the components for the undirected edge
        :param component2: second component for the undirected edge
        """
        self.__add_edge(component1, component2)
        self.__add_edge(component2, component1)

    def __add_edge(self, source: Component, sink: Component):
        """
        Adds an directed edge from source to sink. Therefor, the components are transformed to nodes.
        :param source: start node of the directed edge
        :param sink: end node of the directed edge
        """
        # do not allow self-loops of a node on itself
        if source == sink:
            return

        # adding edges influences if the graph is connected and cyclic
        self.__is_connected = None
        self.__contains_cycle = None

        source_node = self.__get_node_for_component(source)
        self.__add_node(source_node)
        sink_node = self.__get_node_for_component(sink)
        self.__add_node(sink_node)

        # check if source node has already outgoing edges
        if source_node not in self.get_edges().keys():
            self.__edges[source_node] = [sink_node]  # values of dict need to be arrays
        else:
            connected_nodes = self.get_edges().get(source_node)
            # check if source and sink are already connected (to ignore duplicate connection)
            if sink_node not in connected_nodes:
                self.__edges[source_node] = sorted(connected_nodes + [sink_node])

    def get_leaf_nodes(self) -> List[Node]:
        """
        Returns a list of all leaf nodes (=nodes that are only connected to exactly one other node).
        :return: list of leaf nodes
        """
        # leaf nodes only have one outgoing edge
        edges = self.get_edges()
        leaf_nodes = [node for node in self.get_nodes() if len(edges[node]) == 1]
        return leaf_nodes

    def remove_leaf_node(self, node: Node):
        """
        Removes a leaf node and the corresponding edges from the graph.
        :param node: the leaf node to remove
        :raise: ValueError if node is not a leaf node
        """
        if node in self.get_leaf_nodes():
            # remove node from set of nodes
            self.__nodes.discard(node)
            # remove edge where node is sink
            connected_node = self.get_edges()[node][0]
            connected_node_neighbors = self.get_edges()[connected_node]
            connected_node_neighbors.remove(node)
            self.__edges[connected_node] = connected_node_neighbors
            # remove edge where node is source
            self.__edges.pop(node)
        else:
            raise ValueError('Given node is not a leaf node.')

    def remove_leaf_node_by_id(self, node_id: int):
        """
        Removes a leaf node (specified by its node id) and the corresponding edges from the graph.
        :param node_id: the id of the leaf node to remove
        """
        corresponding_node = self.get_node(node_id)
        self.remove_leaf_node(corresponding_node)

    def remove_node(self, node: Union[Node, Component]):
        """
        Removes a node and the corresponding edges from the graph.
        :param node: the node to remove from the current graph
        :raise: ValueError if given node is not contained in the graph
        """
        if isinstance(node, Component):
            node = self.__get_node_for_component(node)
        if node not in self.get_nodes():
            raise ValueError('Node to remove is not contained in Graph.')
        if node in self.get_leaf_nodes():
            self.remove_leaf_node(node)
        else:
            # remove node from set of nodes
            self.__nodes.discard(node)
            # remove edge where node is sink
            connected_nodes = self.get_edges()[node]
            for connected_node in connected_nodes:
                connected_node_neighbors = self.get_edges()[connected_node]
                connected_node_neighbors.remove(node)
                if connected_nodes is []:
                    del self.get_edges()[connected_node]
                else:
                    self.__edges[connected_node] = connected_node_neighbors
            # remove edge where node is source
            del self.get_edges()[node]

            # removing the node could change the connectivity and cyclicity of the graph, so set to unknown
            self.__is_connected = None
            self.__contains_cycle = None

    def get_connected_components(self) -> List['Graph']:
        """
        Returns a list of all connected components of the current graph. If the graph is connected, the list only
        contains one graph, namely the current graph itself.
        :return: a list of all connected components
        """
        subcomponents = []
        unseen_nodes = self.get_nodes()
        while unseen_nodes:
            start_node = unseen_nodes.pop()
            connected_nodes = self.__breadth_search(start_node)
            subcomponents.append(self.__create_subgraph_of_nodes(connected_nodes))
            unseen_nodes.difference_update(connected_nodes)
        return subcomponents

    def __create_subgraph_of_nodes(self, nodes: Set[Node]) -> 'Graph':
        """
        # Returns a subgraph of the current node containing only the given nodes. Instead of creating a new graph as
        subgraph, the internal state (e.g. node_counter, Node variables) of the current graph is retained for the
        subgraph.
        :param nodes: List of Nodes the returned graph should be composed of.
        :return: subgraph containing only the given nodes
        """
        subgraph = copy.deepcopy(self)
        # adjust nodes and edges
        subgraph.__nodes = set(nodes)
        new_edges = dict()
        for source in [key for key in self.get_edges().keys() if key in nodes]:
            sinks = [sink for sink in self.get_edges()[source] if sink in nodes]
            new_edges[source] = sinks
        subgraph.__edges = new_edges

        subgraph.__is_connected = None
        subgraph.__contains_cycle = None
        return subgraph

    def get_node(self, node_id: int):
        """ Returns the corresponding node for a given node id. """
        matching_nodes = [node for node in self.get_nodes() if node.get_id() is node_id]
        if not matching_nodes:
            raise AttributeError('Given node id not found.')
        return matching_nodes[0]

    def to_nx(self):
        """
        Transforms the current graph into a networkx graph
        :return: networkx graph
        """
        graph_nx = nx.Graph()
        for node in self.get_nodes():
            component = node.get_component()
            info = f'\nID={component.get_id()}\nFamilyID={component.get_family_id()}'
        graph_nx.add_node(node, info=info)

        for source_node in self.get_nodes():
            connected_nodes = self.get_edges()[source_node]
            for connected_node in connected_nodes:
                graph_nx.add_edge(source_node, connected_node)
        assert graph_nx.number_of_nodes() == len(self.get_nodes())
        return graph_nx

    def draw(self):
        """ Draws the graph with NetworkX and displays it. """
        graph_nx = self.to_nx()
        labels = nx.get_node_attributes(graph_nx, 'info')
        nx.draw(graph_nx, labels=labels)
        plt.show()

    def get_edges(self) -> Dict[Node, List[Node]]:
        """
        Returns a dictionary containing all directed edges.
        :return: dict of directed edges
        """
        return self.__edges

    def get_nodes(self) -> Set[Node]:
        """
        Returns a set of all nodes.
        :return: set of all nodes
        """
        return self.__nodes

    def get_construction_id(self) -> int:
        """
        Returns the unix timestamp for the creation date of the corresponding construction.
        :return: construction id (aka creation timestamp)
        """
        return self.__construction_id

    def __breadth_search(self, start_node: Node) -> List[Node]:
        """
        Performs a breadth search starting from the given node and returns all node it has seen
        (including duplicates due to cycles within the graph).
        :param start_node: Node of the graph to start the search
        :return: list of all seen nodes (may include duplicates)
        """
        parent_node: Node = None
        queue: List[Tuple[Node, Node]] = [(start_node, parent_node)]
        seen_nodes: List[Node] = [start_node]
        while queue:
            curr_node, parent_node = queue.pop()
            new_neighbors: List[Node] = [n for n in self.get_edges().get(curr_node) if n != parent_node]
            queue.extend([(n, curr_node) for n in new_neighbors if n not in seen_nodes])
            seen_nodes.extend(new_neighbors)
        return seen_nodes

    def is_connected(self) -> bool:
        """
        Returns a boolean that indicates if the graph is connected
        :return: boolean if the graph is connected
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__is_connected is None:
            # choose random start node and start breadth search
            start_node: Node = next(iter(self.get_nodes()))
            seen_nodes: List[Node] = self.__breadth_search(start_node)
            # if we saw all nodes during the breadth search, the graph is connected
            self.__is_connected = set(seen_nodes) == self.get_nodes()
        return self.__is_connected

    def is_cyclic(self) -> bool:
        """
        Returns a boolean that indicates if the graph contains at least one non-trivial cycle.
        A bidirectional edge between two nodes is a trivial cycle, so only cycles of at least three nodes make
        a graph cyclic.
        :return: boolean if the graph contains a cycle
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__contains_cycle is None:
            # choose random start node and start breadth search
            start_node: Node = next(iter(self.get_nodes()))
            seen_nodes: List[Node] = self.__breadth_search(start_node)
            # graph contains a cycle if we saw a node twice during breadth search
            self.__contains_cycle = len(seen_nodes) != len(set(seen_nodes))
        return self.__contains_cycle

    def is_bidirectional(self) -> bool:
        """
        Returns a boolean that indicates if all edges in the Graph are bidirectional.
        :return: boolean if all edges are bidirectional
        """
        if self.get_nodes() == set():
            raise BaseException('Operation not allowed on empty graphs.')

        if self.__is_bidirectional is None:
            edges = [(k, value_elem) for k, value in self.get_edges().items() for value_elem in value]
            for source, sink in edges:
                self.__is_bidirectional = (sink, source) in edges
                if self.__is_bidirectional is False:
                    break

        return self.__is_bidirectional
