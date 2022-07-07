from common.component_identifier import ComponentIdentifier
from preprocessing.graph import Graph
from preprocessing.node import Node
from util import file_handler

import logging
from typing import List, Set

logger = logging.getLogger(__name__)


class GramsFromGraphs:

    def __init__(self, identifier: ComponentIdentifier, separator: str = '|||'):
        self.separator = separator
        self.__identifier = identifier
        self.__content = ''

    def graphs_to_data_set(self, graph_list: List[Graph], output_file: str, context_size: int = 2):
        """
        Parameters:
            graph_list (list): List of graphs to parse to training set
            output_file (str): An output file, in which every line will form a
                training sample. This file might also contain empty lines,
                which are to be ignored.
            context_size (int): Determines how far components can be away from each
                other in the construction. Explained in 'Description' above.
        """
        logger.info('Transforming graphs to grams ...')
        for graph in graph_list:
            for node in graph.get_nodes():
                context_nodes = self.__breadth_search(graph, node, context_size)
                self.__save_data_instances(node, context_nodes)

        # remove \n at the end of content before saving
        if self.__content.endswith('\n'):
            self.__content = self.__content[:-1]
        file_handler.save_string(self.__content, output_file)

        logger.info('Data set generated of %d graphs', len(graph_list))

    def __breadth_search(self, graph: Graph, start_node: Node, context_size: int) -> Set[Node]:
        remaining_path_length = context_size
        queue = [{start_node: remaining_path_length}]
        seen_nodes = {start_node}
        while queue:
            node_dict = queue.pop()
            node = next(iter(node_dict))
            remaining_path_length = node_dict[node] - 1
            if remaining_path_length >= 0:
                new_neighbors = [n for n in graph.get_edges().get(node) if n not in seen_nodes]
                for neighbor in new_neighbors:
                    seen_nodes.add(neighbor)
                    queue.append({neighbor: remaining_path_length})
        return seen_nodes

    def __save_data_instances(self, center_node: Node, context_nodes: Set[Node]):
        center_node_identifier = center_node.get_component().get(self.__identifier)
        context_nodes.remove(center_node)

        for context_component in context_nodes:
            context_node_identifier = context_component.get_component().get(self.__identifier)
            self.__content += str(center_node_identifier + self.separator + context_node_identifier + '\n')
