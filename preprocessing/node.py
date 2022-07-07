from preprocessing.component import Component


class Node:
    def __init__(self, node_id: int, component: Component):
        self.__id: int = node_id
        self.__component: Component = component

    def __eq__(self, other) -> bool:
        """ Defines equality of two nodes. Needed for software tests. """
        if other is None:
            return False
        if not isinstance(other, Node):
            raise TypeError(f'Can not compare different types ({str(type(self))} and {str(type(other))})')
        return self.get_id() == other.get_id() and self.get_component() == other.get_component()

    def __hash__(self) -> int:
        """ Defines hash of a node. Needed for software tests. """
        return hash((self.get_id(), self.get_component()))

    def __lt__(self, other) -> bool:
        """ Defines an order on nodes. """
        if not isinstance(other, Node):
            raise TypeError(f'Can not define order for different types ({str(type(self))} and {str(type(other))})')
        return self.get_component() < other.get_component()

    def get_id(self) -> int:
        return self.__id

    def get_component(self) -> Component:
        return self.__component
