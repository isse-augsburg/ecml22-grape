from common.component_identifier import ComponentIdentifier


class Component:
    def __init__(self, id: str, family_id: str):
        assert id and family_id, 'Creation of Component failed. Fields `id` and `family_id` must not be empty.'
        self.__id: str = id
        self.__family_id: str = family_id

    def equivalent(self, other) -> bool:
        """
        Defines equivalence of a given object and this Component object.
        :param other: object to determine equality with this Component
        :raise: TypeError if given object is not a Component
        :return: boolean that indicates if the given object is equivalent to the current component.
        """
        if not isinstance(other, Component):
            raise TypeError(f'Can not compare different types ({str(type(self))} and {str(type(other))})')
        return self.get_id() == other.get_id() and self.get_family_id() == other.get_family_id()

    def __repr__(self) -> str:
        """
        Returns a string representation of the current Component object.
        :return: string representation
        """
        return f'{self.__class__.__name__}(ID={self.get_id()}, FamilyID={self.get_family_id()})'

    def __hash__(self) -> int:
        """ Defines hash of a component. Needed for software tests. """
        return hash((self.get_id(), self.get_family_id()))

    def __lt__(self, other) -> bool:
        """ Defines an order on component. """
        if not isinstance(other, Component):
            raise TypeError(f'Can not define order for different types ({str(type(self))} and {str(type(other))})')
        return (self.get_id().lower(), self.get_family_id().lower()) < (other.get_id().lower(), other.get_family_id().lower())

    def get_id(self) -> str:
        return self.__id

    def get_family_id(self) -> str:
        return self.__family_id

    def get(self, identifier: ComponentIdentifier) -> str:
        """
        Returns the value of the component for the given ComponentIdentifier.
        :param identifier: component identifier
        :return: value of the component for the component identifier
        """
        if identifier == ComponentIdentifier.COMPONENT:
            return self.get_id()
        elif identifier == ComponentIdentifier.FAMILY:
            return self.get_family_id()
        else:
            raise AttributeError('Value of ComponentIdentifier unknown: ', identifier)
