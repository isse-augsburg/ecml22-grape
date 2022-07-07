from enum import Enum


class ComponentIdentifier(Enum):
    """ Identifier level for components, either the components themselves or their corresponding family. """
    COMPONENT = 'component'
    FAMILY = 'family'
