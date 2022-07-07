"""
This file provides basic functionality for saving content to and loading content from files.
"""

import pickle


def deserialize(filename: str) -> object:
    """
    Loads a pickled file and returns its content.
    Parameters:
        filename (str): Path of the file to load
    Returns:
        Content of the loaded file (object)
    """
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
    return obj


def serialize(obj: object, filename: str) -> None:
    """
    Saves an object into a file with pickle.
    Parameters:
        obj (object): Content to store in the file
        filename (str): Path of the file to load
    """
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)


def load_string(filename: str) -> str:
    """
    Loads a string from a file and returns the content.
    Parameters:
        filename (str): Path of the file to load
    Returns:
        content of the loaded file (str)
    """
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def save_string(content: str, filename: str) -> None:
    """
    Saves a string into a file.
    Parameters:
        content (str): String to save
        filename (str): Path of the file to load
    """
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)
