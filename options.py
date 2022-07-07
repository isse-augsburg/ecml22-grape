import os

# Folder
DIRECTORY = os.path.dirname(__file__) + '/'

DATA_FOLDER = 'data/'
LOG_FOLDER = 'logs/'
RESULT_FOLDER = 'results/'

DATA_LOCATION = os.path.join(DIRECTORY, DATA_FOLDER)

RESULT_LOCATION = os.path.join(DIRECTORY, RESULT_FOLDER)

if not os.path.exists(RESULT_LOCATION):
    os.makedirs(RESULT_LOCATION)

EMBEDDING_KEYWORD = 'embedding'
