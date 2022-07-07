from common.catalog import Catalog

# Hyperparameter settings for component prediction
# Set here which model should be trained/ evaluated
info = {
    'experiment_name': 'part_prediction',   # 'part_prediction_eval', 'part_prediction'
    'catalog': Catalog.A,
    'embedding_size': 20,   # 20, 100, 'one_hot'
    'model_type': 'gcn',  # 'gat', 'gcn'
    'max_epochs': 500
}

# Set the hyperparameters of the specific models here (see hyperparameter_configuration.py)
hp = {
    Catalog.C.value: {
        20: {
            'gcn': {},
            'gat': {}
        },
        100: {
            'gcn': {},
            'gat': {}
        },
        'one_hot': {
            'gcn': {},
            'gat': {}
        }
    },
    Catalog.B.value: {
        20: {
            'gcn': {},
            'gat': {}
        },
        100: {
            'gcn': {},
            'gat': {}
        },
        'one_hot': {
            'gcn': {},
            'gat': {}
        }
    }, Catalog.A.value: {
        20: {
            'gcn': {},
            'gat': {}
        },
        100: {
            'gcn': {},
            'gat': {}
        },
        'one_hot': {
            'gcn': {},
            'gat': {}
        }
    }
}
