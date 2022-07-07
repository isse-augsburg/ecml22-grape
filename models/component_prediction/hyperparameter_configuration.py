from common.catalog import Catalog

"""
This file contains the hyperparameter configurations of the best performing GNN models per catalog (A, B, or C),
component representation (20-dim., 100-dim., or one-hot) and model type (gat or gcn).
"""

hyperparameters = {
    Catalog.C.value: {
        20: {
            'gcn': {
                'dropout_rate': 0.3,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 5
            },
            'gat': {
                'dropout_rate': 0.1,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 4,
                'num_heads': 3
            }
        },
        100: {
            'gcn': {
                'dropout_rate': 0.1,
                'hidden_size': 512,
                'learning_rate': 0.0001,
                'n_hidden_layers': 5
            },
            'gat': {
                'dropout_rate': 0.0,
                'hidden_size': 128,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3,
                'num_heads': 42
            }
        },
        'one_hot': {
            'gcn': {
                'dropout_rate': 0.0,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3
            },
            'gat': {
                'dropout_rate': 0.5,
                'hidden_size': 256,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3,
                'num_heads': 22
            }
        }
    },
    Catalog.B.value: {
        20: {
            'gcn': {
                'dropout_rate': 0.5,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 7
            },
            'gat': {
                'dropout_rate': 0.0,
                'hidden_size': 128,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3,
                'num_heads': 37
            }
        },
        100: {
            'gcn': {
                'dropout_rate': 0.1,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3
            },
            'gat': {
                'dropout_rate': 0.0,
                'hidden_size': 256,
                'learning_rate': 0.0001,
                'n_hidden_layers': 4,
                'num_heads': 14
            }
        },
        'one_hot': {
            'gcn': {
                'dropout_rate': 0.3,
                'hidden_size': 512,
                'learning_rate': 0.0001,
                'n_hidden_layers': 6
            },
            'gat': {
                'dropout_rate': 0.1,
                'hidden_size': 256,
                'learning_rate': 0.0001,
                'n_hidden_layers': 7,
                'num_heads': 6
            }
        }
    }, Catalog.A.value: {
        20: {
            'gcn': {
                'dropout_rate': 0.3,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 5
            },
            'gat': {
                'dropout_rate': 0.0,
                'hidden_size': 128,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3,
                'num_heads': 10,
            }
        },
        100: {
            'gcn': {
                'dropout_rate': 0.3,
                'hidden_size': 512,
                'learning_rate': 0.00010253609633342166,
                'n_hidden_layers': 2,
            },
            'gat': {
                'dropout_rate': 0.3,
                'hidden_size': 1024,
                'learning_rate': 0.0001,
                'n_hidden_layers': 5,
                'num_heads': 2
            }
        },
        'one_hot': {
            'gcn': {
                'dropout_rate': 0.0,
                'hidden_size': 1024,
                'learning_rate': 0.001,
                'n_hidden_layers': 2
            },
            'gat': {
                'dropout_rate': 0.3,
                'hidden_size': 64,
                'learning_rate': 0.0001,
                'n_hidden_layers': 3,
                'num_heads': 36
            }
        }
    }
}
