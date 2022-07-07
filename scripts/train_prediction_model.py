"""
The training script for our GCN- and GAT-based component recommendation model (GcnPredictor, GatPredictor)
with evaluation appended to compare the performance.
"""
from common.component_identifier import ComponentIdentifier
from models.component_prediction.prediction_evaluator import PredictionEvaluator
from models.component_prediction.prediction_trainer import PredictionTrainer
from models.component_prediction.gat_predictor import GatPredictor
from models.component_prediction.gcn_predictor import GcnPredictor
from models.component_prediction.prediction_dataset import PredictionDataset
from models.component_prediction import model_configuration as cfg
from util.file_handler import deserialize
from util.linux_handler import LinuxHandler
from util.windows_handler import WindowsHandler
import options as op

import copy
import dgl
import logging
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import optuna
import os
import sys
import torch


if __name__ == "__main__":

    # load train configuration at the beginning
    cfg_info = copy.deepcopy(cfg.info)
    cfg_hp = copy.deepcopy(cfg.hp)

    torch.manual_seed(0)
    np.random.seed(0)
    dgl.seed(0)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(f'CUDA available? {torch.cuda.is_available()}')

    os_handler = WindowsHandler() if os.name == 'nt' else LinuxHandler()

    os.environ['NUM_WORKER'] = '2'

    model_type = cfg_info['model_type']

    # ---------- configurations ----------------------------------------
    """
    To change from training mode to evaluation mode, set skip_training = True.
    evaluate_final_model determines, if the val (False) or test set (True) is used for evaluation.
    """
    skip_training = False  # if so, expect a saved model in results
    evaluate_final_model = False

    identifier = ComponentIdentifier.COMPONENT
    catalog_name = cfg_info['catalog'].value
    embedding_size = cfg_info['embedding_size']
    print(f'Catalog {catalog_name}\tEmbedding size {embedding_size}')

    # load hyperparameters
    hyperparameters = cfg_hp[catalog_name][embedding_size][model_type]

    # ---------- data loading ----------------------------------------
    data_path = op.DATA_LOCATION

    vocabulary_size = len(deserialize(f'{op.DATA_LOCATION}{catalog_name}_vocabulary.dat'))
    embedding_width = vocabulary_size if embedding_size == 'one_hot' else int(embedding_size)
    transform_to_emb_size = vocabulary_size if embedding_size == 'one_hot' else None

    data_filename = f'{data_path}{catalog_name}_{identifier.value}_{embedding_size}_dgl.hdf5'
    train_dataset = PredictionDataset(data_filename, 'train', transform_to_emb_size)
    eval_set_name = 'test' if evaluate_final_model else 'val'
    val_dataset = PredictionDataset(data_filename, eval_set_name, transform_to_emb_size)

    print('Data loaded.')

    # ---------- mlflow setup ----------------------------------------
    # mlflow.set_tracking_uri(url_to_tracking_server)

    # manage mlflow experiments only within those 3 characteristics parameter values could be compared and tuned
    experiment_name = cfg_info['experiment_name']
    exp_info = MlflowClient().get_experiment_by_name(experiment_name)
    exp_id = exp_info.experiment_id if exp_info else MlflowClient().create_experiment(experiment_name)
    print('mlflow setup done')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def objective(trial):
        with mlflow.start_run(experiment_id=exp_id) as run:  # automatically terminates run
            run_id = run.run_id = run.info.run_uuid
            if hyperparameters.get('run_id', None):
                run_id = hyperparameters['run_id']
            mlflow.set_tag("run_id", run_id)
            print('mlflow run_id', run_id)

            model_output_file = f'{op.RESULT_LOCATION}{catalog_name}_{embedding_size}_{model_type}_{run_id}.dat'
            model_plot_file = f'{op.RESULT_LOCATION}{catalog_name}_{embedding_size}_{model_type}_{run_id}.png'
            attention_plot_prefix = f'{op.RESULT_LOCATION}{catalog_name}_{embedding_size}_{model_type}_{run_id}_attention'

            # Define component prediction models
            hidden_size = hyperparameters.get('hidden_size',
                                              trial.suggest_categorical("hidden_size", [64, 128, 256, 512, 1024]))
            n_hidden_layers = hyperparameters.get('n_hidden_layers', trial.suggest_int("n_layers", 2, 8))
            feature_dropout_rate = hyperparameters.get('dropout_rate',
                                                       trial.suggest_categorical("dropout_rate", [0.0, 0.1, 0.3, 0.5]))

            if model_type == 'gcn':
                model = GcnPredictor(node_feature_dim=embedding_width, hidden_dim=hidden_size,
                                     num_hidden_layers=n_hidden_layers, num_classes=vocabulary_size,
                                     dropout_rate=feature_dropout_rate)
            elif model_type == 'gat':
                num_heads = hyperparameters.get('num_heads', trial.suggest_int("heads", 2, 50, log=True))
                model = GatPredictor(node_feature_dim=embedding_width, hidden_dim=hidden_size, num_heads=num_heads,
                                     num_hidden_layers=n_hidden_layers, num_classes=vocabulary_size,
                                     feat_drop=feature_dropout_rate, attn_drop=0.0, residual=False)
            else:
                raise AttributeError(f"Model Type {model_type} unknown.")

            max_epochs = cfg_info['max_epochs']
            learning_rate = hyperparameters.get('learning_rate', trial.suggest_categorical("lr", [1e-4, 1e-5]))

            # enter model specific logging data
            mlflow.log_param("catalog", catalog_name)
            mlflow.log_param("embedding", embedding_size)
            mlflow.log_param("num_classes", model.get_num_classes())
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("max_epochs", max_epochs)
            mlflow.log_param("node_feature_dim", embedding_width)
            mlflow.log_param("hidden_dim", hidden_size)
            mlflow.log_param("n_hidden_layers", n_hidden_layers)
            mlflow.log_param("feature_dropout_rate", feature_dropout_rate)
            mlflow.log_param("learning_rate", learning_rate)

            if model_type == 'gat':
                mlflow.log_param("num_heads", num_heads)

            # train models
            if skip_training:
                print(f'Load model from {model_output_file}.')
            else:
                trainer = PredictionTrainer(model=model, train_set=train_dataset, val_set=val_dataset,
                                            model_name=model_type, max_epochs=max_epochs, learning_rate=learning_rate,
                                            os_handler=os_handler)
                train_loss, val_loss, best_epoch = trainer.train(device)

                mlflow.log_metric("train_loss", train_loss)
                mlflow.log_metric("val_loss", val_loss)
                mlflow.log_metric("best_epoch", best_epoch)
                trainer.save(model_output_file, model_plot_file)

                mlflow.log_artifact(local_path=model_plot_file)
                mlflow.log_artifact(local_path=model_output_file)

            # evaluate trained model
            print('Start Evaluation')
            evaluator = PredictionEvaluator(val_dataset, model, model_output_file, os_handler)
            investigate_attention = evaluate_final_model and model_type == 'gat'
            evaluation = evaluator.evaluate(k_values_to_test=(1, 2, 3, 5, 10, 15, 20),
                                            investigate_attention=investigate_attention)
            if investigate_attention:
                evaluator.save_entropy_histograms(attention_plot_prefix)
                for idx in range(num_heads):
                    mlflow.log_artifact(local_path=f'{attention_plot_prefix}_{idx}.png')

            mlflow.log_metrics(evaluation)

        return evaluation.get("4 Hit-rate 5")

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
    study.optimize(objective, n_trials=1 if skip_training else 30)

    # calculation time
    df = study.trials_dataframe()

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    logging.info('Study statistics: ')
    logging.info(' Number of finished trials: {}'.format(len(study.trials)))
    logging.info(' Number of pruned trials: {}'.format(len(pruned_trials)))
    logging.info(' Number of complete trials: {}'.format(len(complete_trials)))

    logging.info('Best trial:')
    trial = study.best_trial

    logging.info('  Value: {}'.format(trial.value))

    logging.info('  Params: ')
    for key, value in trial.params.items():
        logging.info('    {}: {}'.format(key, value))
