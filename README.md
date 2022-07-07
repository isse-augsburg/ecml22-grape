# GRAPE: Graph-Based Recommendations for Assemblies using Pretrained Embeddings

**Corresponding Paper: "A Recommendation System for CAD Assembly Modeling based on Graph Neural Networks" (ECML22)**

This project is written with Python `3.8` based on Anaconda (https://www.anaconda.com/distribution/).

## Setup
The default setup installs pytorch and dgl without cuda support for CPU only. If your machine includes a NVIDIA GPU and
you want to benefit from the speed-up, you can replace the line `cpuonly` in the file `requirements.txt` with suitable
version of `cudatoolkit` to install pytorch with cuda support and extend `dgl` with the corresponding cuda version, e.g. `dgl-cuda10.2`.

We strongly recommend to use a virtual environment to ensure consistency, for example:
`conda create -n GRAPE python=3.8`

Install dependencies:
`conda install -c conda-forge -c dglteam -c pytorch --file requirements.txt`



## Structure of the project
* **preprocessing**: contains necessary data structures for graphs (graph, node, component)
* **data_set_generator**: transform data objects into data sets for machine learning (e.g. transform graphs into gram-based samples for Word2Vec)
* **models**: machine learning models (e.g. Word2Vec or GNNs for component prediction)
* **scripts**: folder containing all relevant scripts of the project
* **data**: folder to contain the training, validation and test data sets for the three catalogs - they can be found under https://figshare.com/articles/dataset/ECML22_GRAPE_Data/20239767.


**The hyperparameters of the best performing models can be found in models/component_prediction/hyperparameter_configuration.py.**


## Workflow
All scripts can be found in module `scripts`.
1. Dowload the data from [Figshare](https://figshare.com/articles/dataset/ECML22_GRAPE_Data/20239767) as stated above.
2. Create component embeddings (`train_embedding.py`) and one-hot encoding for components (`create_one_hot_embedding.py`)
3. Use the created representations to generate dgl-readable samples for the GNN models (`create_dgl_instances.py`)
4. Train the GNN-based component prediction models (`train_prediction_model.py`). The hyperparameter configuration of the models can  be set in `models/component_prediction/model_configuration.py`.
