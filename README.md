# Improved model for predicting the retrosynthesizability of compounds by machine learning
This repository shows models created in my paper.
`make_model_and_metrics.py` make model1, model2, model3, and evaluate model1, model2, model3, and base model(RAscore). The result of each model and evaluation are in `model*` and `base`. Datasets used to train and test are also available and in `Data`. 

To run `make_model_and_metrics.py`, you have to install rdkit via Anaconda, clone RAscore repository, and refact `RAscore_path` in `make_model_and_metrics.py`.