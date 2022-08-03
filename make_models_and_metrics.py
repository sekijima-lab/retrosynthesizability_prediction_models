# model1,2,3を作って
# ChemTS, ChEMBL, MERMAID, GDBMedChemで評価する

import os
import swifter
from zipfile import ZipFile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
import optuna
import json
#from keras.layers import Dense, Dropout
from keras import models
from rdkit import Chem
from rdkit.Chem import AllChem
import descriptors
from sklearn.model_selection import train_test_split
import optuna
import time

RAscore_path = "path_of_RAscore"
Dir_path = os.getcwd()

config = os.path.join(RAscore_path, "RAscore/model_building/example_classification_configs/NN_classifier.json")
opt_time_file = os.path.join(RAscore_path, "all_layer/opt_time.txt")

model1_chemts_metrics_file = os.path.join(Dir_path, "model/model1/model1_ChemTS_metrics.json")
model1_chembl_metrics_file = os.path.join(Dir_path, "model/model1/model1_ChEMBL_metrics.json")
model1_gdbmed_metrics_file = os.path.join(Dir_path, "model/model1/model1_GDBMed_metrics.json")
model1_mermaid_metrics_file = os.path.join(Dir_path, "model/model1/model1_MERMAID_metrics.json")

model2_chemts_metrics_file = os.path.join(Dir_path, "model/model2/model2_ChemTS_metrics.json")
model2_chembl_metrics_file = os.path.join(Dir_path, "model/model2/model2_ChEMBL_metrics.json")
model2_gdbmed_metrics_file = os.path.join(Dir_path, "model/model2/model2_GDBMed_metrics.json")
model2_mermaid_metrics_file = os.path.join(Dir_path, "model/model2/model2_MERMAID_metrics.json")

model3_chemts_metrics_file = os.path.join(Dir_path, "model/model3/model3_ChemTS_metrics.json")
model3_chembl_metrics_file = os.path.join(Dir_path, "model/model3/model3_ChEMBL_metrics.json")
model3_gdbmed_metrics_file = os.path.join(Dir_path, "model/model3/model3_GDBMed_metrics.json")
model3_mermaid_metrics_file = os.path.join(Dir_path, "model/model3/model3_MERMAID_metrics.json")

base_chemts_metrics_file = os.path.join(Dir_path, "model/base/base_ChemTS_metrics.json")
base_chembl_metrics_file = os.path.join(Dir_path, "model/base/base_ChEMBL_metrics.json")
base_gdbmed_metrics_file = os.path.join(Dir_path, "model/base/base_GDBMed_metrics.json")
base_mermaid_metrics_file = os.path.join(Dir_path, "model/base/base_MERMAID_metrics.json")

model1_path = os.path.join(Dir_path, 'model/model1/model1.h5')
model2_path = os.path.join(Dir_path, 'model/model2/model2.h5')
model3_path = os.path.join(Dir_path, 'model/model3/model3.h5')

conf = {
"train_size": 0.9,
"test_size": 0.1,
"batch_size": 256,
"epochs": 100,
"descriptor": "fcfp_counts",
"n_trials": 100,
"algorithm": {
    "DNNClassifier": {
    "layer_1": [128, 256, 512],
    "activation_1": ["relu", "elu", "selu", "linear"],
    "dropout_1": 0.1,
    "max_layers": 10,
    "layer_size": [128, 256, 512],
    "layer_activations": ["relu", "elu", "selu", "linear"],
    "layer_droput": {"low": 0,
                    "high": 0.5},
    "learning_rate": {"low": 1e-5,
                    "high": 1e-1}
        }   
    }
}


#ChemTSテストセットと学習データセットの作成
chemts_test_path = os.path.join(Dir_path, "Data/test_ChemTS.csv")
chemts_train_path = os.path.join(Dir_path, "Data/train_ChemTS.csv")
print(f"chemts_test_data = {chemts_test_path}, chemts_train_data = {chemts_train_path}\n")

chemts_test_data = pd.read_csv(chemts_test_path)
chemts_train_data = pd.read_csv(chemts_train_path)

if 'dataset' in chemts_train_data.columns:
    chemts_train_data.drop(columns='dataset', inplace=True)

if not 'descriptor' in chemts_train_data.columns:
    chemts_train_data['descriptor'] = chemts_train_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
    chemts_test_data['descriptor'] = chemts_test_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
else:
    print(f'''chemts_train_data にカラム['descriptor']が見つかったため，更新は行いませんでした''')

#トレーニングデータは90000個に絞る
chemts_train_data = chemts_train_data.sample(n=90000)
print(f"chemts_train_data.shape is changed to {chemts_train_data.shape}")

chemts_train_X = np.stack(chemts_train_data['descriptor'].values)
chemts_train_y = np.stack(chemts_train_data['activity'].values)

chemts_test_X = np.stack(chemts_test_data['descriptor'].values)
chemts_test_y = np.stack(chemts_test_data['activity'].values)

if isinstance(chemts_train_X[0], float):
    chemts_train_X = np.array([[i] for i in chemts_train_X])
    chemts_train_y = np.array([[i] for i in chemts_train_y])



#ChEMBLテストセットの作成
chembl_test_path = os.path.join(RAscore_path, "RAscore/data/uspto_chembl_classification_test.csv")
chembl_train_path = os.path.join(RAscore_path, "RAscore/data/uspto_chembl_classification_train.csv")
print(f"chembl_test_data = {chembl_test_path}, chembl_train_path = {chembl_train_path}\n")

chembl_test_data = pd.read_csv(chembl_test_path)
chembl_train_data = pd.read_csv(chembl_train_path)

if not 'descriptor' in chembl_train_data.columns:
    chembl_train_data['descriptor'] = chembl_train_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
    chembl_test_data['descriptor'] = chembl_test_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
else:
    print(f'''chembl_train_data にカラム['descriptor']が見つかったため，更新は行いませんでした''')

chembl_test_X = np.stack(chembl_test_data['descriptor'].values)
chembl_test_y = np.stack(chembl_test_data['activity'].values)


#トレーニングデータは90000個に絞る
chembl_train_data = chembl_train_data.sample(n=90000)
print(f"chembl_train_data.shape is changed to {chembl_train_data.shape}")

chembl_train_X = np.stack(chembl_train_data['descriptor'].values)
chembl_train_y = np.stack(chembl_train_data['activity'].values)

if isinstance(chembl_train_X[0], float):
    chembl_train_X = np.array([[i] for i in chembl_train_X])
    chembl_train_y = np.array([[i] for i in chembl_train_y])


#GDBMedChemテストセットの作成
gdbmed_test_path = os.path.join(RAscore_path, "RAscore/data/uspto_gdbmedchem_classification_test.csv")
print(f"gdbmed_test_data = {gdbmed_test_path}\n")

gdbmed_test_data = pd.read_csv(gdbmed_test_path)

if not 'descriptor' in gdbmed_test_data.columns:
   gdbmed_test_data['descriptor'] = gdbmed_test_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
else:
    print(f'''gdbmed_test_data にカラム['descriptor']が見つかったため，更新は行いませんでした''')

gdbmed_test_X = np.stack(gdbmed_test_data['descriptor'].values)
gdbmed_test_y = np.stack(gdbmed_test_data['activity'].values)


#MERMAIDテストセットの作成
mermaid_test_path = os.path.join(Dir_path, "Data/mermaid_test.csv")
print(f"mermaid_test_data = {mermaid_test_path}\n")

mermaid_test_data = pd.read_csv(mermaid_test_path)

if not 'descriptor' in mermaid_test_data.columns:
   mermaid_test_data['descriptor'] = mermaid_test_data['smi'].swifter.apply(descriptors.ecfp_counts, radius=3, useFeatures=True, useCounts=True)
else:
    print(f'''mermaid_test_data にカラム['descriptor']が見つかったため，更新は行いませんでした''')

mermaid_test_X = np.stack(mermaid_test_data['descriptor'].values)
mermaid_test_y = np.stack(mermaid_test_data['activity'].values)


###########################既存モデル###################################
def do_base ():
    #既存モデルのロード
    model = keras.models.load_model(os.path.join(RAscore_path, "RAscore/RAscore/models/DNN_chembl_fcfp_counts/model.tf"))

    #既存モデルによるchemtsテストセットの評価
    base_score = model.evaluate(chemts_test_X, chemts_test_y, verbose=0)
    results = {}

    for metric, s in zip(model.metrics_names, base_score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はbase_chemts_metrics_fileで指定したファイルに保存
    with open(base_chemts_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)



    #既存モデルによるCheMBLテストセットの評価
    base_score = model.evaluate(chembl_test_X, chembl_test_y, verbose=0)
    results = {}

    for metric, s in zip(model.metrics_names, base_score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はbase_chembl_metrics_fileで指定したファイルに保存
    with open(base_chembl_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


    #既存モデルによるGDBMedChemテストセットの評価
    base_score = model.evaluate(gdbmed_test_X, gdbmed_test_y, verbose=0)
    results = {}

    for metric, s in zip(model.metrics_names, base_score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はbase_gdbmed_metrics_fileで指定したファイルに保存
    with open(base_gdbmed_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


    #既存モデルによるMERMAIDテストセットの評価
    base_score = model.evaluate(mermaid_test_X, mermaid_test_y, verbose=0)
    results = {}

    for metric, s in zip(model.metrics_names, base_score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はbase_mermaid_metrics_fileで指定したファイルに保存
    with open(base_mermaid_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)

########################################################################


###########################モデル1（add model）###################################
def do_model1():
    #すでに学習済みモデルが存在するならばロード
    if os.path.exists(model1_path):
        model1 = keras.models.load_model(model1_path)
        print(f"loaded {model1_path}\n")
    #しないならば，新しいデータで重み学習ずみモデル作って保存
    else:
        model1 = keras.models.load_model(os.path.join(RAscore_path, "RAscore/RAscore/models/DNN_chembl_fcfp_counts/model.tf"))
        start = time.time()
        model1.fit(chemts_train_X,
                chemts_train_y,
                validation_split=conf['test_size'],
                shuffle=True,
                batch_size=conf['batch_size'],
                epochs=conf['epochs'],
                verbose=False
                )
        end = time.time()
        
        with open(opt_time_file, 'w') as opt_time:
            opt_time.write("optimization time: {}s".format(end - start))
        
        print('summarize updated model1')
        model1.summary()
        model1.save(model1_path)

    #MERMAIDのテストセットで評価
    score = model1.evaluate(chemts_test_X, chemts_test_y, verbose=0)

    results = {}

    for metric, s in zip(model1.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmetrics_fileで指定したファイルに保存
    with open(model1_chemts_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)



    #ChEMBLのテストセットで評価
    score = model1.evaluate(chembl_test_X, chembl_test_y, verbose=0)

    results = {}

    for metric, s in zip(model1.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmetrics_chembl_fileで指定したファイルに保存
    with open(model1_chembl_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


    #GDBMedchemのテストセットで評価
    score = model1.evaluate(gdbmed_test_X, gdbmed_test_y, verbose=0)

    results = {}

    for metric, s in zip(model1.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmodel1_gdbmed_metrics_fileで指定したファイルに保存
    with open(model1_gdbmed_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


    #MERMAIDテストセットの評価
    score = model1.evaluate(mermaid_test_X, mermaid_test_y, verbose=0)
    results = {}

    for metric, s in zip(model1.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はmodel1_mermaid_metrics_fileで指定したファイルに保存
    with open(model1_mermaid_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


###################################################################################



###########################モデル2（reset model）###################################
def do_model2():
    ##すでに学習済みモデルが存在するならばロード
    if os.path.exists(model2_path):
        model2 = keras.models.load_model(model2_path)
        print(f"loaded {model2_path}\n")
    else:
        model2 = keras.models.clone_model(keras.models.load_model(os.path.join(RAscore_path, "RAscore/RAscore/models/DNN_chembl_fcfp_counts/model.tf")))
        metrics=[tf.keras.metrics.AUC(),
                tf.keras.metrics.Accuracy(),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.FalsePositives()
                ]
        with open(os.path.join(RAscore_path, "RAscore/RAscore/models/DNN_chembl_fcfp_counts/best_params.json")) as json_file:
                best_params = json.load(json_file)

        model2.compile(loss='binary_crossentropy', 
                    optimizer=Adam(lr=best_params['learning_rate']),
                    metrics=metrics)
        
        train_X = np.concatenate([chembl_train_X, chemts_train_X]) 
        train_y = np.concatenate([chembl_train_y, chemts_train_y])
        
        print(f"train_X.shape is {train_X.shape}, train_y.shape is {train_y.shape}")

        model2.fit(train_X,
                train_y,
                validation_split=conf['test_size'],
                shuffle=True,
                batch_size=conf['batch_size'],
                epochs=conf['epochs'],
                verbose=False
                )

    print('summarize updated model2')
    model2.summary()
    model2.save(model2_path)
        
    #chemtsのテストセットで評価
    score = model2.evaluate(chemts_test_X, chemts_test_y, verbose=0)

    results = {}

    for metric, s in zip(model2.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmetrics_fileで指定したファイルに保存
    with open(model2_chemts_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)



    #ChEMBLのテストセットで評価
    score = model2.evaluate(chembl_test_X, chembl_test_y, verbose=0)

    results = {}

    for metric, s in zip(model2.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmetrics_chembl_fileで指定したファイルに保存
    with open(model2_chembl_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)




    #GDBMedchemのテストセットで評価
    score = model2.evaluate(gdbmed_test_X, gdbmed_test_y, verbose=0)

    results = {}

    for metric, s in zip(model2.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をgdbmed_metrics_fileで指定したファイルに保存
    with open(model2_gdbmed_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


    #MERMAIDテストセットの評価
    score = model2.evaluate(mermaid_test_X, mermaid_test_y, verbose=0)
    results = {}

    for metric, s in zip(model2.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はmodel2_mermaid_metrics_fileで指定したファイルに保存
    with open(model2_mermaid_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


############################################################################################


###########################モデル3（reset and tune model）###################################
def do_model3():
    class Objective(object):
        def __init__(self, x, y, conf):
            # Hold this implementation specific arguments as the fields of the class.
            self._train_X, self._test_x, self._train_Y, self._test_y = train_test_split(x, y, train_size=conf['train_size'], test_size=conf['test_size'], random_state=42, shuffle=True)
            self._conf = conf
        
        def __call__(self, trial):
            return self._classifier(trial)
        
        def _build_model(self, trial, algorithm):
            layers = []
                
            inputs = tf.keras.Input(shape=(self._train_X.shape[1],))
            layers.append(inputs)
            
            x = Dense(
                trial.suggest_categorical("layer_1", self._conf['algorithm'][algorithm]['layer_1']),
                activation=trial.suggest_categorical("activation_1", 
                                                    self._conf['algorithm'][algorithm]['activation_1']))(layers[-1])
            layers.append(x)
            
            x = Dropout(self._conf['algorithm'][algorithm]['dropout_1'])(layers[-1])
            layers.append(x)
                                        
            num_layers = trial.suggest_int("num_layers", 2, self._conf['algorithm'][algorithm]['max_layers'])
            for l in range(2, num_layers+1):
                x = Dense(
                trial.suggest_categorical("units_{}".format(l), self._conf['algorithm'][algorithm]['layer_size']),
                activation=trial.suggest_categorical("activation_{}".format(l), 
                                                    self._conf['algorithm'][algorithm]['layer_activations']))(layers[-1])
                layers.append(x)
                x = Dropout(round(trial.suggest_float("dropout_{}".format(l), self._conf['algorithm'][algorithm]['layer_droput']['low'], self._conf['algorithm'][algorithm]['layer_droput']['high']), 1))(layers[-1])
                layers.append(x)
            
            out = Dense(1, activation='sigmoid', name = 'target')(layers[-1])
            
            model = Model(inputs=[inputs], outputs=[out])
            
            learning_rate = float(trial.suggest_loguniform("learning_rate", self._conf['algorithm'][algorithm]['learning_rate']['low'], self._conf['algorithm'][algorithm]['learning_rate']['high']))
        
            model.compile(loss='binary_crossentropy', 
                            optimizer=Adam(lr=learning_rate),
                            metrics=[tf.keras.metrics.AUC()])
            
            return model

        def _classifier(self, trial):
            algorithm = list(self._conf['algorithm'].keys())[0]
        
            scores =[]
            losses = []

            model = self._build_model(trial, algorithm)
        
            model.fit(self._train_X,
                        self._train_Y,
                        validation_split=self._conf['test_size'],
                        shuffle=True,
                        batch_size=self._conf['batch_size'],
                        epochs=self._conf['epochs'],
                        verbose=False
                        )
            
            score = model.evaluate(self._test_x, 
                                    self._test_y, 
                                    verbose=0)
            
            scores.append(score[1])
            losses.append(score[0])

            mean_score = np.array(scores).mean()
            #print(model.metrics_names)
            #print(score)
                                
            return mean_score

    ##すでに学習済みモデルが存在するならばロード
    if os.path.exists(model3_path):
        model3 = keras.models.load_model(model3_path)
        print(f"loaded {model3_path}\n")

    else:
        best_params_file = os.path.join(Dir_path, 'model/model3/best_params.json')
        best_value_file = os.path.join(Dir_path, 'model/model3/best_value.txt')
        study = optuna.create_study(direction="maximize")
        train_X = np.concatenate([chembl_train_X, chemts_train_X])
        train_y = np.concatenate([chembl_train_y, chemts_train_y])
        objective = Objective(train_X, train_y, conf)

        if os.path.exists(best_params_file):
            print(f"loaded {best_params_file}\n")
            with open(best_params_file, 'r') as file:
                best_params = json.load(file)

        else:
            study.optimize(objective, n_trials=conf['n_trials'])
            best_params = study.best_best_params
            with open(best_params_file, 'w') as outfile:
                json.dump(study.best_params, outfile)
            with open(best_value_file, 'w') as outfile:
                outfile.write("Best Trial Value: {}".format(study.best_value))
        
        layers = []
        inputs = tf.keras.Input(shape=(train_X.shape[1],))
        layers.append(inputs)

        x = Dense(best_params['layer_1'], activation=best_params['activation_1'])(layers[-1])
        layers.append(x)

        for l in range(2, best_params['num_layers']+1):
            x = Dense(best_params['units_{}'.format(l)], activation=best_params['activation_{}'.format(l)])(layers[-1])
            layers.append(x)
            x = Dropout(round(best_params["dropout_{}".format(l)], 1))(layers[-1])
            layers.append(x)

        out = Dense(1, activation='sigmoid', name = 'target')(layers[-1])

        model3 = Model(inputs=[inputs], outputs=[out])
                
        learning_rate = best_params['learning_rate']

        metrics=[tf.keras.metrics.AUC(),
                tf.keras.metrics.Accuracy(),
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.FalseNegatives(),
                tf.keras.metrics.FalsePositives()
                ]

        model3.compile(loss='binary_crossentropy', 
                    optimizer=Adam(lr=learning_rate),
                    metrics=metrics)

        model3.fit(train_X,
                train_y,
                validation_split=conf['test_size'],
                shuffle=True,
                batch_size=conf['batch_size'],
                epochs=conf['epochs'],
                verbose=False
                )
        print('summarize updated model3')
        model3.summary()
        model3.save(model3_path)
            

    #chemtsのテストセットで評価
    score = model3.evaluate(chemts_test_X, chemts_test_y, verbose=0)

    results = {}

    for metric, s in zip(model3.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmetrics_fileで指定したファイルに保存
    with open(model3_chemts_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)



    #ChEMBLのテストセットで評価
    score = model3.evaluate(chembl_test_X, chembl_test_y, verbose=0)

    results = {}

    for metric, s in zip(model3.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をmetrics_chembl_fileで指定したファイルに保存
    with open(model3_chembl_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)




    #GDBMedchemのテストセットで評価
    score = model3.evaluate(gdbmed_test_X, gdbmed_test_y, verbose=0)

    results = {}

    for metric, s in zip(model3.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)
    #評価結果をgdbmed_metrics_fileで指定したファイルに保存
    with open(model3_gdbmed_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


    #MERMAIDテストセットの評価
    score = model3.evaluate(mermaid_test_X, mermaid_test_y, verbose=0)
    results = {}

    for metric, s in zip(model3.metrics_names, score):
        if metric in results.keys():
            results[metric].append(s)
        else:
            results[metric] = []
            results[metric].append(s)

    for metric, value in results.items():
        results[metric] = round(np.array(value).mean(),2)

    #評価結果はmodel3_mermaid_metrics_fileで指定したファイルに保存
    with open(model3_mermaid_metrics_file, 'w') as outfile:
        json.dump(str(results), outfile)


#####################################################################################


if __name__ == "__main__":
    do_base()
    do_model1()
    do_model2()
    do_model3()
