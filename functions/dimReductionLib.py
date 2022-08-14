import pickle
import numpy as np
from sklearn.decomposition import PCA,FastICA,FactorAnalysis
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from tensorflow.keras.models import Model
import os
import json
# tensor flow libraries
import tensorflow as tf
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.optimizers import Adam

def normData(X):
    norm_model = StandardScaler().fit(X)
    dataX = norm_model.transform(X)
    with open(f"drMethods/norm_model.pkl", 'wb') as norm_file:
        pickle.dump(norm_model, norm_file)
    return dataX

def dimReductionModels(X_dir,dim,drmethod,norm,rand_seed=0,iterations=7):

    if not os.path.isdir("drMethods"):
        os.mkdir("drMethods")

    with open(f"{X_dir}/Seq_features.pickle", 'rb') as seqData:
        X = pickle.load(seqData)

    print("RAW shape",X.shape)

    if drmethod == "PCA":
       X = normData(X)
       print("dimension: ",dim,"PCA model .....")
       modeldr = PCA(n_components=dim, svd_solver='randomized', whiten=True).fit(X)

    elif drmethod == "RP":
       print("dimension: ",dim,"RP model .....")
       modeldr = random_projection.SparseRandomProjection(n_components=dim, random_state=rand_seed).fit(X)

    elif drmethod == "SVD":
       print("dimension: ",dim,"SVD model .....")
       modeldr = TruncatedSVD(n_components=dim, n_iter=iterations, random_state=rand_seed).fit(X)

    elif drmethod == "ICA":
        print("dimension: ", dim, "ICA model .....")
        modeldr = FastICA(n_components=dim,random_state=0).fit(X)

    elif drmethod == "FA":
        print("dimension: ", dim, "FA model .....")
        modeldr = FactorAnalysis(n_components=dim, random_state=0).fit(X)
    else:
        raise Exception("invalid option")

    with open(f"drMethods/{drmethod}_model.pkl", 'wb') as file:
        pickle.dump(modeldr,file)



def dimReduction(X_raw, DR_method,norm=False): #### this function chooses a dimensionality reduction method in order to obtain the embebbed space
    folder_mod="drMethods" # this is the path where the dimensionality reduction models are saved important

    # if norm:
    #    print("Normalising the data")
    #    norm_model=pickle.load(open(f"{folder_mod}/norm_model.pkl", 'rb'))
    #    Xs=norm_model.transform(X_raw)
    # else:
    #    print("not Normalising the data")
    #    Xs=X_raw

    if DR_method == 'RAW':
        X_s = X_raw
        print("RAW data")

    elif DR_method == 'PCA':
        norm_model = pickle.load(open(f"{folder_mod}/norm_model.pkl", 'rb'))
        Xs = norm_model.transform(X_raw)
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        pca_reload = pickle.load(open(DR_path, 'rb'))
        X_s = pca_reload.transform(Xs)
        print("PCA data")

    elif DR_method == "RP":
        Xs = X_raw
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        rp_reload = pickle.load(open(DR_path, 'rb'))
        X_s = rp_reload.transform(Xs)
        print("RP data")

    elif DR_method == "SVD":
        Xs = X_raw
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        svd_reload = pickle.load(open(DR_path, 'rb'))
        X_s = svd_reload.transform(Xs)
        print("SVD data")

    elif DR_method == "ICA":
        Xs = X_raw
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        ica_reload = pickle.load(open(DR_path, 'rb'))
        X_s = ica_reload.transform(Xs)
        print("ICA data")

    elif DR_method == "FA":
        Xs = X_raw
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        fa_reload = pickle.load(open(DR_path, 'rb'))
        X_s = fa_reload.transform(Xs)
        print("FA data")

    elif DR_method == "AE":
        Xs=X_raw
        print(f"{folder_mod}/AE_hyperparameters.json")
        with open(f"{folder_mod}/AE_hyperparameters.json","r") as hp_file:
             encoder_hp = json.load(hp_file)
        with open(f"{folder_mod}/AE_model.json", "r") as model_json_file:
             encoder_model_json=model_json_file.read()
        encoder_model = model_from_json(encoder_model_json)
        encoder_model.load_weights(f"{folder_mod}/AE_model.h5")
        encoder_model.compile(optimizer=Adam(learning_rate= encoder_hp['learning_rate'])
                              ,loss='mean_squared_error', metrics=['accuracy'])
        encoder = Model(inputs=encoder_model.input, outputs=encoder_model.get_layer("bottleneck").output)
        X_s = encoder.predict(Xs)
        print("encoder shape",X_s.shape)
    else:
        raise Exception("invalid option")
    return X_s