import pickle
import numpy as np
from sklearn.decomposition import PCA,FastICA,FactorAnalysis
from sklearn import random_projection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import os

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

    if norm:
       print("Normalising the data ....")
       X=normData(X)

    if drmethod == "PCA":
       print("dimension: ",dim,"PCA model .....")
       modeldr = PCA(n_components=dim, svd_solver='randomized', whiten=True).fit(X)

    elif drmethod == "RP":
       print("dimension: ",dim,"RP model .....")
       modeldr = random_projection.SparseRandomProjection(n_components=dim, random_state=rand_seed).fit(X)

    elif drmethod == "SVD":
       print("dimension: ",dim,"SVD model .....")
       modeldr = TruncatedSVD(n_components=dim, n_iter=iterations, random_state=rand_seed).fit(X)

    elif drmethod == "ICA":
        modeldr = FastICA(n_components=dim,max_iter=500,random_state=0).fit(X)

    elif drmethod == "FA":
        modeldr = FactorAnalysis(n_components=dim, random_state=0).fit(X)
    else:
        raise Exception("invalid option")

    with open(f"drMethods/{drmethod}_model.pkl", 'wb') as file:
        pickle.dump(modeldr,file)


def dimReduction(X_raw, DR_method,norm=False): #### this function chooses a dimensionality reduction method in order to obtain the embebbed space
    folder_mod="drMethods" # this is the path where the dimensionality reduction models are saved important
    if norm:
       print("Normalising the data")
       norm_model=pickle.load(open(f"{folder_mod}/norm_model.pkl", 'rb'))
       Xs=norm_model.transform(X_raw)
    else:
       print("not Normalising the data")
       Xs=X_raw
    if DR_method == 'RAW':
        X_s = X_raw
        print("RAW data")
    elif DR_method == 'PCA':
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        pca_reload = pickle.load(open(DR_path, 'rb'))
        X_s = pca_reload.transform(Xs)
        print("PCA data")
    elif DR_method == "RP":
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        rp_reload = pickle.load(open(DR_path, 'rb'))
        X_s = rp_reload.transform(Xs)
        print("RP data")
    elif DR_method == "SVD":
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        svd_reload = pickle.load(open(DR_path, 'rb'))
        X_s = svd_reload.transform(Xs)
        print("SVD data")
    elif DR_method == "ICA":
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        ica_reload = pickle.load(open(DR_path, 'rb'))
        X_s = ica_reload.transform(Xs)
        print("ICA data")
    elif DR_method == "FA":
        DR_path = f"{folder_mod}/{DR_method}_model.pkl"
        fa_reload = pickle.load(open(DR_path, 'rb'))
        X_s = fa_reload.transform(Xs)
        print("FA data")
    else:
        raise Exception("invalid option")
    return X_s