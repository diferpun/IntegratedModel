import pickle
from functions.modelLib          import CM_pred2,LoadModel
import os
from functions.dimReductionLib   import dimReductionModels
from functions.dataProcessingLib import dataGen
from functions.evalMetricslib2    import save_metrics
from functions.utils             import getFolders
import re


if __name__ == "__main__":

    mainfolder = "./models"   #"/home/diego/Downloads/models2/all"  # "/home/diego/Downloads/final_test/all_results_hp/AE"
    modelsfolders, drm, dimensions = getFolders(mainfolder)
    Test_Files = ["Test", "76CAMEO", "MEMS400"]
    dataDir = "dataSets"

    for i,modfold in enumerate(modelsfolders):
        for dstest in Test_Files:
            with open(f"{mainfolder}/{modfold}/Pred_{dstest}.pickle", 'rb') as Pbatch2:
                pred1 = pickle.load(Pbatch2)
            N_test=len(pred1)
            save_metrics(Np=N_test, Test_dic=pred1, CM_pred=pred1,
                         folder=f"{mainfolder}/{modfold}/{dstest}_res", cutoffs=[0.20, 0.20])


