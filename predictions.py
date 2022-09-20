import pickle
from functions.modelLib          import CM_pred2,LoadModel
import os
from functions.dimReductionLib   import dimReductionModels
from functions.dataProcessingLib import dataGen
from functions.evalMetricslib    import save_metrics
from functions.utils             import getFolders
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

if __name__ == "__main__":

    mainfolder                     = "/home/diego/Downloads/final_test/all_results_hp/PCA33"
    modelsfolders, drm, dimensions = getFolders(mainfolder)
    Test_Files                     = ["Test","76CAMEO", "MEMS400"]
    dataDir                        = "dataSets"

    for i,modfold in enumerate(modelsfolders):
        print(f"model_{i}",modfold,drm[i],dimensions[i])
        finalfolder=f"{mainfolder}/{modfold}"
        for dstest in Test_Files:
            Lmax = 430
            dr = drm[i]
            rawflag = False
            isnorm = False

            if dr == "RAW":
                diml = [46]
                rawflag = True

            if dr == "AE":
                os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
                diml = [24]
                rawflag = True

            if not rawflag:
                dimReductionModels(X_dir=dataDir, dim=dimensions[i], drmethod=dr, norm=isnorm)

            x_test = dataGen(DR=dr, data_file=f"{dataDir}/{dstest}", Lmin=26, padd=None, Lmax=Lmax, istrain=False, norm=isnorm)
            model, metrics = LoadModel(n_folder=finalfolder, n_model="model.json", n_h5="model.h5")
            print(x_test[0].keys(), len(x_test))
            N_test=len(x_test)
            Yp = CM_pred2(net=model, Np_pred=N_test, Test_Data=x_test, folder_pred=finalfolder, test_name=dstest)
            save_metrics(Np=N_test, Test_dic=x_test, CM_pred=Yp, folder=f"{finalfolder}/{dstest}_res", cutoffs=[0.20, 0.20])











    # modelfolder = f"Model_ResNet64_2D_PCA_epochs_50_dim_24_seed_997_10_08_2022_2036"
    # finalfolder = f"{mainfolder}/{modelfolder}"
    # Test_Files=["76CAMEO","MEMS400"] # test files
    #
    # dataDir = "dataSets"
    # modelsDir = ""
    # Lmax = 430
    # dr = "PCA"
    # rawflag = False
    # isnorm = False
    #
    # if dr == "RAW":
    #     diml = [46]
    #     rawflag = True
    #
    # if dr == "AE":
    #     os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    #     diml = [24]
    #     rawflag = True
    #
    #
    # print(rawflag)
    # if not rawflag:
    #     dimReductionModels(X_dir=dataDir, dim=24, drmethod=dr, norm=isnorm)
    # x_test = dataGen(DR=dr, data_file=f"{dataDir}/{Test_Files[1]}", Lmin=26, padd=None, Lmax=Lmax, istrain=False, norm=isnorm)
    # print(x_test[0].keys(), len(x_test))
    # N_test=len(x_test)
    # for trg in x_test:
    #    print(trg['fseq'].shape)
    # model, metrics = LoadModel(n_folder=finalfolder, n_model="model.json", n_h5="model.h5")
    # Yp = CM_pred2(net=model, Np_pred=N_test, Test_Data=x_test, folder_pred=finalfolder, test_name=Test_Files[1])
    # save_metrics(Np=N_test, Test_dic=x_test, CM_pred=Yp, folder=f"{finalfolder}/{Test_Files[1]}_res", cutoffs=[0.20, 0.20])






    # print(model.summary())
    # with open(f"{dataDir}/{Test_Files[0]}.pickle", 'rb') as tdat:
    #      Test_Data = pickle.load(tdat)
    #print(Test_Data[0].keys())


    # for testData in Test_Files:
    #     print(testData)
    #     with open(f"{folder_test}/{testData}.pickle", 'rb') as tdat:
    #         Test_Data = pickle.load(tdat)
    #     Y = CM_pred2(net=model, Np_pred=len(Test_Data), Test_Data=Test_Data)
    #     print(f"{testData} predictions saving....")
    #     with open(f"{folder_mod}/Pred_{testData}.pickle", "wb") as File:
    #          pickle.dump(Y, File)