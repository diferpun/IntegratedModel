from functions.dimReductionLib   import dimReductionModels
from functions.dataProcessingLib import dataGen
from functions.modelLib          import ResNet_Final,Train,Store_model,CM_pred,LoadModel,graphLossAcc,CM_pred2
from tensorflow.keras.optimizers import Adam
from functions.evalMetricslib    import save_metrics
import tensorflow as tf
import pickle
import numpy as np
import random
import time
import csv
import os


if __name__ == '__main__':
   ########## Important definitions ###################################################################

   #dataDir   = "/home/sanlucp71/dataSets"
   #modelsDir = "/home/sanlucp71"

   dataDir   = "dataSets"
   modelsDir = ""
   Lmax=430
   dr = "AE"
   rawflag=False
   isnorm=False
   ds=["Train","Valid","Test"]
   diml  =[15,27,30]
   seedsl =[1986,1946,159]

   if dr=="RAW":
      diml=[46]
      rawflag=True

   if dr=="AE":
      os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
      diml=[24]
      rawflag=True

   #sdr = np.random.randint(0, 2000, 1)[0]  # seed weights
   #sdr = 0

   ############ Dimension loop #########################################################################
   for sdrindx,i in enumerate(diml):
      sdr=seedsl[sdrindx]
      print("bandera RAW",rawflag,sdr,i)
      ##############Data Processing###################################################################

      if not rawflag:
         dimReductionModels(X_dir=dataDir,dim= i,drmethod=dr,norm=isnorm)

      x_train = dataGen(DR=dr,data_file=f"{dataDir}/{ds[0]}",Lmin=26,padd=10,Lmax=Lmax,istrain=False,norm=isnorm)
      x_valid = dataGen(DR=dr, data_file=f"{dataDir}/{ds[1]}",Lmin=26,padd=10,Lmax=Lmax, istrain=False, norm=isnorm)
      x_test  = dataGen(DR=dr, data_file=f"{dataDir}/{ds[2]}",Lmin=26,padd=None,Lmax=Lmax, istrain=False, norm=isnorm)

      print("dimensions",x_train[0]['fseq'].shape,x_valid[0]['fseq'].shape,x_test[0]['fseq'].shape)
      ####################### Training #######################################################################
      start_time = time.time()

      epch    = 50
      N_train = len(x_train)  # number of trainig     chain proteins
      N_valid = len(x_valid)  # number of validation  chain proteins
      N_test  = len(x_test)
      lr = 0.01  # learning rate values
      opt = Adam(learning_rate=lr)
      model = ResNet_Final(feature1D_shape=i, feature2D_deep=4, reg_params=None, rseed=sdr)
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
      print(model.summary())
      results = Train(model, [N_train,N_valid], Data=[x_train, x_valid], epochs=epch, max_lr=0.1)
      end_time= time.time() - start_time

      # ########### Testing and save the model ##############################################################

      strore_folder=Store_model(k_model=model, Metric_res=results, fold_out=modelsDir,
                                lab=f"ResNet64_2D_{dr}_epochs_{epch}_dim_{i}_seed_{sdr}",dr=dr)
      model, metrics = LoadModel(n_folder=strore_folder,n_model="model.json",n_h5="model.h5")
      graphLossAcc(n_folder=strore_folder, met=metrics)
      Yp = CM_pred2(net=model, Np_pred=N_test, Test_Data=x_test,folder_pred=strore_folder,test_name="Test")  #yp_keys 'name', 'sequence', 'pred'
      save_metrics(Np=N_test, Test_dic=x_test, CM_pred=Yp, folder=f"{strore_folder}/Test_res", cutoffs=[0.20, 0.20])

      # with open(f'{modelsDir}/models/{dr}/time_file.csv', 'a') as f:
      #    # create the csv writer
      #    fieldnames = ['time', 'dim']
      #    writer = csv.DictWriter(f, fieldnames=fieldnames)
      #    writer.writerow({'time': end_time/3600, 'dim': i})