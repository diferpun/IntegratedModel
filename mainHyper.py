from functions.modelLib          import randomSearch
from functions.dimReductionLib   import dimReductionModels
from functions.dataProcessingLib import dataGen
from functions.modelLib          import ResNet_Final,Train,Store_model,CM_pred,LoadModel,graphLossAcc,CM_pred2
from tensorflow.keras.optimizers import Adam,Nadam,Adamax
from functions.modelLib          import ResNet_Final2
from functions.evalMetricslib    import save_metrics
import time
import os
import csv
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



if __name__ == '__main__':
   ########## important paths ###################################################################
   #hyper_grid_random = randomSearch(20)

   hyper_grid_random=[(Adam, 0.001,	1,1e-05)]
   dataDir    = "/home/andres_david_0496/dataSets"
   modelsDir  = "/home/andres_david_0496"

   #dataDir   = "dataSets"
   #modelsDir = "."

   Lmax=430
   dr = "RAW"
   rawflag=False
   isnorm=False
   ds=["Train","Valid","Test"]
   dim  = 24
   srd  = 808
   epch = 50 ################################## importante #########################################

   if dr=="RAW":
      dim=46
      rawflag=True

   if dr=="AE":
      os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
      dim=24
      rawflag=True

   ############ Dimension loop #########################################################################
   for i,hyper_comb  in enumerate(hyper_grid_random):

      print("bandera RAW",rawflag,srd,i)
      ##############Data Processing###################################################################
      start_time = time.time()
      if not rawflag:
         dimReductionModels(X_dir=dataDir,dim=dim,drmethod=dr,norm=isnorm)

      x_train = dataGen(DR=dr,data_file=f"{dataDir}/{ds[0]}",Lmin=26,padd=10,Lmax=Lmax,istrain=False,norm=isnorm)
      x_valid = dataGen(DR=dr, data_file=f"{dataDir}/{ds[1]}",Lmin=26,padd=10,Lmax=Lmax, istrain=False, norm=isnorm)
      x_test  = dataGen(DR=dr, data_file=f"{dataDir}/{ds[2]}",Lmin=26,padd=None,Lmax=Lmax, istrain=False, norm=isnorm)

      print("dimensions",x_train[0]['fseq'].shape,x_valid[0]['fseq'].shape,x_test[0]['fseq'].shape)
      ####################### Training #######################################################################

      N_train =len(x_train)  # number of trainig     chain proteins
      N_valid =len(x_valid)  # number of validation  chain proteins
      N_test  =len(x_test)
      lr =   hyper_comb[1]  #0.01  # learning rate values
      opt =  hyper_comb[0](learning_rate=lr)  #Adam(learning_rate=lr)
      model = ResNet_Final(feature1D_shape=dim, feature2D_deep=4, reg_params=[hyper_comb[2],hyper_comb[3]], rseed=srd)
      model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
      print(model.summary())
      results = Train(model, [N_train,N_valid], Data=[x_train, x_valid], epochs=epch, max_lr=0.1)
      end_time= time.time() - start_time

      # ########### Testing and save the model ##############################################################

      strore_folder=Store_model(k_model=model, Metric_res=results, fold_out=modelsDir,
                                lab=f"ResNet64_2D_{dr}{i}_epochs_{epch}_dim_{dim}_seed_{srd}",dr=dr)
      model, metrics = LoadModel(n_folder=strore_folder,n_model="model.json",n_h5="model.h5")
      graphLossAcc(n_folder=strore_folder, met=metrics)

      #Yp = CM_pred2(net=model, Np_pred=N_test, Test_Data=x_test,folder_pred=strore_folder,test_name="Test")  #yp_keys 'name', 'sequence', 'pred'
      #save_metrics(Np=N_test, Test_dic=x_test, CM_pred=Yp, folder=f"{strore_folder}/Test_res", cutoffs=[0.20, 0.20])

      with open(f'{strore_folder}/hyperparameters.csv', 'a') as f:
         # create the csv writer
         hyper_comb_str = str(hyper_comb)[1:-1].split(",")
         hyper_comb_str[0] = hyper_comb_str[0].split(".")[-1].replace(">", "")
         hyper_comb_str[0] = hyper_comb_str[0].replace("'", "")
         hyper_comb_dic = {"optimizer": hyper_comb_str[0], "learning_rate": hyper_comb_str[1],
                           "regularizer": f"L{hyper_comb_str[2]}", "regularizer_value":hyper_comb_str[3]}
         fieldnames = list(hyper_comb_dic.keys())
         writer = csv.DictWriter(f, fieldnames=fieldnames)
         writer.writeheader()
         writer.writerow(hyper_comb_dic)

      with open(f'{modelsDir}/models/{dr}/time_file.csv', 'a') as f:
         # create the csv writer
         fieldnames = ['time', 'dim']
         writer = csv.DictWriter(f, fieldnames=fieldnames)
         if i == 0:
            writer.writeheader()
         writer.writerow({'time': end_time/3600, 'dim': dim})
