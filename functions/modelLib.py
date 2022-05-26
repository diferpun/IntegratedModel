########## model libraries ################

# Important libraries
import os
import datetime as dt
import numpy as np
import pickle
import gc

# tensor flow libraries
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Flatten, Conv2D,ReLU,Softmax
from tensorflow.keras.initializers import GlorotUniform
import tensorflow as tf
import  matplotlib.pyplot as plt


#################### outer contact ######################################

def outer_concat(X):
    if (X.shape[0] is None):
          L=228
    else:
          L = X.shape[0]           # length of the protein (L)
    v = tf.range(0, L, 1)          # range from 0 to L
    i, j = tf.meshgrid(v, v)       # all posible pairs index
    m = (i+j)/2
    incoming2= X
    m = tf.cast(m, tf.int32)
    out1 = tf.nn.embedding_lookup(incoming2, i)
    out2 = tf.nn.embedding_lookup(incoming2, m)
    out3 = tf.nn.embedding_lookup(incoming2, j)
    #concatante final feature dim together
    out = tf.concat([out3, out2, out1], axis=3)
    return out

def Rotate(X2):
    X2= tf.transpose(X2, perm=[2, 0, 1, 3])
    return X2


################## Residual block function 1D #################################################### 

def ResBlock_Conv1D(X,n_filters,s_filters,v_strides,stage,block,reg_params,rseed=0):
    
    #####################################################################
    # ResBlock_Conv1D Function                                          #                        
    # Inputs                                                            #      
    # X tensor                                                          #         
    # n_filters=[F1,F2] number of filters, deep or number of chanels    #                                                              
    # s_filters=[W1,W2] size of kernels                                 #                                 
    # v_strides=[S1,S2] stride                                          #                          
    # stage= identifier number 1,2,3 etc                                #
    # block= identifier block letter a,b,c etc                          #
    #                                                                   # 
    # Output                                                            #
    #                                                                   #
    # X= Procesed Tensor                                                # 
    #                                                                   #
    #####################################################################

    conv_name_base = 'res1D' + str(stage) + block

    if reg_params is None:
       reg_param=None
    elif reg_params[0]==1:
       reg_param = l1(reg_params[1])
    elif reg_params[0]==2:
       reg_param = l2(reg_params[1])

    # Filter sizes
    F1,F2=n_filters
    W1,W2=s_filters
    S1,S2= v_strides

    if  X.shape[2]==F2:
        X_shortcut = X
    else:
        X_shortcut=Conv1D(filters=F2,kernel_size=1,strides=1
                          ,padding='same',name=conv_name_base+'id',
                          kernel_initializer=GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)

        X_shortcut=BatchNormalization()(X_shortcut)

    # First layer
    X=Conv1D(filters=F1,kernel_size=W1,strides=S1
             ,padding='same',name=conv_name_base+'2b',
             kernel_initializer=GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)
    X=BatchNormalization()(X)
    X=ReLU()(X)

    #Second layer
    X = Conv1D(filters=F2, kernel_size=W2, strides=S2
               ,padding='same', name=conv_name_base+'2c',
               kernel_initializer=GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)

    X = BatchNormalization()(X)

    ## Short cut
    X = layers.Add()([X, X_shortcut])
    X = ReLU()(X)
    return X

def ResBlock_Conv2D(X, n_filters, s_filters,v_strides,stage,block,reg_params,rseed=0):
    
    #####################################################################
    #                                                                   # 
    # ResBlock_Conv2D Function                                          #                        
    # Inputs                                                            #      
    # X tensor                                                          #         
    # n_filters=[F1,F2] number of filters, deep or number of chanels    #                                                              
    # s_filters=[w1,w2] size of kernels                                 #                                 
    # v_strides=[S1,S2] stride                                          #                          
    # stage= identifier number 1,2,3 etc                                #
    # block= identifier block letter a,b,c etc                          #
    #                                                                   # 
    # Output                                                            #
    #                                                                   #
    # X= Procesed Tensor                                                # 
    #                                                                   #
    #####################################################################
    
    conv_name_base = 'res2D' + str(stage) + block

    if reg_params is None:
       reg_param=None
    elif reg_params[0]==1:
       reg_param = l1(reg_params[1])
    elif reg_params[0]==2:
       reg_param = l2(reg_params[1])

    
    # Retrieve Filters
    F1,F2 = n_filters
    w1,w2 = s_filters
    S1,S2 = v_strides
    
    if  X.shape[3]==F2:
        X_shortcut= X
    else:
        X_shortcut=Conv2D(filters=F2,kernel_size=(1,1),strides=(1,1)
                          ,padding='same',name=conv_name_base+'id',
                          kernel_initializer=GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)
        X = BatchNormalization()(X) 
    
    # first component of main path (≈3 lines)
    X = Conv2D(filters = F1, kernel_size = (w1, w1), strides = (S1,S1), padding = 'same'
               ,name = conv_name_base + '2a'
               ,kernel_initializer = GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)

    X = BatchNormalization()(X)
    X = ReLU()(X)
    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (w2,w2), strides = (S2,S2)
               ,padding = 'same', name = conv_name_base + '2b'
               ,kernel_initializer = GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)
    X = BatchNormalization()(X)
    
    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.Add()([X, X_shortcut])
    X = ReLU()(X)
    return X

# Resnet network
def ResNet_Final(feature1D_shape=46,feature2D_deep=4,reg_params=None,rseed=0):

    if reg_params is None:
        reg_param = None
    elif reg_params[0] == 1:
        reg_param = l1(reg_params[1])
    elif reg_params[0] == 2:
        reg_param = l2(reg_params[1])

    ########## Inputs ######################################

    X_input = Input((1, feature1D_shape))
    X_cc= Input((None,1,feature2D_deep))

    ############ Sequence features Resnet 1D ###############
    X = ResBlock_Conv1D(X_input, n_filters=[20,20], s_filters=[17,17], v_strides=[1,1], stage=1, block='a',reg_params=reg_params,rseed=rseed)
    X = ResBlock_Conv1D(X_input, n_filters=[20,20], s_filters=[17,17], v_strides=[1,1], stage=1, block='b',reg_params=reg_params,rseed=rseed)

    ########### Outer concatenation #######################
    X = layers.Lambda(outer_concat)(X)
    X = layers.Concatenate(axis=3)([X,X_cc])
    X = layers.Lambda(Rotate)(X)
    ############ Coevolution features Resnet 2D ##############
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],1,'a',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],1,'b',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],2,'a',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],2,'b',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],3,'a',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],3,'b',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],4,'a',reg_params,rseed=rseed)
    X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],4,'b',reg_params,rseed=rseed)
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],5,'a')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],5,'b')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],6,'a')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],6,'b')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],7,'a')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],7,'b')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],8,'a')
    # X=ResBlock_Conv2D(X,[64,64],[3,3],[1,1],8,'b')
    ################## Salida dnumber of filters in the previous layerel Modelo #################
    X = Conv2D(3,(3, 3),strides=(1, 1),padding='same',kernel_initializer=GlorotUniform(seed=rseed),kernel_regularizer=reg_param)(X)
    X = Softmax()(X)
    mod = Model([X_input,X_cc], X, name="ResNet")
    return mod

def Train(net,Nprot,Data=[None,None],epochs=1,max_lr=None):
    
    #############################################################################################
    #                                                                                             # 
    # Train Function                                                                              #                        
    # Inputs                                                                                      #      
    # - net= Keras model
    # - Nprot=[# of train proteins,# of validation proteins]
    # - Xseq_data= A list with two matrices Train and validation for the sequence features        #                                                                                       
    # Matrix features has the dimession (#aminoacids for all proteins,46)                         #                                                                                            
    # - Xcov_data= A list with two list that contain train and validation Coevolutionary fetures  #                                      
    # Coevolutionary fetures are matrices of (L,L)                                                #                                           
    # - Y_data=A list with two list that contains train and validation labels                     #
    # labels are represented as a flatten 2D contact map (# all posible pairs,3)                  #
    # 
    # Output                                                                                      #
    # Metrics_tv is a list witch contains eight numpy arrays                                      #
    #                                                                                             #
    #  Metrics_tv[0]= Training Loss of each protein in the last epoch                             #
    #  Metrics_tv[1]= Training accuracy of each protein in the last epoch                         #
    #  Metrics_tv[2]= average of Training loss in each epoch                                      #
    #  Metrics_tv[3]= average of Training accuracy in each epoch                                  #
    #  Metrics_tv[4]= Validation Loss of each protein in the last epoch                           #                                
    #  Metrics_tv[5]= Validation accuracy of each protein in the last epoch                       #                                                                                                                                      
    #  Metrics_tv[6]= average of validation loss in each epoch                                    #
    #  Metrics_tv[7]= average of validation accuracy loss in each epoch                           # 
    #                                                                                             #
    ###############################################################################################                                                                                                                                                                   

    ####### the Dataset is store in separate variables for train and validation ##############
    
    Np_t,Np_v=Nprot # number of proteins
    Data_train,Data_val=Data # training and validation sequencial features

    ############# vectors are crated in order to save the loss and accuracy metrics ##########
    
    losst_p=np.zeros((1,Np_t))
    acct_p=np.zeros((1,Np_t))
    losst_epch = np.zeros((1,epochs))
    acct_epch = np.zeros((1,epochs))
    lossv_epch = np.zeros((1,epochs))
    accv_epch = np.zeros((1,epochs))
    Metrics_tv={}
    #########################################################################################
    dynamic_lr = tf.keras.backend.get_value(net.optimizer.lr)
    # if max_lr is not None:
    #    delta_lr = (max_lr - dynamic_lr) / (int(epochs / 2))

    ############### the train proccess star #################################################
    
    for i in range(epochs):
        for j in range (Np_t):
            Xs =tf.convert_to_tensor(np.reshape(Data_train[j]['fseq'], (Data_train[j]['fseq'].shape[0],1,Data_train[j]['fseq'].shape[1])))
            Xc =tf.convert_to_tensor(np.reshape(Data_train[j]['fcoev'],(Data_train[j]['fcoev'].shape[0],Data_train[j]['fcoev'].shape[1],1,Data_train[j]['fcoev'].shape[2])))
            Y =tf.convert_to_tensor(np.reshape(Data_train[j]['label'],(1,Data_train[j]['label'].shape[0],Data_train[j]['label'].shape[1],Data_train[j]['label'].shape[2])))
            hist = net.train_on_batch([Xs,Xc],Y)
            print("epoch: ", i, "length: ", Xs.shape[0], "# Protein : ", j, "loss: ", round(hist[0], 3), "acc: ",round(hist[1], 2),"learning rate",dynamic_lr)
            losst_p[0,j]=hist[0]
            acct_p[0,j]=hist[1]
        tf.keras.backend.clear_session()
        gc.collect()
        losst_epch[0,i]=np.median(losst_p, axis=1).item()
        acct_epch[0,i]=np.median(acct_p, axis=1).item()
        vl_prot, vac_prot,vl_m,va_m=Val(net,Np_v,Data_val)
        lossv_epch[0,i]=vl_m
        accv_epch[0,i]=va_m
        print("loss_tr: ",losst_epch[0,i],"acc__tr: ",acct_epch[0,i],"loss_val", vl_m, "acc_val: ", va_m)

        # if max_lr is not None:
        #     if i<int(epochs/2):
        #        new_lr=np.around(dynamic_lr+delta_lr,4)
        #     else:
        #        new_lr=np.around(dynamic_lr-delta_lr,4)
        #     tf.keras.backend.set_value(net.optimizer.lr,new_lr)
        #     dynamic_lr = tf.keras.backend.get_value(net.optimizer.lr)

    # Metrics train and validation Metrics tv
    Metrics_tv['Tloss_per_protein']=losst_p[0]
    Metrics_tv['Tacc_per_protein']=acct_p[0]
    Metrics_tv['Tloss_per_epoch']=losst_epch[0]
    Metrics_tv['Tacc_per_epoch']=acct_epch[0]
    Metrics_tv['Vloss_per_protein']=vl_prot[0]
    Metrics_tv['Vacc_per_protein']=vac_prot[0]
    Metrics_tv['Vloss_per_epoch']=lossv_epch[0]
    Metrics_tv['Vacc_per_epoch']=accv_epch[0]

    return Metrics_tv

def Val(net_val,Np_val,Data_val):
    
    ###############################################################################################
    #                                                                                             # 
    # Val Function                                                                                #                        
    # Inputs                                                                                      #      
    # - net= Keras model                                                                          #
    # - Np_val= Number proteins for validation                                                    #
    # - Ra_val= Protein Range is a matrix of (#Proteins,2). Ex: pr1 0-100, pr2 100-150            #                                                                                      
    # - Xs_val= np array with sequential features (#aminoacids for all proteins,26)               #                            
    # - Xc_val= A list that contain train and validation Coevolutionary fetures                   #                                      
    # Coevolutionary fetures are matrices of (L,L)                                                #
    # - Y_val=labels are represented as a flatten 2D contact map (# all posible pairs,3)          #
    #                                                                                             #
    #                                                                                             #
    # Output                                                                                      #
    #  Metrics_tv is a list witch contains eight numpy arrays                                     #
    #                                                                                             #                                                                                                                                              
    #  val_loss= Validation Loss of each protein                                                  #                                
    #  val_acc= Validation accuracy of each protein                                               #                                                                                                                                                          
    #  val_lm= averace of validation loss                                                         #
    #  Metrics_tv= averace of validation accuracy                                                 # 
    #                                                                                             #
    ###############################################################################################
    
    print("Validation ................")
    val_loss=np.zeros((1,Np_val))
    val_acc=np.zeros((1,Np_val))      

    for i in range(Np_val):
        Xs_val = tf.convert_to_tensor(np.reshape(Data_val[i]['fseq'], (Data_val[i]['fseq'].shape[0], 1, Data_val[i]['fseq'].shape[1])))
        Xc_val = tf.convert_to_tensor(np.reshape(Data_val[i]['fcoev'], (Data_val[i]['fcoev'].shape[0], Data_val[i]['fcoev'].shape[1], 1, Data_val[i]['fcoev'].shape[2])))
        Y_val = tf.convert_to_tensor(np.reshape(Data_val[i]['label'],(1,Data_val[i]['label'].shape[0],Data_val[i]['label'].shape[1],Data_val[i]['label'].shape[2])))
        val_met=net_val.test_on_batch([Xs_val,Xc_val],Y_val)
        val_loss[0,i] = val_met[0]
        val_acc[0,i]  = val_met[1]
    val_lm = np.median(val_loss, axis=1)
    val_am = np.median(val_acc, axis=1)
    val_lm = val_lm.item()
    val_am = val_am.item()
    return val_loss,val_acc,val_lm,val_am


def CM_pred(net,Np_pred,Test_Data,folder_pred="",test_name=""):
    
    #############################################################################
    # inputs 
    # net_pred trained model in keras
    # Np_pred number of proteins to predict
    # Test_Data a dic with fseq (sequencial features) and fcoev (coevolition features)
    # Output
    # Y_pred A list with the probabilities of contact (L,L,3)
    ############################################################################

    Y_pred=[]
    for i in range(Np_pred):
        Xs_test = tf.convert_to_tensor(np.reshape(Test_Data[i]['fseq'],
                 (Test_Data[i]['fseq'].shape[0],1,Test_Data[i]['fseq'].shape[1])))
        Xc_test = tf.convert_to_tensor(np.reshape(Test_Data[i]['fcoev'],
                 (Test_Data[i]['fcoev'].shape[0],Test_Data[i]['fcoev'].shape[1],1,
                  Test_Data[i]['fcoev'].shape[2])))
        Y=net.predict_on_batch([Xs_test,Xc_test])
        Y=np.reshape(Y,(Y.shape[1],Y.shape[2],Y.shape[3]))
        print("name: ",Test_Data[i]['name'],"length: ", Y.shape[0], "# Protein : ", i)
        Y_pred.append(Y[:,:,1])

    with open(f"{folder_pred}/Pred_{test_name}.pickle", "wb") as File:
             pickle.dump(Y_pred,File)

    return Y_pred


def CM_pred2(net, Np_pred, Test_Data, folder_pred="", test_name=""):
    #############################################################################
    # inputs
    # net_pred trained model in keras
    # Np_pred number of proteins to predict
    # Test_Data a dic with fseq (sequencial features) and fcoev (coevolition features)
    # Output
    # Y_pred A list with the probabilities of contact (L,L,3)
    ############################################################################

    Y_pred =  []


    for i in range(Np_pred):
        Xs_test = tf.convert_to_tensor(np.reshape(Test_Data[i]['fseq'],
                                                  (Test_Data[i]['fseq'].shape[0], 1, Test_Data[i]['fseq'].shape[1])))
        Xc_test = tf.convert_to_tensor(np.reshape(Test_Data[i]['fcoev'],
                                                  (Test_Data[i]['fcoev'].shape[0], Test_Data[i]['fcoev'].shape[1], 1,
                                                   Test_Data[i]['fcoev'].shape[2])))

        Y = net.predict_on_batch([Xs_test, Xc_test])
        Y = np.reshape(Y, (Y.shape[1], Y.shape[2], Y.shape[3]))
        print("name: ", Test_Data[i]['name'], "length: ", Y.shape[0], "# Protein : ", i)
        Y_dic = {"name":Test_Data[i]['name'] ,"sequence": Test_Data[i]['sequence']
                ,"pred":Y[:, :, 1],"label":Test_Data[i]['label'][:, :, 1]}

        Y_pred.append(Y_dic)
    print(len(Y_pred))

    with open(f"{folder_pred}/Pred_{test_name}.pickle", "wb") as File:
        pickle.dump(Y_pred, File)
    return Y_pred


def Store_model(k_model, Metric_res, fold_out="", lab="CM",dr=""):
    ########################################################################
    # Store_model this function store the model when the training is done  #
    # INPUTS                                                               #
    # k.model Keras model                                                  #
    # Metrics_res metrics of the model                                     #
    # folder the path where the files will be saved                        #
    ########################################################################

    aux="/"
    date = dt.datetime.now().strftime('%d_%m_%Y_%H%M')

    if fold_out =="" or fold_out is None:
       aux=""

    folder = f"{fold_out}{aux}models/{dr}/Model_{lab}_{date}"

    if not os.path.isdir(f"{folder}"):
        os.makedirs(f"{folder}")

    model_json = k_model.to_json()
    with open(f"{folder}/model.json", "w") as json_file:
        json_file.write(model_json)
    k_model.save_weights(f"{folder}/model.h5")
    print("Saved model to disk")
    with open(f"{folder}/metrics.pickle", 'wb') as f:
        pickle.dump(Metric_res, f)
    return folder

def LoadModel(n_folder, n_model, n_h5):
    ########################################################################
    # Load_Model this function load the model inside a specifc folder      #
    # INPUTS                                                               #
    # n_folder the path of the saved model                                 #
    # n_model name of json file                                            #
    # n_h5    name of h5 file                                              #
    #  OUTPUTS                                                             #
    # loaded_model model from the folder                                   #
    # d_metrics   output metrics for analisys                              #
    #                                                                      #
    ########################################################################

    json_file = open(os.path.join(n_folder, n_model), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights
    loaded_model.load_weights(os.path.join(n_folder, n_h5))
    print("Loaded model from disk")
    # load metrics
    with open(f"{n_folder}/metrics.pickle", 'rb') as metric:
        d_metrics = pickle.load(metric)
    return loaded_model, d_metrics

def graphLossAcc(n_folder,met):
    with open(f"{n_folder}/metrics.pickle", "rb") as Data:
        Metrics = pickle.load(Data)
    #print(Metrics.keys())
    plt.figure(1)
    plt.plot(met['Tloss_per_epoch'])
    plt.plot(met['Vloss_per_epoch'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    #plt.show()
    plt.savefig(f"{n_folder}/loss.png")
    plt.close()
    plt.figure(2)
    plt.plot(met['Tacc_per_epoch'])
    plt.plot(met['Vacc_per_epoch'])
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    #plt.show()
    plt.savefig(f"{n_folder}/acc.png")
    plt.close()


