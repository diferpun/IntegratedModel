import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_rows", 1000, "display.max_columns", 1000)
from os import  path
import os

def save_metrics(Np,Test_dic,CM_pred,folder,cutoffs=[0,0.5]):

    header_red =["TopPrec_L","F1_L","Prec_L","Rec_L",
                 "TopPrec_L\\2","F1_L\\2", "Prec_L\\2","Rec_L\\2",
                 "TopPrec_L\\5","F1_L\\5", "Prec_L\\5", "Rec_L\\5"]
    header_full=["MCC","F1","Prec","Rec"]
    ranges_red  = ["ExtraLong","Long","MediumLong","Medium","Short"]
    ranges_full = ["ExtraLong","Long","MediumLong","MediumLongShort"]

    cutoff_red,cutoff_full=cutoffs

    trg=[]
    Lp=[]

    for i in range(Np):
        trg.append(Test_dic[i]['name'])
        Lp.append(len(Test_dic[i]['sequence']))

    X_red=mergeReduceList(Np=Np,CM_pred=CM_pred,Test_Data=Test_dic,cutoff=cutoff_red)
    X_full=mergeFullList(Np=Np,CM_pred=CM_pred,Test_Data=Test_dic,cutoff=cutoff_full)

    # print(X_red.shape)
    # print(X_full.shape)
    # print(X_full[0,0,:])

    folder_red=f"{folder}/red_met/"
    folder_full=f"{folder}/full_met/"

    if not path.exists(folder_red):
        os.makedirs(folder_red)

    if not path.exists(folder_full):
        os.makedirs(folder_full)

    for j,rg in enumerate(ranges_red):
        reduce_df = pd.DataFrame(data=X_red[:,:,j],columns=header_red)
        reduce_df[reduce_df <= 0] = 0
        reduce_df.insert(0, "Target",trg)
        reduce_df.insert(1, "Length", Lp)
        reduce_df.to_csv(f"{folder_red}/{rg}.csv", decimal='.', sep=';', float_format='%.3f')

    for j, rg in enumerate(ranges_full):
        full_df = pd.DataFrame(data=X_full[:,j,:],columns=header_full)
        full_df .loc[full_df ['F1']<= 0,'F1'] = 0
        full_df .loc[full_df ['Prec']<= 0,'Prec'] = 0
        full_df .loc[full_df ['Rec']<= 0,'Rec'] = 0
        full_df.insert(0, "Target", trg)
        full_df.insert(1, "Length", Lp)
        full_df.to_csv(f"{folder_full}/{rg}.csv", decimal='.', sep=';', float_format='%.3f')

    with open(f"{folder}/cutoff.txt", "w") as the_file:
        the_file.write(f"Reduce List cutoff {str(cutoff_red)}\n")
        the_file.write(f"Full   List cutoff {str(cutoff_full)}\n")

def F1(TP, FP, TN, FN):
    epsilon = 0.000001
    precision = TP * 1. / (TP + FP + epsilon)
    recall = TP * 1. / (TP + FN + epsilon)
    F1 = (2. * precision * recall) / (precision + recall + epsilon)
    return F1, precision, recall

def MCC(TP, FP, TN, FN):
    epsilon = 0.000001
    MCC = (TP * TN - FP * FN) / np.sqrt(epsilon + 1.0 * (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    #print(TP,TN,FN,FP)
    return MCC

##calculate MCC of a predicted contact matrix using a given score cutoff
##here we consider three cases: long-range contacts, long + medium-range contacts,
# long + medium- + short-range contacts

def CalcMCCF1(pred=None, truth=None, probCutoff=0):
    seqLen = pred.shape[0]   ###### first dimension of the contact map #######
    seqLen2 = pred.shape[1]  ###### second dimension of the contact map #######
    pred_binary = (pred > probCutoff) #### loggits greater of a cutoff matrix of zero and ones ########
    truth_binary = (0 < truth)        ####  contact map greater than cero matrix of zero and ones #######
    pred_truth = pred_binary * 2 + truth_binary #### adition of 2 times predictions and truth contacts #####
    mask_ER = np.triu_indices(seqLen,48,m=seqLen2) ######### indices de de la diagonal mayor a la 42 ######
    mask_LR = np.triu_indices(seqLen,24,m=seqLen2) ######### indices de de la diagonal mayor a la 24 ######
    mask_MLR = np.triu_indices(seqLen,12,m=seqLen2) ######### indices de de la diagonal mayor a la 12 ######
    mask_SMLR = np.triu_indices(seqLen,6,m=seqLen2) ######### indices de de la diagonal mayor a la 6 ######
    metrics = []

    for mask in [mask_ER,mask_LR,mask_MLR,mask_SMLR]:
        res = pred_truth[mask]   ##### extract only the information inside de masks ##########
        total = res.shape[0]
        count = np.bincount(res, minlength=4) #### returns array frecuency inside a minlenght
        assert (total == np.sum(count))
        ## pred=0, truth=0   true negative
        TN = count[0]
        ## pred=0, truth=1   false negative
        FN = count[1]
        ## pred=1, truth=0   false positive
        FP = count[2]
        ## pred=1, truth=1   true positive
        TP = count[3]
        MCC_m = MCC(TP, FP, TN, FN) #### this function returns matthews correlation coefficient
        F1_score, precision, recall = F1(TP, FP, TN, FN) ####### his function returns matthews F1_socre, precision and recall
        metrics.extend([MCC_m,F1_score*100,precision*100,recall*100]) ############## this function concatenates the metrics for the contact ranges
    return np.array(metrics) ##### returns the numpy array of the list


def reduceListMetrics(pred=None, truth=None, ratio=[1, 0.5, 0.2],probCutoff=0):
    # input

    pred_truth = np.dstack((pred, truth))     # 3D concatenation
    M1s = np.ones_like(truth, dtype=np.int8)  # matrix full of ones with the shape L,L
    mask_ER = np.triu(M1s, 48)        # mask for Extra long range diagonal >48
    mask_LR = np.triu(M1s, 24)        # mask for long range  diagonal >24
    mask_MLR = np.triu(M1s, 12)       # mask for long + medium range 12<diagonal
    mask_SMLR = np.triu(M1s, 6)       # mask short  short + medium + long 6< diagonal
    mask_MR = mask_MLR - mask_LR      # mask for medium range  12<=diagonal<24
    mask_SR = mask_SMLR - mask_MLR    # mask for short range 6<=diagonal<12
    seqLen = pred.shape[0]            # sequence length

    # calculation of all contacts
    cont_types =["ELR","LR","LMR","MR","SR"]
    red_met_dic={}

    for i,mask in enumerate([mask_ER,mask_LR,mask_MLR,mask_MR,mask_SR]):
        red_met = []
        res = pred_truth[mask.nonzero()]                            # retrieve the data inside the mask
        res_sorted = res[(-res[:, 0]).argsort()]                    # sort the data in descendent order
        res_sorted = np.where(res_sorted<0,0,res_sorted)            # delete the data with -1
        res_sorted= np.where(res_sorted>probCutoff,1,res_sorted)    #
        res_sorted= np.where(res_sorted<=probCutoff,0,res_sorted)
        cm_red_real=np.sum(res_sorted[:,1])
        for r in ratio:
            numTops = int(seqLen * r) # L/k
            numTops = min(numTops,res_sorted.shape[0])
            aux=res_sorted[:numTops]
            tp = np.sum(np.logical_and(aux[:,0]== 1,aux[:,1] == 1))
            fp = np.sum(np.logical_and(aux[:,0]== 1,aux[:,1] == 0))
            Recall=tp/(cm_red_real+0.0001)
            Precision=tp/(tp+fp+0.00001)
            F1_score =(2*Precision*Recall)/(Precision+Recall+0.00001)
            topPre=res_sorted[:numTops,1].sum()/(numTops+0.00001)
            red_met.extend([topPre*100,F1_score*100,Precision * 100,Recall * 100])
        red_met=np.array(red_met)
        red_met_dic[cont_types[i]]=red_met
    return red_met_dic

def mergeReduceList(Np,CM_pred,Test_Data,cutoff=0):
    #########################################
    # inputs
    # Np         number of proteins
    # CM_pred    contact maps predictions
    # Test_Data  real contact maps
    # Outputs
    # red_met    a 3D matrix
    # first dimension  (Np): is the number of proteins
    # second dimension (12): is the concatenation of Toprecision, F1-score, Precision and recall
    # For list sizes L, L/2, L/5
    # third dimension (5)
    # mask_ER,mask_LR,mask_MLR,mask_MR,mask_SR

    red_met = np.zeros((Np,12,5))  ### inicialice matriz of matrix ("be carefoul of 12 dimension 4x3")
    for i in range(Np):
        Y_true = Test_Data[i]['label'][:, :, 1] # real contact map
        Y_pred = CM_pred[i] # predicted contact map
        reduce_metrics_dic = reduceListMetrics(pred=Y_pred, truth=Y_true,probCutoff=cutoff) # metrics calculation
        for j,cmtype in enumerate(reduce_metrics_dic.keys()): # contact ranges mask_ER,mask_LR,mask_MLR,mask_MR,mask_SR inside thidr dimesion
             red_met [i, :, j] = reduce_metrics_dic[cmtype]   # saves the information of all proteins in a three dimensional matrix
    return red_met

def mergeFullList(Np,CM_pred,Test_Data,cutoff=0):
    #########################################
    # inputs
    # Np         number of proteins
    # CM_pred    contact maps predictions
    # Test_Data  real contact maps
    # Outputs
    # X_m   a 3D matrix
    # first dimension  (Np): is the number of proteins
    # second dimension (4): is the metrics MCC, F1-score, Precision and recall
    # third dimension (4):  Extra long ,Long ,Medium long, short+medium+long

    full_met = np.zeros((Np, 16)) #### the dimension can be varible
    for i in range(Np):
        Y_true = Test_Data[i]['label'][:, :, 1] # real contact map
        Y_pred = CM_pred[i] # predicted contact map
        full_met[i,:] = CalcMCCF1(pred=Y_pred, truth=Y_true,probCutoff=cutoff) # metrics calculation
        #print(full_met)
    X_m=full_met.reshape((Np,4,4)) # reshape in order to separate the contacs ranges
    return X_m

def logit_histogram(Y_pred):
    print(Y_pred.shape)
    M1s = np.ones_like(Y_pred, dtype=np.int8)
    mask = np.triu(M1s, 6)
    cm_ud = Y_pred[mask.nonzero()]
    print(cm_ud.shape)
    plt.hist(cm_ud,bins=50)
    plt.show()

