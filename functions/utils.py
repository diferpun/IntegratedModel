import numpy as np
import os
import pickle
from functions.evalMetricslib import reduceListMetrics,CalcMCCF1
import re

def Pair_generator(CM_prob, dis=1,issort=False):
    ##########################################################
    # Input
    # CM_prob is a (L,L) with the contact probability
    # dis j-i residue distanse filtter
    # issort  order or not the probabilities
    # Output
    # Pairs 2D index pairs with the highest probability
    #########################################################

    CM_prob=np.ndarray.flatten(CM_prob)
    protL = int(np.sqrt(CM_prob.shape[0]))

    ###### Mesh grid is created all possible pairs #################################
    PP = np.mgrid[0:protL, 0:protL]
    ia = PP[0].reshape((protL * protL, 1))
    ja = PP[1].reshape((protL * protL, 1))
    Paux = np.concatenate((ia, ja), axis=1)
    ind_2D = np.where((Paux[:, 1] - Paux[:, 0]) >= dis)  ## filter by distance

    ###### Pairs and probabies not in order ########################################
    Pairs = Paux[ind_2D[0]]
    ind_1D = Pairs[:, 1] + (Pairs[:, 0] * protL)
    Cont_prob = CM_prob[ind_1D]

    if issort:
        ########## Pairs and probabies in order ####################################
        Ind_ord = np.flipud(np.argsort(Cont_prob))
        Pairs = Pairs[Ind_ord]
        Cont_prob = Cont_prob[Ind_ord]

    return Pairs, Cont_prob


def Save_rr_Format(seq, pairs, cm_p, mod_num, targ_name, fili_name,tprob=0,fold="rr_files"):
    ##########################################################
    # Input
    # seq       protein sequence
    # pairs     ordered pairs from Pair_generator
    # cm_p      contact map probability
    # mod_num   integer
    # targ_name integer
    # fili_name name of the rr file
    # tprop     threshold probability
    # fold      folder where rr are saved
    # Output
    # Pairs 2D index pairs with the highest probability
    #########################################################


    if not os.path.isdir(fold):
         os.makedirs(fold)

    File_rr = open(f"{fold}/{fili_name}.rr", "w+")
    ##### fill the sequence #################
    File_rr.write('%s\n' % "PFRMAT RR")
    NumP = '{:d}'.format(targ_name).zfill(4)
    File_rr.write('TARGET T%s\n' % NumP)
    File_rr.write('MODEL %d\n' % mod_num)
    count_f = 0
    ##### fill the sequence #################
    for j in seq:
        count_f += 1
        File_rr.write('%s' % j)
        if count_f >= 50 and count_f % 50 == 0:
            File_rr.write("\n")
    File_rr.write("\n")
    ######## fill pairs and probability #########
    for i in range(int(pairs.shape[0])):
        ia = pairs[i, 0]
        ja = pairs[i, 1]
        #print(ia+1,ja+1,cm_p[i] > tprob)
        if np.round(cm_p[i],10) > tprob:
           File_rr.write('%d %d %d %d %.4f\n' % (ia + 1, ja + 1, 0, 8, cm_p[i].astype(float)))
    File_rr.write('END')
    File_rr.close()
    return 'ok'

def rr_to_matriz(target,dir_main):
    flag_error = False
    s,L=get_rr_sequence(target,dir_main)
    CM = np.zeros((L, L))
    with open(f"{dir_main}/{target}.rr", 'r') as lines:
         for i,line in enumerate(lines):
             line=line.lstrip()
             if (line[0].isnumeric() or line[0]=='-'):
                 aux=line.split()
                 if int(aux[0]) <= 0 or int(aux[0]) > L or int(aux[1]) <= 0 or int(aux[1]) > L:
                     flag_error = True
                     return CM, flag_error
                 else:
                     #print(aux[0],aux[1],float(aux[-1]))
                     CM[int(aux[0]) - 1, int(aux[1]) - 1] = float(aux[-1])
                     CM[int(aux[1]) - 1, int(aux[0]) - 1] = float(aux[-1])
         # if np.sum(CM)==0:
         #    return CM, flag_error
         return CM, flag_error

def get_rr_sequence(target,dir_main,seq_init=3):
    seq=""
    with open(f"{dir_main}/{target}.rr", 'r') as lines:
        for i, line in enumerate(lines):
             #print(i,line)
             line = line.lstrip()
             if seq_init<=i:
                 aux=line.split(" ")
                 endflag=aux[0].find('END')
                 if((not line[0].isnumeric()) and line[0]!='-' and endflag==-1):
                     line = line.rstrip('\n')
                     seq+=line
    # print(seq)
    # print(len(seq))
    return seq,len(seq)

def getFolders(modfolder):
    folders = [i for i in os.listdir(modfolder) if not os.path.isfile(f"{modfolder}/{i}")]
    components=[]
    drmethods=[]
    for j in folders:
        auxs = j.split("_")
        components.append(int(auxs[7]))
        drmethods.append(re.sub(r'[^a-zA-Z]', '', auxs[3]))
        #print(j,auxs[7],auxs[3])
    return folders,drmethods,components,