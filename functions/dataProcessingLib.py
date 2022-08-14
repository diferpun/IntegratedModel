from   tensorflow.keras.utils import to_categorical
import numpy as np
import pickle
import math
from   .dimReductionLib import dimReduction
import os

def Transform_and_save (ldic,folder_out):  # this function decode the information in python 3 format
    n=int(len(ldic))
    ks=list(ldic[0].keys())
    Raptor_D = []
    for i in range(n):               #  this loop runs for each elements in a list of dics
        dic = {}
        for j in range(len(ks)):     # this loop runs for each key in the dic
            if isinstance(ldic[i][ks[j]],bytes):  # this line controlls if the content is a bytes element
                dic[ks[j].decode('utf-8')] = ldic[i][ks[j]].decode('utf-8') # decode the key and the content
            else:
                dic[ks[j].decode('utf-8')] = ldic[i][ks[j]] # decode only the key
        Raptor_D.append(dic) # append dicctionary into a new list
    with open(folder_out+'.pickle', 'wb') as data_file: # save the new dictionary
        pickle.dump(Raptor_D,data_file)

def Seqs_information(ldic):
    N = len(ldic)           # the number of elements in the list
    dk='sequence'           # key sequence
    dk1='name'              # key name
    Seq_r=np.zeros((N,2))   # initialize the matrix ranges
    Seq_T=[]                # initialize the sequence list
    names=[]                # initialize the names(PDB_id+chain)
    for i in range(N):      # this loop runs for each list item
        if i == 0:
            Seq_r[i,0]=int(0)
            Seq_r[i,1]=int(len(ldic[i][dk]))
        else:
            Seq_r[i,0]=int(Seq_r[i-1,1])
            Seq_r[i,1]=int(Seq_r[i,0]+len(ldic[i][dk]))
        Seq_T.append(ldic[i][dk])
        names.append(ldic[i][dk1])
    ranges=np.int32(Seq_r)
    return names,Seq_T,ranges  # this function return a lists names, sequences and numpy arras Rangos

def Sequence_Features(ldic,DR_method=None,istrain=False,norm=False):

    n_p=len(ldic)                         # number of elements inside the list
    Ndat = 46                             # number of features
    names,SeqL,Ran=Seqs_information(ldic) # sequence information names, sequence and ranges
    Mdat = Ran[-1, 1]                     # number of rows for the unified matrix
    X_dat=np.zeros((Mdat,Ndat))           # initialize the data matrix
    for i in range(n_p):                  # this foor loop get each dctionary in the list
        aux=np.append(ldic[i]['PSSM'],ldic[i]['PSFM'], axis=1) # numpy aux matrix concatenate PSSM and PSFM features
        aux=np.append(aux,ldic[i]['SS3'],axis=1)               # numpy aux matrix concatenate SS3 features
        aux=np.append(aux,ldic[i]['ACC'],axis=1)               # numpy aux matrix concatenate ACC features
        X_dat[Ran[i,0]:Ran[i,1],:]=aux                         # aux matrix is saved in X_dat matrix
    # print(X_dat.shape) (1402007, 46)

    print("save Xseq",istrain and DR_method=="RAW")

    # if istrain and DR_method=="RAW":  # if the the input is the trainig dataset the sequence matrix is saved
    #     with open(f"dataSets/Seq_features.pickle", 'wb') as data_seq:  # this line of code saves the sequence feature matrix
    #        pickle.dump(X_dat,data_seq)
    #     print('save',X_dat.shape)

    X_dat=dimReduction(X_dat,DR_method,norm)
    print(f"{DR_method}",X_dat.shape[1])
    return names,SeqL,Ran,X_dat # this function returns names, sequence, ranges, and features matrix


def coevolution_features(ldic):
    n_p=len(ldic)         # number of elements inside the list of dics
    X_cv=[]               # list wich contains the coevolution features
    for i in range(n_p):  #  this foor loop get each dctionary in the list
        Xcv1=ldic[i]['ccmpredZ'] # this variable saves ccmpredZ coevolution features
        Xcv2=ldic[i]['OtherPairs'] # this variable saves 'OtherPairs' coevolution features
        XcvT=np.dstack((Xcv1,Xcv2)) # this line concatenates the two previous variables
        X_cv.append(XcvT) # this line append the information in a list
    return X_cv     # this function return the coevolution feattures

def label_cm(ldic):
    n_p=len(ldic)  # number of elements inside the list of dics
    Y_cm=[]        # list wich contains contact maps matrix
    for i in range(n_p):
        Y_aux=ldic[i]['contactMatrix']  # returns the contact matrix
        Y_aux=np.where(Y_aux==-1,2,Y_aux) # replace -1 value to 2
        #print(Y_aux.shape)
        Y_aux=to_categorical(Y_aux) # transorm one hot matrix
        if Y_aux.shape[2]<3: # some proteins has not 3 classes
            T_z=np.zeros((Y_aux.shape[0],Y_aux.shape[1],1))
            Y_aux=np.concatenate((Y_aux,T_z), axis=2) # add a matrix full of zeros and concatenate
        Y_cm.append(np.int8(Y_aux)) # append in one list the matrices
    return Y_cm

def Extract_features(file,DR_method=None,istrain=False,norm=False):  # this function extrac feathures from the Raptor-X-contact transformed data
    with open(file , 'rb') as f:
         Raptor_dic = pickle.load(f)
    names,seq,ranges,Xs=Sequence_Features(Raptor_dic,DR_method,istrain,norm)
    Xc=coevolution_features(Raptor_dic)
    Ycm=label_cm(Raptor_dic)
    Xldic=[]
    L=ranges[:,1]-ranges[:,0]
    for i in range(len(names)):
      Daux={'name':names[i],'sequence':seq[i],'fseq':Xs[ranges[i,0]:ranges[i,1]],'fcoev': Xc[i],'label':Ycm[i]}
      Xldic.append(Daux)
    return Xldic,L

def Prot_padding(ldic,b_round): # this function make a padding to obtain shape data multiples of b_round
    num_p=len(ldic) # number of proteins
    ldicp=[]   # list of dictionaries
    for i in range(num_p):  # this for loop modfy each protein
        l=len(ldic[i]['sequence']) # size of the sequence
        b=int(math.ceil(l/b_round) * b_round) # this line approximate the size to upper multiples of b_round
        padd=int(b-l)  # this line calulates the size of the padding
        Xs=np.pad(ldic[i]['fseq'], [(0, padd), (0, 0)])   ########### the padding is made for the sequence features
        Xc=np.pad(ldic[i]['fcoev'],[(0, padd), (0, padd), (0, 0)]) ########### the padding is made for the coevolution features
        Y=np.pad(ldic[i]['label'], [(0, padd), (0, padd), (0, 0)]) ########### the padding is made for the contact map
        Y[:,l:,0]=1  # padding is established as no contact due to 2D contact map is in one hot encoding format
        Y[l:,:l,0]=1 # padding is established as no contact due to 2D contact map is one hot encoding
        D = {'name': ldic[i]['name'] ,'sequence': ldic[i]['sequence']
            ,'fseq': Xs,'fcoev':Xc,'label':np.int8(Y)}  # the data with the padding is saved
        ldicp.append(D)
    return ldicp # this function return the data

def SelecbySize(Ldic,Lprot,liminf=26,limsup=400): # Select a group of dataset by protein size
    indx=np.where(np.logical_and(Lprot >= liminf, Lprot <= limsup))[0]
    Rdic=[]
    for i in indx:
        Rdic.append(Ldic[i])
    return Rdic,indx

def dataGen(DR,data_file,Lmin=26,Lmax=430,padd=10,istrain=False,norm=False):
    print(f"{DR},{data_file}")
    Data_Dic,L_dat=Extract_features(file=f"{data_file}.pickle",DR_method=DR,istrain=istrain,norm=norm)
    if(padd is not None):
       Data_Dic=Prot_padding(Data_Dic,padd)
    Data_Dic_R,ind_dat=SelecbySize(Data_Dic,L_dat,Lmin,Lmax)
    print(len(Data_Dic_R))
    # with open(f"{folder}/{data_file}_L{Lmax}.pickle", 'wb') as data_file_1:
    #      pickle.dump(Data_Dic_R,data_file_1)
    return Data_Dic_R

