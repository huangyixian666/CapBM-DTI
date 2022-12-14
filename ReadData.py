from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
import pandas as pd
import numpy as np
# from tensorflow.keras.datasets import mnist
# from keras.utils import np_utils
# from keras.utils.vis_utils import plot_model
# from tensorflow.keras.models import Sequential,Model
# from tensorflow.keras.layers import Dense,Input,Conv1D,LSTM,MaxPooling1D,Flatten,Dropout,concatenate,Reshape,GlobalAveragePooling1D,BatchNormalization
# from keras.layers.recurrent import SimpleRNN
# from keras import layers
# from keras.optimizers import Adam
# from keras.optimizers import adam_v2
# from CapsuleLayer import Capsule
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from bert import bert
import matplotlib
# import tensorflow as tf
# from sklearn.model_selection import train_test_split
from MPNN import *
import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def onehot(seq_list,seq_len):
    bases = ['K','L','G','A','R','S','V','I','E','P','F','T','D','N','Q','H','C','W','Y','M']
    X = np.zeros((len(seq_list),seq_len,len(bases)+1))
    # print(X.shape)
    for i,seq in enumerate(seq_list):
        for l,aa in enumerate(str(seq)):
            if l<int(seq_len):
                if aa in bases:
                    X[i,l,bases.index(aa)+1] = 1
                else:
                    X[i,l,0] = 1
    return X


def newdata(dti_fname,protein_encoder,seq_len,drug_encoder,name):
    dti_df=pd.read_csv(dti_fname,sep=" ",header=None)
    # dti_df
    protein_list=dti_df[1].tolist()
    length=dti_df[1].map(lambda x:len(str(x)))
    print ("Sequence_max_length:"+str(length.max()))
    # protein_list
    if protein_encoder=="onehot":
        protein_df=onehot(protein_list,seq_len)

    if protein_encoder=="bert":
        protein_df=bert(protein_list)

    drug_df_list=[]

    if "fingerprint" in drug_encoder:
        drug_list=dti_df[0].tolist()
        j=0
        column_list=[]
        while j<1024:
            k=j+1
            column_list.append("fingerprints"+str(k))
            j=j+1

        drug_fingerprint_list=[]
        for x,drug in enumerate(drug_list):
            drug_fingerprint=[]
            if Chem.MolFromSmiles(drug):
                mol=Chem.MolFromSmiles(drug)
                fps=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
                fingerprints=fps.ToBitString()
                i=0
                while i<1024:
                    drug_fingerprint.append(float(fingerprints[i]))
                    i=i+1
            else:
                print(str(drug))
                print("Above smile transforms to fingerprint error!!!")
                print("Please remove "+str(x+1)+" line")
                i=0
                while i<1024:
                    drug_fingerprint.append(0)
                    i=i+1
            drug_fingerprint_list.append(drug_fingerprint)
        drug_df=pd.DataFrame(data=drug_fingerprint_list,columns=column_list)
        print("Drug data:"+str(drug_df.shape))
        drug_df_list.append(drug_df)

    if "MPNN" in drug_encoder:
        drug_df=graphs_from_smiles(dti_df[0].tolist())
        print("drug[0]:"+str(len(drug_df[0])))
        print("drug[1]:"+str(len(drug_df[1])))
        print("drug[2]:"+str(len(drug_df[2])))
        drug_df_list.append(drug_df)
    
    y=dti_df[2].tolist()
    
    print("protein data:"+str(protein_df.shape))
    print("label:"+str(len(y)))

    return np.array(protein_df),drug_df_list,y
 