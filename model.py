from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis as PA
import pandas as pd
import numpy as np
# from tensorflow.keras.datasets import mnist
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Input,Conv1D,LSTM,MaxPooling1D,Flatten,Dropout,concatenate,Reshape,GlobalAveragePooling1D,BatchNormalization
# from keras.layers.recurrent import SimpleRNN
# from keras import layers
# from keras.optimizers import Adam
from keras.optimizers import adam_v2
# from CapsuleLayer import Capsule
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt
from bert import bert
import matplotlib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from MPNN import *
from ReadData import *



import tensorflow as tf
from sklearn.model_selection import train_test_split
# import pickle
def split(protein,drug,y,drug_encoder,name):
    if not os.path.exists(name):
        os.makedirs(name)
    if drug_encoder == "fingerprint":
        train_p,test_p,train_d,test_d,train_y,test_y=train_test_split(protein,drug[0],y,test_size = 0.2, random_state = 1,stratify=y)#1
        with open(name+"/sample_summary.txt","w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug\ttest_drug\ttrain_y\ttest_y\n")
                fw.write(str(train_p.shape)+"\t"+str(test_p.shape)+"\t"+str(train_d.shape)+"\t"+str(test_d.shape)+"\t"+str(len(train_y))+"\t"+str(len(test_y)))
        # print(train_p.shape,test_p.shape,train_d.shape,test_d.shape,len(train_y),len(test_y))
    if drug_encoder == "MPNN":
        train_p,test_p,train_d0,test_d0,train_d1,test_d1,train_d2,test_d2,train_y,test_y=train_test_split(protein,drug[0][0],drug[0][1],drug[0][2],y,test_size = 0.2, random_state = 1,stratify=y)#
        with open(name+"/sample_summary.txt","w") as fw:
                fw.write("train_protein\ttest_protein\ttrain_drug_d0\ttest_drug_d0\ttrain_drug_d1\ttest_drug_d1\ttrain_drug_d2\ttest_drug_d2\ttrain_y\ttest_y\n")
                fw.write(str(train_p.shape)+"\t"+str(test_p.shape)+"\t"+str(len(train_d0))+"\t"+str(len(test_d0))+"\t"+str(len(train_d1))+"\t"+str(len(test_d1))+"\t"+str(len(train_d2))+"\t"+str(len(test_d2))+"\t"+str(len(train_y))+"\t"+str(len(test_y)))
        # print(train_p.shape,test_p.shape,len(train_d0),len(test_d0),len(train_d1),len(test_d1),len(train_d2),len(test_d2),len(train_y),len(test_y))
        # train_d=(tf.ragged.constant(train_d0, dtype=tf.float32),tf.ragged.constant(train_d1, dtype=tf.float32),tf.ragged.constant(train_d2, dtype=tf.int64))
        test_d=(tf.ragged.constant(test_d0, dtype=tf.float32),tf.ragged.constant(test_d1, dtype=tf.float32),tf.ragged.constant(test_d2, dtype=tf.int64))
        train_d=(train_d0,train_d1,train_d2)
    train_y=np_utils.to_categorical(train_y,num_classes=2)
    test_y=np_utils.to_categorical(test_y,num_classes=2)
    
    return train_p,test_p,train_d,test_d,train_y,test_y

def model_onehot_MPNN_capsule(atom_dim=29,bond_dim=7,seq_len=1000,target_dense=200,kernel_size=5,num_capsule=2,routings=3,
   batch_size=64,
   message_units=64,
   message_steps=4,
   num_attention_heads=8,
   dense_units=512
   ):
    sequence_input_1 = Input(shape=(seq_len,21))
    model_p=Flatten()(sequence_input_1)
    model_p=Dense(target_dense,activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model=concatenate([model_p,model_d])
    model=Reshape((-1,8))(model)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    length = Length()(capsule)
    model=keras.Model(inputs=[sequence_input_1,[atom_features, bond_features, pair_indices, molecule_indicator]],outputs=length)
    # plot_model(model,to_file='onehot_MPNN_capsule/onehot_MPNN_capsule.png',show_shapes=True)
    model.summary()

    return model


def model_bert_MPNN_capsule(atom_dim=29,bond_dim=7,seq_len=1000,target_dense=200,kernel_size=5,num_capsule=2,routings=3,
   batch_size=64,
   message_units=64,
   message_steps=4,
   num_attention_heads=8,
   dense_units=512
   ):

    sequence_input_1 = Input(shape=(1024))
    # model_p=Flatten()(sequence_input_1)
    model_p=Dense(target_dense,activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model=concatenate([model_p,model_d])
    model=Reshape((-1,8))(model)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    length = Length()(capsule)
    model=keras.Model(inputs=[sequence_input_1,[atom_features, bond_features, pair_indices, molecule_indicator]],outputs=length)
    # plot_model(model,to_file='onehot_MPNN_capsule/onehot_MPNN_capsule.png',show_shapes=True)
    model.summary()

    return model


def model_bert_MPNN_dense(atom_dim=29,bond_dim=7,seq_len=1000,target_dense=200,
   batch_size=64,
   message_units=64,
   message_steps=4,
   num_attention_heads=8,dense_units_2=512,
    dense_units=512):
    sequence_input_1 = Input(shape=(1024))
    model_p=Dense(200,activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )
    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    model=concatenate([model_p,model_d])
    x=Dense(dense_units_2,activation='relu')(model)
    output=Dense(2,activation='sigmoid')(x)

    model=keras.Model(inputs=[sequence_input_1,[atom_features, bond_features, pair_indices, molecule_indicator]],outputs=output)
    # plot_model(model,to_file='bert_MPNN_dense/bert_MPNN_dense.png',show_shapes=True)
    model.summary()

    return model

from tensorflow import keras
def model_onehot_MPNN_dense(atom_dim=29,bond_dim=7,seq_len=1000,target_dense=200,
   batch_size=64,
   message_units=64,
   message_steps=4,
   num_attention_heads=8,dense_units_2=512,
   dense_units=512):
    sequence_input_1 = Input(shape=(seq_len,21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p=Flatten()(sequence_input_1)
    model_p=Dense(target_dense,activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")
    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    model_d = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])


    model=concatenate([model_p,model_d])

    x=Dense(dense_units_2,activation='relu')(model)
    output=Dense(2,activation='sigmoid')(x)

    model=keras.Model(inputs=[sequence_input_1,[atom_features, bond_features, pair_indices, molecule_indicator]],outputs=output)
    #plot_model(model,to_file='onehot_MPNN_dense/onehot_MPNN_dense.png',show_shapes=True)
    model.summary()

    return model

import tensorflow as tf


def model_onehot_fingerprint_dense(param):
    sequence_input_1 = Input(shape=(param['seq_len'],21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p=Flatten()(sequence_input_1)
    model_p=Dense(param['target_dense'],activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)


    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d=Dense(param['drug_dense'],activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model=concatenate([model_p,model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    x=Dense(20,activation='relu')(model)
    output=Dense(2,activation='sigmoid')(x)

    model=Model(inputs=[sequence_input_1,sequence_input_2],outputs=output)
    # plot_model(model,to_file='onehot_fingerprint_dense/onehot_fingerprint_dense.png',show_shapes=True)
    model.summary()
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])
    

    return model



def model_bert_fingerprint_dense(param):
    sequence_input_1 = Input(shape=(1024))
    model_p=Dense(param['target_dense'],activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d=Dense(param['drug_dense'],activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model=concatenate([model_p,model_d])

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    model=Dense(20,activation='relu')(model)
    # x=Dense(50,activation='relu')(model)
    output=Dense(2,activation='sigmoid')(model)

    model=Model(inputs=[sequence_input_1,sequence_input_2],outputs=output)
    # plot_model(model,to_file='bert_fingerprint_dense/bert_fingerprint_dense.png',show_shapes=True)
    model.summary()
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model

def model_onehot_fingerprint_capsule(param): #
    # sequence_input_1 = Input(shape=(seq_len,21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    sequence_input_1 = Input(shape=(param['seq_len'],21))
    # model_p=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_1)
    # model_p=BatchNormalization()(model_p)
    # model_p=GlobalAveragePooling1D()(model_p)
    model_p=Flatten()(sequence_input_1)
    model_p=Dense(param['target_dense'],activation='relu')(model_p)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d=Dense(param['drug_dense'],activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model=concatenate([model_p,model_d])
    model=Reshape((-1,8))(model)

    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1, padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule = 16, routings = param['routings'], share_weights=True)(primarycaps)

    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)

    model=Model(inputs=[sequence_input_1,sequence_input_2],outputs=length)
    # plot_model(model,to_file='onehot_fingerprint_capsule/onehot_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model


def model_bert_fingerprint_capsule(param):
    # sequence_input_1 = Input(shape=(1024))
    # model_p=Dense(200,activation='relu')(sequence_input_1)
    # model_p = BatchNormalization()(model_p)


    sequence_input_1 = Input(shape=(1024))
    model_p=Dense(param['target_dense'],activation='relu')(sequence_input_1)
    model_p = BatchNormalization()(model_p)

    sequence_input_2 = Input(shape=(1024))
    # cnn2=Conv1D(filters=filter_num,kernel_size=kernel_size,padding="same",activation="relu")(sequence_input_2)
    # cnn2=MaxPooling1D(pool_size=2,strides=2)(cnn2)
    model_d=Dense(param['drug_dense'],activation='relu')(sequence_input_2)
    model_d = BatchNormalization()(model_d)

    model=concatenate([model_p,model_d])
    model=Reshape((-1,8))(model)

    # primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=kernel_size, strides=1, padding='valid')
    # capsule = Capsule(num_capsule=num_capsule, dim_capsule = 16, routings = routings, share_weights=True)(primarycaps)
    primarycaps = PrimaryCap(model, dim_vector=8, n_channels=8, kernel_size=param['kernel_size'], strides=1, padding='valid')
    capsule = Capsule(num_capsule=param['num_capsule'], dim_capsule = 16, routings = param['routings'], share_weights=True)(primarycaps)
    length = Length()(capsule)

    # cnn1=Dropout(0.25)(cnn1)
    # x=LSTM(num_lstm,return_sequences=True)(model)
    # x=Dense(20,activation='relu')(model)
    # output=Dense(2,activation='sigmoid')(x)
    model=Model(inputs=[sequence_input_1,sequence_input_2],outputs=length)
    # plot_model(model,to_file='bert_fingerprint_capsule/bert_fingerprint_capsule.png',show_shapes=True)
    model.summary()
    model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

    return model
