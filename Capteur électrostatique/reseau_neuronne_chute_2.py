# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 09:17:06 2023

@author: esteb
"""

import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
#import tensorflow_addons as tfa
import csv
from statistics import mean

#////mise en forme des données ////

chemin = "C:\\Users\\esteb\\OneDrive\\Bureau\\Polytech\\FI4\\Projet\\BDD_chute"

liste_excel = [f for f in os.listdir(chemin) if f.endswith('.csv')]

for file in liste_excel: 

    if file.startswith('chute'): 
        file_path = os.path.join(chemin, file) 
        df = pd.read_csv(file_path, delimiter=";") 
        last_row_index = len(df) - 1 
        df.iloc[last_row_index,1] = 1 
        df.to_csv(file_path, sep=";", index=False) 

    else: 
        file_path = os.path.join(chemin, file) 
        df = pd.read_csv(file_path, delimiter=";") 
        last_row_index = len(df) - 1 
        df.iloc[last_row_index,1] = 0 
        df.to_csv(file_path, sep=";", index=False) 

bibli_excel = {}
for file in liste_excel:
        file_path = os.path.join(chemin, file)
        df = pd.read_csv(file_path, delimiter=";")
        bibli_excel[file] = df
        
bibli_excel_2 = {}

for file,df in bibli_excel.items():
    deuxieme_colonne = df.iloc[:,1]
    bibli_excel_2[file] = deuxieme_colonne.values
    
#//// Diviser les données ////

n = len(liste_excel)
m = int(n/2)
train_excel = liste_excel[0:int(n*0.35)]+liste_excel[m:m+int(n*0.35)]
val_excel = liste_excel[int(n*0.35):int(n*0.5)]+liste_excel[m+int(n*0.35):m+int(n*0.5)]
# val_excel = liste_excel[int(n*0.35):int(n*0.45)]+liste_excel[m+int(n*0.35):m+int(n*0.45)]
# test_excel = liste_excel[int(n*0.45):int(n*0.5)]+liste_excel[m+int(n*0.45):n]

num_features = df.shape[1]

#//// Normaliser ////

for cle, tableau in bibli_excel_2.items():
    derniere_ligne = tableau[-1:]
    tableau = tableau[:-1]
    moyenne = (tableau.mean()) 
    ecart_type = (tableau.std())
    tableau = (tableau-moyenne)/ecart_type
    tableau = np.concatenate((tableau, derniere_ligne))
    bibli_excel_2[cle] = tableau

#//// fenetrage des données ////

train_rawdata=np.empty((0,99))
for cle in train_excel:
    tableau = bibli_excel_2[cle]
    train_rawdata = np.vstack((train_rawdata,tableau))

train_dataset=tf.data.Dataset.from_tensor_slices(train_rawdata).shuffle(train_rawdata.shape[0])

val_rawdata=np.empty((0,99))
for cle in val_excel:
    tableau = bibli_excel_2[cle]
    val_rawdata = np.vstack((val_rawdata,tableau))

val_dataset=tf.data.Dataset.from_tensor_slices(val_rawdata)

def rawtoxy(sample):
    x = sample[:-1]
    y = sample[-1]
    '''tf.print("x=", x)
    tf.print("gt=", y)
    '''
    return x, tf.cast(y,tf.int32)

train_datasetwithlabels = train_dataset.map(rawtoxy).batch(12).prefetch(1)
val_datasetwithlabels = val_dataset.map(rawtoxy).batch(12).prefetch(1)
#test_datasetwithlabels = test_dataset.map(rawtoxy).batch(12).prefetch(1)

#/// Création du modèle ///

def model():
#/// neuronne de convolution ///
    x = tf.keras.layers.Input(shape=(98, 1),name="input")
#/// 1ère convolution ///
    c1 = tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(x)
    c2 = tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=x+c2
    c1 = tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(out)
    c2 = tf.keras.layers.Conv1D(filters=16, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=out+c2
    p1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(out)
    pc1 = tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(p1)
#/// 2ème convolution ///
    c1 = tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(pc1)
    c2 = tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=c2+pc1
    c1 = tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(out)
    c2 = tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=c2+pc1
    p1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(out)
    pc1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(p1)
#/// 3ème convolution ///
    c1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(pc1)
    c2 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=pc1+c2
    c1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(out)
    c2 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=c2+out
    p1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(out)
    pc1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(p1)
#/// 4ème convolution ///
    c1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(pc1)
    c2 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=pc1+c2
    c1 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(out)
    c2 = tf.keras.layers.Conv1D(filters=64, kernel_size=(3,), padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='relu')(c1)
    out=out+c2  
    flat = tf.keras.layers.Flatten()(out)
#/// neuronne dense de prise de décision /!\ ///
    decision = tf.keras.layers.Dense(units=1, kernel_regularizer=tf.keras.regularizers.l2(0.0001), activation='sigmoid')(flat)
    #prob = tf.keras.layers.Softmax()(decision)
    model = tf.keras.Model(inputs=x, outputs=[decision])
    return model

model=model()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),loss = 'binary_crossentropy', metrics = 'accuracy')
#model.compile(loss = tfa.losses.SigmoidFocalCrossEntropy('binary_crossentropy'), metrics = 'accuracy')

print(model.summary())
NB_EPOCH = 400

all_callbacks={}
#return all_callbacks
# -> terminate on NaN loss values
all_callbacks['TerminateOnNaN_callback']=tf.keras.callbacks.TerminateOnNaN()
# -> apply early stopping
early_stopping_patience= 10
all_callbacks['earlystopping_callback']=tf.keras.callbacks.EarlyStopping(
                          monitor='val_loss',
                          patience=early_stopping_patience,
                          restore_best_weights=True
                        )
all_callbacks['reduceLROnPlateau_callback']=tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                            patience=(early_stopping_patience*2)//3, min_lr=0.000001, verbose=True)

#-> checkpoint each epoch

all_callbacks['tensorboard_callback']=tf.keras.callbacks.TensorBoard(log_dir='log_dir',
                                                    histogram_freq=1,
                                                    write_graph=True,
                                                    update_freq='epoch',
                                                    )


history = model.fit(x=train_datasetwithlabels, batch_size = None, epochs = NB_EPOCH, validation_data = val_datasetwithlabels, callbacks=list(all_callbacks.values()))

modele_dict = history.history
val_loss = modele_dict['val_loss']
epochs = range(1, len(val_loss)+1)
plt.plot(epochs, val_loss, 'bo', label='loss')
plt.show()

y_predict = model.predict(val_datasetwithlabels)

# Affichage des prédictions pour les 10 premiers échantillons 
chute=0
pas_chute=0
for i in range(len(val_excel)): 
    y_pred_i = y_predict[i] 
 #   y_pred_i=np.round(y_predict+threshold - 0.5).astype(int)
    y_true_i = val_excel[i] 
    if y_pred_i > 0.5:
        y_pred_i = 1
        chute+=1
    else :
        y_pred_i=0
        pas_chute+=1
    print(f"Échantillon {i}: vraie étiquette = {y_true_i}, prédiction = {y_pred_i}")


model.save("model.keras")