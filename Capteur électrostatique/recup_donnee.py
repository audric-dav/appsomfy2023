# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 13:27:27 2022

@author: esteb
"""
import csv
import serial

header=["Temps", "Amplitude"]
recup_donnee = "C:\\Users\\esteb\\OneDrive\\Bureau\\Polytech\\FI4\\Projet\\Pas_chute\\60.csv"
tab = []
portserie= serial.Serial('COM4',baudrate=9600,timeout=None)

for i in range(1,100):
    donnees_arduino=portserie.readline().decode('ascii')
    tab.append(int (donnees_arduino))
    #print(donnees_arduino)
    
with open(recup_donnee,"w",newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(header)
    for i in range (len(tab)):
        writer.writerow([i,tab[i]])
        
print(tab)

portserie.close()