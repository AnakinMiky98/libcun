import ast
import csv
import os
import sys
import numpy
import pickle
from pickle import dump

import numpy as np
import glob
import shutil

from os import listdir
from os.path import isfile, join
#from tfsnippet.utils import makedirs

output_folder = 'processed'
os.makedirs(output_folder, exist_ok=True)


def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                         dtype=np.float32,
                         delimiter=',')
    print(temp)
    print(dataset, category, filename, temp.shape)
    with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
        dump(temp, file)


def load_data(dataset):
    if dataset == 'SMD':
        dataset_folder = 'ServerMachineDataset'
        file_list = os.listdir(os.path.join(dataset_folder, "train"))
        for filename in file_list:
            if filename.endswith('.txt'):
                load_and_save('train', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test', filename, filename.strip('.txt'), dataset_folder)
                load_and_save('test_label', filename, filename.strip('.txt'), dataset_folder)



    elif dataset == 'Exa':
        dataset_folder = 'Exathlon'
        print("<<INFO>>: inizio elaborazione EXA")

        dir_out = os.path.join("processed")
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

        #TRAIN
        nome_files = [j for j in listdir(dataset_folder) if isfile(join(dataset_folder, j))]
        for f in nome_files:
            if 'train' in f:
                with open(os.path.join(dataset_folder, f)) as train_file:
                    csv_reader = csv.reader(train_file, delimiter=',')
                    next(csv_reader)
                    # Converto il file di train in pickel
                    temp_train = []
                    for row in csv_reader:
                        floats = [float(val) for val in row[1:]]
                        temp_train.append(floats)
                    print("<<INFO>>: Numero di colonne Train: ", len(floats))

                print("<<INFO>>: dimensione train ", len(temp_train))
                temp1 = np.array(temp_train)
                with open(os.path.join(dir_out, "Exa_train.pkl"), 'wb') as f:
                    pickle.dump(temp1, f)
                print('<<INFO>>: terminata elaborazione train\n')

            #TEST
            elif 'test' in f:
                #Mi leggo il file di test, esrapolo il file di label e converto tutto il pickel
                with open(os.path.join(dataset_folder, f)) as test_file:
                    csv_reader = csv.reader(test_file, delimiter=',')
                    #salto la riga di intestazione
                    next(csv_reader)
                    #devo creare il file csv di lable leggendo solo l'ultimo valore dell'ultima colonna del test
                    temp_label = []
                    temp_test = []
                    for row in csv_reader:
                        floats = [float(val) for val in row[1:]]
                        temp_test.append(floats)
                        temp_label.append(float(row[len(row)-1]))
                print("<<INFO>>: Numero di colonne Test: ", len(floats))
                print("<<INFO>>: dimensione test ", len(temp_test), " label ", len(temp_label))

                #creo il pkl di label e test
                temp1 = np.array(temp_test)
                with open(os.path.join(dir_out, "Exa_test.pkl"), 'wb') as f:
                    pickle.dump(temp1, f)
                print('<<INFO>>: terminata elaborazione test')

                temp1 = np.array(temp_label)
                with open(os.path.join(dir_out, "Exa_test_label.pkl"), 'wb') as f:
                     pickle.dump(temp1, f)
                print('<<INFO>>: terminata elaborazione label \n')









    elif dataset == 'SMAP' or dataset == 'MSL':
        dataset_folder = 'data'
        with open(os.path.join(dataset_folder, 'labeled_anomalies.csv'), 'r') as file:
            csv_reader = csv.reader(file, delimiter=',')
            res = [row for row in csv_reader][1:]
        res = sorted(res, key=lambda k: k[0])
        label_folder = os.path.join(dataset_folder, 'test_label')
        makedirs(label_folder, exist_ok=True)
        data_info = [row for row in res if row[1] == dataset and row[0] != 'P-2']
        labels = []
        for row in data_info:
            anomalies = ast.literal_eval(row[2])
            length = int(row[-1])
            label = np.zeros([length], dtype=np.bool)
            for anomaly in anomalies:
                label[anomaly[0]:anomaly[1] + 1] = True
            labels.extend(label)
        labels = np.asarray(labels)
        print(dataset, 'test_label', labels.shape)
        with open(os.path.join(output_folder, dataset + "_" + 'test_label' + ".pkl"), "wb") as file:
            dump(labels, file)

        def concatenate_and_save(category):
            data = []
            for row in data_info:
                filename = row[0]
                temp = np.load(os.path.join(dataset_folder, category, filename + '.npy'))
                data.extend(temp)
            data = np.asarray(data)
            print(dataset, category, data.shape)
            with open(os.path.join(output_folder, dataset + "_" + category + ".pkl"), "wb") as file:
                dump(data, file)

        for c in ['train', 'test']:
            concatenate_and_save(c)


if __name__ == '__main__':
    datasets = ['SMD', 'SMAP', 'MSL', 'Exa']
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            if d in datasets:
                load_data(d)
    else:
        print("""
        Usage: python data_preprocess.py <datasets>
        where <datasets> should be one of ['SMD', 'SMAP', 'MSL']
        """)
