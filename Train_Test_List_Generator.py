import os
from scipy.io import loadmat
import pandas as pd


DATAPATH = '.\preliminary_cl'
LABLE_FILE_NAME = 'reference.txt'
FOLDER_NAME = 'TRAIN'
new_labels = {'path': [], 'if_normal': [], 'if_abnormal': []}
valset_ratio = 0.25

if __name__ == '__main__':
    dataset_size = len(open(os.path.join(DATAPATH, LABLE_FILE_NAME)).readlines())
    print('Dataset size is:', dataset_size)
    labels = open(os.path.join(DATAPATH, LABLE_FILE_NAME))
    for line in labels:
        file_name, if_abnormal = line.split()
        file_path = os.path.join(DATAPATH, FOLDER_NAME, file_name+'.mat')
        try:
            mat = loadmat(file_path)['data']
        except Exception:
            print('Not a valid mat.')
        else:
            new_labels['path'].append(file_path)
            if if_abnormal == '1':
                new_labels['if_normal'].append(0)
                new_labels['if_abnormal'].append(1)
            else:
                new_labels['if_normal'].append(1)
                new_labels['if_abnormal'].append(0)
    ANNOTATION = pd.DataFrame(new_labels).sample(frac=1)
    ANNOTATION.index = range(len(ANNOTATION))
    ANNOTATION[0: int(dataset_size * valset_ratio)].to_csv('valset_annotation.csv')
    print('Valset_annotation file is saved.')
    ANNOTATION[int(dataset_size * valset_ratio):].to_csv('trainset_annotation.csv')
    print('Trainset_annotation file is saved.')



