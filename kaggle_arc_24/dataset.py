import json

import pandas as pd
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import numpy as np
import albumentations as A
import cv2


class Arc24DatasetTransformations(Dataset):

    def __init__(
            self, 
            data_path:str, 
            # solution_path:str, 
            max_dim_task:int=30, 
            pad_val:int=0,
            max_n_tasks:int=10
            ) -> None:
        super(Arc24DatasetTransformations, self).__init__()

        with open(data_path, 'r') as data_f:
            data = json.load(data_f)

        # with open(solution_path, 'r') as solutions_f:
        #     solutions = json.load(solutions_f)

        self.df = self.init_data(data)
        self.max_dim_task = max_dim_task
        self.pad_val = pad_val
        self.max_n_tasks = max_n_tasks


    def init_data(self, data):
        df = pd.DataFrame()
        keys = sorted(data.keys())
        data_processor = tqdm(keys, total=len(keys))
        # process train data
        for k in data_processor:  
            train_input = [d['input'] for d in data[k]['train']]
            train_output = [d['output'] for d in data[k]['train']]
            test_input = [d['input'] for d in data[k]['test']]
            new_row = pd.DataFrame({'key':[k], 'train_input':[train_input], 'train_output':[train_output], 'test_input':[test_input]})
            df = pd.concat([df, new_row], ignore_index=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):

        row = self.df.loc[idx]

        train_input = row['train_input']
        train_output = row['train_output']

        pad_if_needed = A.PadIfNeeded(self.max_dim_task, self.max_dim_task, border_mode=cv2.BORDER_CONSTANT, value=self.pad_val)

        n_samples = len(train_input)
        tasks_input = np.zeros((n_samples, self.max_dim_task, self.max_dim_task))
        tasks_output = np.zeros((n_samples, self.max_dim_task, self.max_dim_task))
        for n in range(n_samples):
            sample_input = train_input[n]
            sample_output = train_output[n]
            # prepare input data
            task_input = np.array(sample_input)
            task_input = pad_if_needed(image=task_input)['image']
            tasks_input[n] = task_input

            # prepare output data
            task_output = np.array(sample_output)
            task_output = pad_if_needed(image=task_output)['image']
            tasks_output[n] = task_output

        n_empty_tasks_to_add = self.max_n_tasks - n_samples

        if n_empty_tasks_to_add > 0:
            pad_tasks = np.zeros((n_empty_tasks_to_add, self.max_dim_task, self.max_dim_task))
            tasks_input = np.concatenate((tasks_input, pad_tasks))
            tasks_output = np.concatenate((tasks_output, pad_tasks))

        assert tasks_input.shape == tasks_output.shape, "Shapes between input/output don't match"
        return tasks_input, tasks_output



# train_dataset = Arc24DatasetTransformations(data_path=r'C:\Users\tommy\Developer\Kaggle-ARC-24\arc-prize-2024\arc-agi_training_challenges.json')
# train_input, train_output = train_dataset.__getitem__(23)
# print(train_input.shape, train_output.shape)