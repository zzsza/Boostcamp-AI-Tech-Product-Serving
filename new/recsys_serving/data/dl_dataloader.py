import os
import numpy as np
import pandas as pd

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl


class Dataloader(pl.LightningDataModule):
    def __init__(self, data_shuffle, batch_size, seed, test_size):
        super().__init__()
        self.data_shuffle = data_shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.test_size = test_size

    def load_data(self, data_path):
        users = pd.read_csv(os.path.join(data_path, "users.csv"))
        books = pd.read_csv(os.path.join(data_path, "books.csv"))
        train = pd.read_csv(os.path.join(data_path, "train_ratings.csv"))
        test = pd.read_csv(os.path.join(data_path, "test_ratings.csv"))
        sub = pd.read_csv(os.path.join(data_path, "submit", "sample_submission.csv"))

        ids = pd.concat([train['user_id'], sub['user_id']]).unique()
        isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

        idx2user = {idx: id for idx, id in enumerate(ids)}
        idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}

        user2idx = {id: idx for idx, id in idx2user.items()}
        isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}

        train['user_id'] = train['user_id'].map(user2idx)
        sub['user_id'] = sub['user_id'].map(user2idx)
        test['user_id'] = test['user_id'].map(user2idx)

        train['isbn'] = train['isbn'].map(isbn2idx)
        sub['isbn'] = sub['isbn'].map(isbn2idx)
        test['isbn'] = test['isbn'].map(isbn2idx)

        field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)
        data = {
            'train': train,
            'test': test.drop(['rating'], axis=1),
            'field_dims': field_dims,
            'users': users,
            'books': books,
            'sub': sub,
            'idx2user': idx2user,
            'idx2isbn': idx2isbn,
            'user2idx': user2idx,
            'isbn2idx': isbn2idx,
        }
        return data

    def dl_data_loader(self, data):
        train_dataset = TensorDataset(
            torch.LongTensor(data['X_train'].values),
            torch.LongTensor(data['y_train'].values)
        )
        valid_dataset = TensorDataset(
            torch.LongTensor(data['X_valid'].values),
            torch.LongTensor(data['y_valid'].values)
        )
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.data_shuffle)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.data_shuffle)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        data['train_dataloader'] = train_dataloader
        data['valid_dataloader'] = valid_dataloader
        data['test_dataloader'] = test_dataloader
        return data

    def split_dataset(self, data):
        x_train, x_valid, y_train, y_valid = train_test_split(
            data['train'].drop(['rating'], axis=1),
            data['train']['rating'],
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=self.data_shuffle
        )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = x_train, x_valid, y_train, y_valid
        return data
