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
        users['user_id'] = users['user_id'].map(user2idx)

        train['isbn'] = train['isbn'].map(isbn2idx)
        sub['isbn'] = sub['isbn'].map(isbn2idx)
        test['isbn'] = test['isbn'].map(isbn2idx)
        books['isbn'] = books['isbn'].map(isbn2idx)

        idx, context_train, context_test = self.prepare_context_data(users, books, train, test)
        field_dims = np.array(
            [
                len(user2idx),
                len(isbn2idx),
                6,
                len(idx['loc_city2idx']),
                len(idx['loc_state2idx']),
                len(idx['loc_country2idx']),
                len(idx['category2idx']),
                len(idx['publisher2idx']),
                len(idx['language2idx']),
                len(idx['author2idx'])
            ],
            dtype=np.uint32
        )

        data = {
            'train': context_train,
            'test': context_test.drop(['rating'], axis=1),
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

    def prepare_context_data(self, users, books, ratings1, ratings2):
        """
        :param users: <pd.DataFrame> users.csv
        :param books: <pd.DataFrame> books.csv
        :param ratings1: <pd.DataFrame> rating of train dataset
        :param ratings2: <pd.DataFrame> rating of test dataset
        """

        users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
        users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
        users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
        users = users.drop(['location'], axis=1)

        ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

        columns = ['isbn', 'category', 'publisher', 'language', 'book_author']
        context_df = ratings.merge(users, on='user_id', how='left').merge(books[columns], on='isbn', how='left')
        train_df = ratings1.merge(users, on='user_id', how='left').merge(books[columns], on='isbn', how='left')
        test_df = ratings2.merge(users, on='user_id', how='left').merge(books[columns], on='isbn', how='left')

        # Indexing
        loc_city2idx = {v: k for k, v in enumerate(context_df['location_city'].unique())}
        loc_state2idx = {v: k for k, v in enumerate(context_df['location_state'].unique())}
        loc_country2idx = {v: k for k, v in enumerate(context_df['location_country'].unique())}

        train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
        train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
        train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
        test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
        test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
        test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

        train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
        train_df['age'] = train_df['age'].apply(self.age_map)
        test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
        test_df['age'] = test_df['age'].apply(self.age_map)

        category2idx = {v: k for k, v in enumerate(context_df['category'].unique())}
        publisher2idx = {v: k for k, v in enumerate(context_df['publisher'].unique())}
        language2idx = {v: k for k, v in enumerate(context_df['language'].unique())}
        author2idx = {v: k for k, v in enumerate(context_df['book_author'].unique())}

        train_df['category'] = train_df['category'].map(category2idx)
        train_df['publisher'] = train_df['publisher'].map(publisher2idx)
        train_df['language'] = train_df['language'].map(language2idx)
        train_df['book_author'] = train_df['book_author'].map(author2idx)
        test_df['category'] = test_df['category'].map(category2idx)
        test_df['publisher'] = test_df['publisher'].map(publisher2idx)
        test_df['language'] = test_df['language'].map(language2idx)
        test_df['book_author'] = test_df['book_author'].map(author2idx)

        idx = {
            "loc_city2idx": loc_city2idx,
            "loc_state2idx": loc_state2idx,
            "loc_country2idx": loc_country2idx,
            "category2idx": category2idx,
            "publisher2idx": publisher2idx,
            "language2idx": language2idx,
            "author2idx": author2idx,
        }
        return idx, train_df, test_df

    def context_data_loader(self, data):
        train_dataset = TensorDataset(
            torch.LongTensor(data['X_train'].values),
            torch.LongTensor(data['y_train'].values)
        )
        valid_dataset = TensorDataset(
            torch.LongTensor(data['X_valid'].values),
            torch.LongTensor(data['y_valid'].values)
        )
        test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=self.data_shuffle)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.data_shuffle)
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

    @classmethod
    def age_map(cls, x: int) -> int:
        x = int(x)
        if x < 20:
            return 1
        elif 20 <= x < 30:
            return 2
        elif 30 <= x < 40:
            return 3
        elif 40 <= x < 50:
            return 4
        elif 50 <= x < 60:
            return 5
        else:
            return 6
