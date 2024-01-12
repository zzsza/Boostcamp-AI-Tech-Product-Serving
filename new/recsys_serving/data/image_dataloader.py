import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl


class Dataloader(pl.LightningDataModule):
    def __init__(self, data_path, device, vector_create, data_shuffle, batch_size, seed, test_size):
        super().__init__()
        self.data_path = data_path
        self.device = device
        self.vector_create = vector_create
        self.data_shuffle = data_shuffle
        self.batch_size = batch_size
        self.seed = seed
        self.test_size = test_size

    def load_data(self):
        users = pd.read_csv(self.data_path + 'users.csv')
        books = pd.read_csv(self.data_path + 'books.csv')
        train = pd.read_csv(self.data_path + 'train_ratings.csv')
        test = pd.read_csv(self.data_path + 'test_ratings.csv')
        sub = pd.read_csv(self.data_path + 'sample_submission.csv')

        ids = pd.concat([train['user_id'], sub['user_id']]).unique()
        isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

        idx2user = {idx: id for idx, id in enumerate(ids)}
        idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}

        user2idx = {id: idx for idx, id in idx2user.items()}
        isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}

        train['user_id'] = train['user_id'].map(user2idx)
        sub['user_id'] = sub['user_id'].map(user2idx)

        train['isbn'] = train['isbn'].map(isbn2idx)
        sub['isbn'] = sub['isbn'].map(isbn2idx)

        img_train = self.prepare_img_data(train, books, user2idx, isbn2idx, train=True)
        img_test = self.prepare_img_data(test, books, user2idx, isbn2idx, train=False)

        data = {
            'train': train,
            'test': test.drop(['rating'], axis=1),
            'users': users,
            'books': books,
            'sub': sub,
            'idx2user': idx2user,
            'idx2isbn': idx2isbn,
            'user2idx': user2idx,
            'isbn2idx': isbn2idx,
            'img_train': img_train,
            'img_test': img_test,
        }
        return data

    def prepare_img_data(self, df, books, user2idx, isbn2idx, train=False):
        """
        :param df: <pd.DataFrame> source dataframe
        :param books: <pd.DataFrame>
        :param user2idx: <Dict> user information
        :param isbn2idx: <Dict> index information of each book
        :param train: <bool>
        """
        books_ = books.copy()
        books_['isbn'] = books_['isbn'].map(isbn2idx)

        if train:
            df_ = df.copy()
        else:
            df_ = df.copy()
            df_['user_id'] = df_['user_id'].map(user2idx)
            df_['isbn'] = df_['isbn'].map(isbn2idx)

        df_ = pd.merge(df_, books_[['isbn', 'img_path']], on='isbn', how='left')
        df_['img_path'] = df_['img_path'].apply(lambda x: self.data_path + x)
        img_vector_df = df_[['img_path']].drop_duplicates().reset_index(drop=True).copy()
        data_box = []

        for idx, path in tqdm(enumerate(sorted(img_vector_df['img_path']))):
            data = self.get_image_vector(path)
            if data.size()[0] == 3:
                data_box.append(np.array(data))
            else:
                data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))

        img_vector_df['img_vector'] = data_box
        df_ = pd.merge(df_, img_vector_df, on='img_path', how='left')
        return df_

    def imafe_data_loader(self, data):
        train_dataset = ImageDataset(
            data['X_train'][['user_id', 'isbn']].values,
            data['X_train']['img_vector'].values,
            data['y_train'].values
        )
        valid_dataset = ImageDataset(
            data['X_valid'][['user_id', 'isbn']].values,
            data['X_valid']['img_vector'].values,
            data['y_valid'].values
        )
        test_dataset = ImageDataset(
            data['img_test'][['user_id', 'isbn']].values,
            data['img_test']['img_vector'].values,
            data['img_test']['rating'].values
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=True
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False
        )

        data['train_dataloader'] = train_dataloader
        data['valid_dataloader'] = valid_dataloader
        data['test_dataloader'] = test_dataloader
        return data

    def split_dataset(self, data):
        x_train, x_valid, y_train, y_valid = train_test_split(
            data['img_train'][['user_id', 'isbn', 'img_vector']],
            data['img_train']['rating'],
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=self.data_shuffle
        )
        data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = x_train, x_valid, y_train, y_valid
        return data

    @classmethod
    def get_image_vector(cls, path):
        img = Image.open(path)
        scale = transforms.Resize((32, 32))
        tensor = transforms.ToTensor()
        img_fe = Variable(tensor(scale(img)))
        return img_fe


class ImageDataset(Dataset):
    def __init__(self, user_isbn_vector, img_vector, label):
        """
        :parm user_isbn_vector: <np.ndarray> vectorized user and isbn data
        :parm img_vector: <np.ndarray> vectorized image data
        :parm label: <np.ndarray> vectorized validation data
        """
        self.user_isbn_vector = user_isbn_vector
        self.img_vector = img_vector
        self.label = label

    def __len__(self):
        return self.user_isbn_vector.shape[0]

    def __getitem__(self, i):
        item = {
            'user_isbn_vector': torch.tensor(self.user_isbn_vector[i], dtype=torch.long),
            'img_vector': torch.tensor(self.img_vector[i], dtype=torch.float32),
            'label': torch.tensor(self.label[i], dtype=torch.float32)
        }
        return item
