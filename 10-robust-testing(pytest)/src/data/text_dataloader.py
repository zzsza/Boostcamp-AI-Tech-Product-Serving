import os
import re
import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
from tqdm.auto import tqdm
from nltk import tokenize
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
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
        users = pd.read_csv(os.path.join(self.data_path, "users.csv"))
        books = pd.read_csv(os.path.join(self.data_path, "books.csv"))
        train = pd.read_csv(os.path.join(self.data_path, "train_ratings.csv"))
        test = pd.read_csv(os.path.join(self.data_path, "test_ratings.csv"))
        sub = pd.read_csv(os.path.join(self.data_path, "submit", "sample_submission.csv"))

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

        train_set = self.generate_summary_vectors(
            df=train,
            books=books,
            user2idx=user2idx,
            isbn2idx=isbn2idx,
            device=self.device,
            train=True,
            user_summary_merge_vector=self.vector_create,
            item_summary_vector=self.vector_create
        )

        test_set = self.generate_summary_vectors(
            df=test,
            books=books,
            user2idx=user2idx,
            isbn2idx=isbn2idx,
            device=self.device,
            train=False,
            user_summary_merge_vector=self.vector_create,
            item_summary_vector=self.vector_create
        )
        data = {
            'train': train,
            'test': test,
            'users': users,
            'books': books,
            'sub': sub,
            'idx2user': idx2user,
            'idx2isbn': idx2isbn,
            'user2idx': user2idx,
            'isbn2idx': isbn2idx,
            'text_train': train_set,
            'text_test': test_set,
        }
        return data

    def generate_summary_vectors(
            self,
            df,
            books,
            user2idx,
            isbn2idx,
            device,
            train=False,
            user_summary_merge_vector=False,
            item_summary_vector=False
    ):
        books_ = books.copy()
        books_['isbn'] = books_['isbn'].map(isbn2idx)

        if train == True:
            df_ = df.copy()
        else:
            df_ = df.copy()
            df_['user_id'] = df_['user_id'].map(user2idx)
            df_['isbn'] = df_['isbn'].map(isbn2idx)

        df_ = pd.merge(df_, books_[['isbn', 'summary']], on='isbn', how='left')
        df_['summary'].fillna('None', inplace=True)
        df_['summary'] = df_['summary'].apply(lambda x: self.clean_text(x))
        df_['summary'].replace({'': 'None', ' ': 'None'}, inplace=True)
        df_['summary_length'] = df_['summary'].apply(lambda x: len(x))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased').to(device)

        output_path = os.path.join(self.data_path, "text_vector")
        train_item_vetor_path = os.path.join(output_path, "train_item_summary_vector.npy")
        test_item_vetor_path = os.path.join(output_path, "test_item_summary_vector.npy")
        train_user_vetor_path = os.path.join(output_path, "train_user_summary_merge_vector.npy")
        test_user_vetor_path = os.path.join(output_path, "test_user_summary_merge_vector.npy")

        if user_summary_merge_vector and item_summary_vector:
            print('- Create User Summary Merge Vector')
            user_summary_merge_vector_list = []
            for user in tqdm(df_['user_id'].unique()):
                vector = self.text_to_vector(self.merge_summary(df_, user, 5), tokenizer, model, device)
                user_summary_merge_vector_list.append(vector)
            user_review_text_df = pd.DataFrame(df_['user_id'].unique(), columns=['user_id'])
            user_review_text_df['user_summary_merge_vector'] = user_summary_merge_vector_list
            vector = np.concatenate([
                user_review_text_df['user_id'].values.reshape(1, -1),
                user_review_text_df['user_summary_merge_vector'].values.reshape(1, -1)
            ])

            print('- Checking text output path:', output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if train:
                np.save(train_user_vetor_path, vector)
            else:
                np.save(test_user_vetor_path, vector)

            print('- Create Item Summary Vector')
            item_summary_vector_list = []
            books_text_df = df_[['isbn', 'summary']].copy()
            books_text_df = books_text_df.drop_duplicates().reset_index(drop=True)
            books_text_df['summary'].fillna('None', inplace=True)
            for summary in tqdm(books_text_df['summary']):
                vector = self.text_to_vector(summary, tokenizer, model, device)
                item_summary_vector_list.append(vector)
            books_text_df['item_summary_vector'] = item_summary_vector_list
            vector = np.concatenate([
                books_text_df['isbn'].values.reshape(1, -1),
                books_text_df['item_summary_vector'].values.reshape(1, -1)
            ])

            print('- Checking output path:', output_path)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            if train:
                np.save(train_item_vetor_path, vector)
            else:
                np.save(test_item_vetor_path, vector)
        else:
            print('- Check Vectorizer')
            print('- Vector Load')
            if train:
                user = np.load(train_user_vetor_path, allow_pickle=True)
            else:
                user = np.load(test_user_vetor_path, allow_pickle=True)
            user_review_text_df = pd.DataFrame([user[0], user[1]]).T
            user_review_text_df.columns = ['user_id', 'user_summary_merge_vector']
            user_review_text_df['user_id'] = user_review_text_df['user_id'].astype('int')

            if train:
                item = np.load(train_item_vetor_path, allow_pickle=True)
            else:
                item = np.load(test_item_vetor_path, allow_pickle=True)
            books_text_df = pd.DataFrame([item[0], item[1]]).T
            books_text_df.columns = ['isbn', 'item_summary_vector']
            books_text_df['isbn'] = books_text_df['isbn'].astype('int')

        df_ = pd.merge(df_, user_review_text_df, on='user_id', how='left')
        df_ = pd.merge(df_, books_text_df[['isbn', 'item_summary_vector']], on='isbn', how='left')
        return df_

    def processing(self, target):
        """
        Split and load the model dataset
        """
        train_dataset = DeepConnDataset(
            target['X_train'][['user_id', 'isbn']].values,
            target['X_train']['user_summary_merge_vector'].values,
            target['X_train']['item_summary_vector'].values,
            target['y_train'].values
        )
        valid_dataset = DeepConnDataset(
            target['X_valid'][['user_id', 'isbn']].values,
            target['X_valid']['user_summary_merge_vector'].values,
            target['X_valid']['item_summary_vector'].values,
            target['y_valid'].values
        )
        test_dataset = DeepConnDataset(
            target['text_test'][['user_id', 'isbn']].values,
            target['text_test']['user_summary_merge_vector'].values,
            target['text_test']['item_summary_vector'].values,
            target['text_test']['rating'].values
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=self.data_shuffle,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=self.data_shuffle,
        )
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=self.data_shuffle,
        )

        target['train_dataloader'] = train_dataloader
        target['valid_dataloader'] = valid_dataloader
        target['test_dataloader'] = test_dataloader

        return target

    def split_dataset(self, target):
        x_train, x_valid, y_train, y_valid = train_test_split(
            target['text_train'][
                ['user_id', 'isbn', 'user_summary_merge_vector', 'item_summary_vector']
            ],
            target['text_train']['rating'],
            test_size=self.test_size,
            random_state=self.seed,
            shuffle=self.data_shuffle
        )

        target['X_train'], target['X_valid'], target['y_train'], target['y_valid'] = x_train, x_valid, y_train, y_valid
        return target

    @classmethod
    def clean_text(cls, summary):
        summary = re.sub("[.,\'\"''""!?]", "", summary)
        summary = re.sub("[^0-9a-zA-Z\\s]", " ", summary)
        summary = re.sub("\s+", " ", summary)
        summary = summary.lower()
        return summary

    @classmethod
    def merge_summary(cls, df, user_id, max_summary):
        return " ".join(
            df[df['user_id'] == user_id]
            .sort_values(by='summary_length', ascending=False)['summary']
            .values[:max_summary]
        )

    @classmethod
    def text_to_vector(cls, text, tokenizer, model, device):
        for sent in tokenize.sent_tokenize(text):
            text_ = "[CLS] " + sent + " [SEP]"
            tokenized = tokenizer.tokenize(text_)
            indexed = tokenizer.convert_tokens_to_ids(tokenized)
            segments_idx = [1] * len(tokenized)
            token_tensor = torch.tensor([indexed])
            sgments_tensor = torch.tensor([segments_idx])
            with torch.no_grad():
                outputs = model(token_tensor.to(device), sgments_tensor.to(device))
                encode_layers = outputs[0]
                sentence_embedding = torch.mean(encode_layers[0], dim=0)
        return sentence_embedding.cpu().detach().numpy()


class DeepConnDataset(Dataset):
    def __init__(self, user_isbn_vector, user_summary_merge_vector, item_summary_vector, label):
        """
        :parm <np.ndarray> user_isbn_vector: vectorized user and isbn data
        :parm <np.ndarray> user_summary_merge_vector: vectorized user and summary data
        :parm <np.ndarray> item_summary_vector: vectorized summary data
        :parm <np.ndarray> label: vectorized validation data
        """
        self.user_isbn_vector = user_isbn_vector
        self.user_summary_merge_vector = user_summary_merge_vector
        self.item_summary_vector = item_summary_vector
        self.label = label

    def __len__(self):
        return self.user_isbn_vector.shape[0]

    def __getitem__(self, i):
        item = {
            'user_isbn_vector': torch.tensor(
                self.user_isbn_vector[i],
                dtype=torch.long
            ),
            'user_summary_merge_vector': torch.tensor(
                self.user_summary_merge_vector[i].reshape(-1, 1),
                dtype=torch.float32
            ),
            'item_summary_vector': torch.tensor(
                self.item_summary_vector[i].reshape(-1, 1),
                dtype=torch.float32
            ),
            'label': torch.tensor(
                self.label[i],
                dtype=torch.float32
            ),
        }

        return item
