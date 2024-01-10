import os
import tqdm
import torch
import torch.nn as nn
from torch.nn import MSELoss
from torch.optim import SGD, Adam


class Trainer:
    def __init__(
            self,
            model,
            model_name,
            data_path,
            model_path,
            lr,
            loss_fn,
            epochs,
            optimizer,
            accelerator,
    ):
        self.device = accelerator

        self.model = model
        self.model_name = model_name
        self.data_path = data_path
        self.model_path = model_path

        self.lr = lr
        self.loss_option = loss_fn
        self.epochs = epochs
        self.optimizer = optimizer

    def train(self, dataloader):
        minimum_loss = 999999999

        loss_fn = None
        if self.loss_option == 'MSE':
            loss_fn = MSELoss()
        elif self.loss_option == 'RMSE':
            loss_fn = RMSELoss()

        optimizer = None
        if self.optimizer == 'SGD':
            optimizer = SGD(self.model.parameters(), lr=self.lr)
        elif self.optimizer == 'ADAM':
            optimizer = Adam(self.model.parameters(), lr=self.lr)

        for epoch in tqdm.tqdm(range(self.epochs)):
            self.model.train()
            total_loss = 0
            batch = 0

            for idx, data in enumerate(dataloader['train_dataloader']):
                if self.model_name == "cnn_fm":
                    x, y = [
                        data['user_isbn_vector'].to(self.device),
                        data['img_vector'].to(self.device)
                    ], data['label'].to(self.device)
                elif self.model_name == "deepconn":
                    x, y = (
                        [
                            data['user_isbn_vector'].to(self.device),
                            data['user_summary_merge_vector'].to(self.device),
                            data['item_summary_vector'].to(self.device)
                        ],
                        data['label'].to(self.device)
                    )
                else:
                    x, y = data[0].to(self.device), data[1].to(self.device)

                y_hat = self.model(x)
                loss = loss_fn(y.float(), y_hat)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                batch += 1

            valid_loss = self.valid(dataloader, loss_fn)

            print(f'Epoch: {epoch + 1}, Train_loss: {total_loss / batch:.3f}, valid_loss: {valid_loss:.3f}')
            if minimum_loss > valid_loss:
                minimum_loss = valid_loss
                os.makedirs(self.model_path, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.model_path, "model.pt")
                )

    def test(self, dataloader):
        predicts = list()

        model_name = self.model_name.lower()
        model_output_path = os.path.join(self.model_path, "model.pt")
        self.model.load_state_dict(torch.load(model_output_path, map_location=torch.device(self.device)))
        self.model.eval()

        for idx, data in enumerate(dataloader['test_dataloader']):
            if model_name == 'cnn_fm':
                x, _ = [
                    data['user_isbn_vector'].to(self.device),
                    data['img_vector'].to(self.device)
                ], data['label'].to(self.device)
            elif model_name == 'deepconn':
                x, _ = [
                    data['user_isbn_vector'].to(self.device),
                    data['user_summary_merge_vector'].to(self.device),
                    data['item_summary_vector'].to(self.device)
                ], data['label'].to(self.device)
            else:
                x = data[0].to(self.device)

            y_hat = self.model(x)
            predicts.extend(y_hat.tolist())

        return predicts

    def valid(self, dataloader, loss_fn):
        self.model.eval()

        total_loss = 0
        batch = 0

        for idx, data in enumerate(dataloader['valid_dataloader']):
            if self.model_name == 'CNN_FM':
                x, y = [
                    data['user_isbn_vector'].to(self.device),
                    data['img_vector'].to(self.device)
                ], data['label'].to(self.device)
            elif self.model_name == 'DeepCoNN':
                x, y = [
                    data['user_isbn_vector'].to(self.device),
                    data['user_summary_merge_vector'].to(self.device),
                    data['item_summary_vector'].to(self.device)
                ], data['label'].to(self.device)
            else:
                x, y = data[0].to(self.device), data[1].to(self.device)

            y_hat = self.model(x)
            loss = loss_fn(y.float(), y_hat)
            total_loss += loss.item()
            batch += 1

        valid_loss = total_loss / batch
        return valid_loss


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss
