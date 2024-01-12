from loguru import logger
from pydantic import ValidationError

import nltk
import data
import models
from trainer import Trainer

model = None


class ModelOptions:
    def __init__(self, model_name, data_path, model_path, accelerator):
        self.model_name = model_name
        self.data_path = data_path
        self.model_path = model_path
        self.accelerator = accelerator

    def load_model(
            self,
            embeddings,
            embed_dim: int = 16,
            **kwargs
    ):
        global model
        logger.info(f"Loading model - {self.model_name}")

        if self.model_name == "deepconn":
            deepconn_embed_dim = kwargs.get("deepconn_embed_dim", 32)
            word_dim = kwargs.get("word_dim", 768)
            out_dim = kwargs.get("out_dim", 32)
            conv_1d_out_dim = kwargs.get("conv_1d_out_dim", 50)
            deepconn_latent_dim = kwargs.get("deepconn_latent_dim", 10)
            model = models.DeepCoNN(
                embeddings,
                embed_dim,
                deepconn_embed_dim,
                word_dim,
                out_dim,
                conv_1d_out_dim,
                deepconn_latent_dim,
            ).to(self.accelerator)

        elif self.model_name == "fm":
            model = models.FactorizationMachineModel(
                embeddings,
                embed_dim
            ).to(self.accelerator)

        elif self.model_name == "ffm":
            model = models.FieldAwareFactorizationMachineModel(
                embeddings,
                embed_dim
            ).to(self.accelerator)

        elif self.model_name == "ncf":
            dropout = kwargs.get("dropout", 0.2)
            mlp_dims = kwargs.get("mlp_dims", (16, 16))
            model = models.NeuralCollaborativeFiltering(
                embeddings,
                dropout,
                embed_dim,
                mlp_dims
            ).to(self.accelerator)

        elif self.model_name == "wdn":
            dropout = kwargs.get("dropout", 0.2)
            mlp_dims = kwargs.get("mlp_dims", (16, 16))
            model = models.WideAndDeepModel(
                embeddings,
                dropout,
                embed_dim,
                mlp_dims
            ).to(self.accelerator)

        elif self.model_name == "dcn":
            dropout = kwargs.get("dropout", 0.2)
            mlp_dims = kwargs.get("mlp_dims", (16, 16))
            num_layers = kwargs.get("num_layers", 3)
            model = models.DeepCrossNetworkModel(
                embeddings,
                dropout,
                embed_dim,
                mlp_dims,
                num_layers
            ).to(self.accelerator)

        elif self.model_name == "cnn_fm":
            cnn_embed_dim = kwargs.get("cnn_embed_dim", 64)
            cnn_latent_dim = kwargs.get("cnn_latent_dim", 12)
            model = models.CNNFM(
                embeddings,
                cnn_embed_dim,
                cnn_latent_dim
            ).to(self.accelerator)

        else:
            raise ValidationError("Not defined model.")

    def get_trainer(
            self,
            model,
            lr: float = 1e-3,
            loss_fn: str = "RMSE",
            epochs: int = 10,
            optimizer: str = "ADAM",
    ):
        trainer = Trainer(
            model, self.model_name, self.data_path, self.model_path, lr, loss_fn, epochs, optimizer, self.accelerator
        )
        return trainer

    def get_embedding(
            self,
            vector_create: bool = False,
            data_shuffle: bool = True,
            batch_size: int = 1024,
            seed: int = 42,
            test_size: float = 0.2,
    ):
        embeddings = None
        if self.model_name == "deepconn":
            nltk.download('punkt')
            dataloader = data.TextLoader(
                self.data_path,
                self.accelerator,
                vector_create,
                data_shuffle,
                batch_size,
                seed,
                test_size,
            )
            text_dataloader = dataloader.load_data()
            text_dataloader = dataloader.split_dataset(text_dataloader)
            embeddings = dataloader.processing(text_dataloader)

        elif self.model_name in ("fm", "ffm"):
            dataloader = data.ContextLoader(
                data_shuffle,
                batch_size,
                seed,
                test_size,
            )
            context_dataloader = dataloader.load_data(self.data_path)
            context_dataloader = dataloader.split_dataset(context_dataloader)
            embeddings = dataloader.context_data_loader(context_dataloader)

        elif self.model_name in ("ncf", "wdn", "dcn"):
            dataloader = data.DlLoader(
                data_shuffle,
                batch_size,
                seed,
                test_size,
            )
            dl_dataloader = dataloader.load_data(self.data_path)
            dl_dataloader = dataloader.split_dataset(dl_dataloader)
            embeddings = dataloader.dl_data_loader(dl_dataloader)

        elif self.model_name == "cnn_fm":
            dataloader = data.ImageLoader(
                self.data_path,
                self.accelerator,
                vector_create,
                data_shuffle,
                batch_size,
                seed,
                test_size,
            )
            img_dataloader = dataloader.load_data()
            img_dataloader = dataloader.split_dataset(img_dataloader)
            embeddings = dataloader.imafe_data_loader(img_dataloader)

        else:
            raise ValidationError("Not defined dataloader.")
        return embeddings

    @classmethod
    def get_model(cls):
        global model
        return model
