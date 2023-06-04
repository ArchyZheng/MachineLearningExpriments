# %% prepare data
import torchvision
import lightning as pl
import torch


# %%
class DatasetModule(pl.LightningDataModule):
    def __init__(self, resize=(28, 28)):
        super().__init__()
        # transform the data into tensor and resize it into a formal shape
        self.test_dataset = None
        normal = torchvision.transforms.Compose(
            [torchvision.transforms.Resize(resize),
             torchvision.transforms.ToTensor()]
        )
        # download the dataset
        self.train_dataset = torchvision.datasets.FashionMNIST(root='data', train=True, transform=normal, download=True)
        self.val_dataset = torchvision.datasets.FashionMNIST(root='data', train=False, transform=normal, download=True)

    def setup(self, stage: str) -> None:
        # split the val_dataset into val_dataset and test_dataset
        self.val_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset=self.val_dataset,
            lengths=[int(len(self.val_dataset) * 0.8), int(len(self.val_dataset) * 0.2)]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=32)


# %%
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=28 * 28, out_features=10),
        )

    def forward(self, input_image):
        return self.model(input_image)


class ModelModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.loss = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log('train loss', loss)
        return loss

    def validation_step(self, batch, idx):
        X, y = batch
        y_hat = self.model(X)
        loss = self.loss(y_hat, y)
        self.log('validation loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=1e-3)


# %%
from lightning.pytorch.loggers import CometLogger

comet = CometLogger(
    api_key='lNyK4LLQynW9EQrhnWPWfvHTk',
    project_name='SoftmaxRegression'
)

trainer = pl.Trainer(logger=[comet], max_epochs=10)
model = ModelModule()
dataset = DatasetModule()
dataset.setup(stage='None')
trainer.fit(model=model, train_dataloaders=dataset.train_dataloader(), val_dataloaders=dataset.val_dataloader())