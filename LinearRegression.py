# %% get dataset
import lightning as pl
import torch
from torch.utils.data import Dataset


class LinearDataset(Dataset):
    def __init__(self, w, b, noise, num=1100):
        super().__init__()
        self.w = w
        self.b = b
        self.num_total = num
        self.noise = noise
        self.X = torch.randn(self.num_total, len(self.w))
        noise = torch.randn(self.num_total, 1) * self.noise
        self.y = torch.matmul(self.X, self.w.reshape((-1, 1))) + self.b + noise

    def __len__(self):
        return self.num_total

    def __getitem__(self, item):
        return self.X[item], self.y[item]


# %% pl.Dataset
class Dataset(pl.LightningDataModule):
    def __init__(self, num_train, num_test):
        super().__init__()
        self.num_train = num_train
        self.num_test = num_test
        self.dataset = LinearDataset(w=torch.tensor([4, 2.3]), b=9.8, noise=0.3)

    def setup(self, stage: str) -> None:
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset=self.dataset,
            lengths=[self.num_train, self.num_test]
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=32)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(dataset=self.test_dataset, batch_size=32)


# %% pytorch module
import torch.nn as nn


class model(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(input_length, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, input):
        return self.l1(input)


# %% pytorch lightning module
class LightModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.module = model(input_length=2)
        self.loss = torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.module(x)
        loss = self.loss(y_hat, y)
        self.log('train loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.module.parameters(), lr=1e-3)
        return optimizer

# %%
dataset = Dataset(num_train=1000, num_test=100)
dataset.setup(stage='None')
train_dataloader = dataset.train_dataloader()
test_dataloader = dataset.test_dataloader()

from lightning.pytorch.loggers import CometLogger

comet_logger = CometLogger(
    api_key='lNyK4LLQynW9EQrhnWPWfvHTk',
    project_name='LinearRegression_D2l'
)
train = LightModule()
trainer = pl.Trainer(logger=[comet_logger], max_epochs=100)
trainer.fit(model=train, train_dataloaders=train_dataloader)