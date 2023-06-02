import unittest

import torch

from LinearRegression import LinearDataset
from LinearRegression import Dataset


class DatasetTest(unittest.TestCase):
    def test_length_of_dataset(self):
        dataset = LinearDataset(w=torch.Tensor([10, 2]), b=0.3, noise=0.3)
        self.assertEqual(len(dataset), 1100)

    def test_iter_dataset(self):
        dataset = LinearDataset(w=torch.Tensor([10, 2]), b=0.3, noise=0.3)
        X, y = next(iter(dataset))
        w = torch.tensor([10, 2], dtype=torch.float32)
        error = y - (torch.matmul(X, w.T) + 0.3)
        self.assertTrue(error < 0.5)

    def test_dataloader(self):
        dataset = Dataset(num_train=1000, num_test=100)
        dataset.setup(stage='None')
        train_dataloader = dataset.train_dataloader()
        X, y = next(iter(train_dataloader))
        self.assertEqual(len(X), 32)
        self.assertEqual(len(y), 32)


if __name__ == '__main__':
    unittest.main()
