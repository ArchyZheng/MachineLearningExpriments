import unittest
import LinearRegression


class ModelTest(unittest.TestCase):
    def test_forward(self):
        model = LinearRegression.model(input_length=2)
        dataset = LinearRegression.Dataset(num_train=1000, num_test=100)
        dataset.setup(stage='None')
        train_dataloader = dataset.train_dataloader()
        X, y = next(iter(train_dataloader))
        y_hat = model(X)
        self.assertEqual(len(y_hat), 32)  # add assertion here


if __name__ == '__main__':
    unittest.main()
