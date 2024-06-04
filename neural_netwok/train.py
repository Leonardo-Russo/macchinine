import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from dataset import MachinineDataModule
from model import MLP

def main(args):
    mnist_data = MachinineDataModule(batch_size=args.batch_size, train_data_path= args.train_data_path,  eval_data_path=args.eval_data_path, num_samples=args.num_samples)
    mlp_model = MLP(input_size=args.input_size, hidden_sizes=args.hidden_sizes, output_size=args.output_size)
    trainer = Trainer(max_epochs=args.epochs, devices=1, accelerator='auto')
    trainer.fit(mlp_model, mnist_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST dataset")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--num_samples', type=int, default=60000, help='number of samples to train (default: 60000)')
    parser.add_argument('--input_size', type=int, default=4, help='input size of MLP (default: 784)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[20, 10], help='sizes of hidden layers (default: [128, 64])')
    parser.add_argument('--output_size', type=int, default=2, help='output size of MLP (default: 10)')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train (default: 1)')
    parser.add_argument('--train_data_path', type=str, default=None, help='path to the dataset cvs')
    parser.add_argument('--eval_data_path', type=str, default=None, help='path to the dataset cvs')

    args = parser.parse_args()
    main(args)