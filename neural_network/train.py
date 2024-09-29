import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.core.datamodule import LightningDataModule
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

from dataset import MacchinineDataModule
from model import MLP

### NOTE:
# - Lowering kappa also lowers the total error, but somehow lowering it also causes the getImage function to have more problems in finding a viable sample.

def train(args):
    macchinine_data = MacchinineDataModule(batch_size=args.batch_size, num_workers=args.num_workers, train_data_path= args.train_data_path,  eval_data_path=args.eval_data_path, num_samples=args.num_samples, kappa=args.kappa, debug=args.debug.lower()=="true")
    model = MLP(input_size=args.input_size, hidden_sizes=args.hidden_sizes, output_size=args.output_size, debug=args.debug.lower()=="true")
    if args.debug.lower() == "true":
        print(model)
    # Create a logger with a custom version name
    logger = TensorBoardLogger('lightning_logs', name=args.name)
    trainer = Trainer(max_epochs=args.epochs, devices=1, accelerator='auto', logger=logger)
    # trainer = Trainer(max_epochs=args.epochs, devices=1, accelerator='auto')
    trainer.fit(model, macchinine_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a simple MLP on MNIST dataset")
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for training (default: 32)')
    parser.add_argument('--num_samples', type=int, default=60000, help='number of samples to train (default: 60000)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers')
    parser.add_argument('--input_size', type=int, default=6, help='input size of MLP (default: 784)')
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[128, 64], help='sizes of hidden layers (default: [128, 64])')
    parser.add_argument('--output_size', type=int, default=2, help='output size of MLP (default: 10)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--train_data_path', type=str, default=None, help='path to the dataset cvs')
    parser.add_argument('--eval_data_path', type=str, default=None, help='path to the dataset cvs')
    parser.add_argument('--debug', type=str, default="False", help='debugging mode')
    parser.add_argument('--kappa', type=float, default=0.4, help='distortion coefficient')
    parser.add_argument('--name', type=str, default=None, help='name of the model')

    args = parser.parse_args()
    train(args)