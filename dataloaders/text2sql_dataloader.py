from pathlib import Path
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from typing import Callable

from text2sql.datasets.text2sql_dataset import Text2SQLDataset
from text2sql.datasets.prepcosessing_utils import split_data_ensuring_representation, load_glove
from text2sql.datasets.visualizer import display_and_save_dataset_statistics


class Text2SQLDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for the Text2SQL task.
    """
    def __init__(self, data_dir: Path, embedding_dim, batch_size: int,collate_fn:Callable):
        """
        Initialize the DataModule.
        Args:
            data_dir: Path to the data directory
            embedding_dim: Embedding dimension
            batch_size: Batch size
            collate_fn: Collate function for the DataLoader
        """
        super().__init__()

        # Load GloVe embeddings
        self.glove = load_glove(embedding_dim)

        # Create a vocabulary index
        self.vocab_index = {word: i for i, word in enumerate(self.glove.itos)}

        # Set the batch size
        self.batch_size = batch_size

        # Set the data directory
        self.data_dir = Path(data_dir)

        # Ensure the data directory exists
        assert self.data_dir.exists(), "Data directory not found!"

        # Set the collate function
        self.collate_fn = collate_fn

    def setup(self, stage=None,get_stats=False):
        """
        Setup the data splits and datasets.
        Args:
            stage: Stage of training (train, val, test)
            get_stats: Flag to get statistics of the dataset
        """

        # Load the data and split into train, validation, and test sets
        self.data_splits = split_data_ensuring_representation(self.data_dir)

        # Display and save dataset statistics
        if get_stats:
            output_dir = Path('/mobileye/RPT/users/kfirs/kfir_project/text2sql/text2sql/dataset_statistics')
            display_and_save_dataset_statistics(self.data_splits['train'],output_dir=output_dir,df_type='train') # Display and save dataset statistics
            display_and_save_dataset_statistics(self.data_splits['dev'],
                                                output_dir=output_dir,df_type='dev')  # Display and save dataset statistics
            display_and_save_dataset_statistics(self.data_splits['test'],
                                                output_dir=output_dir,df_type='test')  # Display and save dataset statistics

        # Create datasets
        self.train_dataset = Text2SQLDataset(self.data_splits['train'], vocab_index=self.vocab_index, augment=True) # Augment the training dataset
        self.val_dataset = Text2SQLDataset(self.data_splits['dev'], vocab_index=self.vocab_index) # Validation dataset
        self.test_dataset = Text2SQLDataset(self.data_splits['test'], vocab_index=self.vocab_index) # Test dataset

        # Set the number of classes
        self.num_classes = len(self.data_splits['train']['label'].unique())

    def train_dataloader(self, num_workers=4):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self, num_workers=4):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=num_workers,collate_fn=self.collate_fn)

    def test_dataloader(self, num_workers=4):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=num_workers,collate_fn=self.collate_fn)

