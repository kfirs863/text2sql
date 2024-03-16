import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.loggers import CometLogger
import torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import LearningRateMonitor
from torch.optim.lr_scheduler import ReduceLROnPlateau


from text2sql.dataloaders.text2sql_dataloader import Text2SQLDataModule
from text2sql.datasets.prepcosessing_utils import load_glove, create_collate_fn


class TextCNN(pl.LightningModule):
    def __init__(self, embedding_dim:int=300, num_classes=2, filter_sizes=(3, 4, 5), num_filters=50, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()

        # Load GloVe embeddings
        embeddings = load_glove(embedding_dim=embedding_dim)

        # Embedding layer
        self.embedding = nn.Embedding(len(embeddings), embeddings.dim, padding_idx=0)

        # Load pre-trained embeddings
        self.embedding.weight.data.copy_(embeddings.vectors)

        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(k, embeddings.dim))
            for k in filter_sizes
        ])

        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Accuracy metric for multi-class classification for train and validation
        self.train_accuracy = Accuracy(num_classes=num_classes, average='macro', task='multiclass')
        self.val_accuracy = Accuracy(num_classes=num_classes, average='macro', task='multiclass')

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_length, embedding_dim]
        x = x.unsqueeze(1)  # Add channel dimension: [batch_size, 1, seq_length, embedding_dim]

        # Apply convolution and activation for each filter size
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # [batch_size, num_filters, some_length]

        # Apply global max pooling for each feature map
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [batch_size, num_filters]

        x = torch.cat(x, 1)  # Concatenate the convolutional features
        x = self.dropout(x)  # Apply dropout
        x = self.fc(x)  # Pass through the fully connected layer
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3,weight_decay=1e-5)
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6,threshold=0.01),
            'monitor': 'val_acc',
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]


if __name__ == '__main__':
    # Set the embedding dimension and data directory
    embedding_dim = 300
    data_dir = Path(__file__).parent / 'data'

    # Set the filter sizes and number of filters
    filter_sizes = [2, 3, 4,5]
    num_filters = 200

    # Create a collate function
    collate_fn = create_collate_fn(min_seq_length=max(filter_sizes))

    # Initialize the data module
    data_module = Text2SQLDataModule(data_dir=data_dir,embedding_dim=embedding_dim,batch_size=8,collate_fn=collate_fn)
    data_module.setup(get_stats=False)  # Load and prepare data

    # Initialize the model
    model = TextCNN(
        embedding_dim=embedding_dim,
        num_classes=data_module.num_classes,  # Set the number of classes
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        dropout=0.5
    )

    # Initialize a CometML experiment
    comet_logger = CometLogger(
        api_key="jsPqM9osr1ZfIKWiEeiAlitCa",
        project_name="TextCNN",
    )

    # Initialize a trainer
    trainer = pl.Trainer(
        max_epochs=50,
        logger=comet_logger,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        callbacks=[LearningRateMonitor(logging_interval='epoch')],
    )

    # Fit the model
    trainer.fit(model, data_module.train_dataloader(num_workers=16), val_dataloaders=data_module.val_dataloader(num_workers=16))

    # Test the model
    trainer.test(model, data_module.test_dataloader(num_workers=16))