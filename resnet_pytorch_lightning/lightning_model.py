
import torch
import torch.nn as nn
import lightning.pytorch as pl
import torch.nn.functional as F
from torchvision import models
from torchmetrics.functional import precision, recall, f1_score

class LightningModel(pl.LightningModule):
    def __init__(self, num_classes=10):
        super().__init__()
        self.model = models.resnet50(weights='IMAGENET1K_V2')
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean() * 100
        self.precision = precision(y_hat, y,task="multiclass", num_classes=10)
        self.recall = recall(y_hat, y,task="multiclass", num_classes=10)
        self.f1 = f1_score(y_hat, y,task="multiclass", num_classes=10)
        self.log_dict({'train_loss': loss, 'train_acc': acc, 'train_precision': self.precision, 'train_recall': self.recall, 'train_f1': self.f1}, 
        on_step=False, on_epoch=True, logger=True)

        return {'loss': loss, 'acc': acc, 'precision': self.precision, 'recall': self.recall, 'f1': self.f1}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean() * 100
        self.precision = precision(y_hat, y,task="multiclass", num_classes=10)
        self.recall = recall(y_hat, y,task="multiclass", num_classes=10)
        self.f1 = f1_score(y_hat, y,task="multiclass", num_classes=10)
        self.log_dict({'val_loss': loss, 'val_acc': acc, 'val_precision': self.precision, 'val_recall': self.recall, 'val_f1': self.f1}, 
        on_step=False, on_epoch=True, logger=True)

        return {'loss': loss, 'acc': acc, 'precision': self.precision, 'recall': self.recall, 'f1': self.f1}
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (torch.argmax(y_hat, dim=1) == y).float().mean() * 100
        self.log_dict({'test_loss': loss, 'test_acc': acc}, logger=True)

        return {'loss': loss, 'acc': acc}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}