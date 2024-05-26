import datetime

import mlflow
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from io import BytesIO
from logger import Logger
from torch.optim import lr_scheduler, Adam
logger = Logger(__name__).get_logger()


class ModelTraining:
    def __init__(self):
        self.val_loss_history = []
    @staticmethod
    def train(model, train_loader, val_loader, s3fs, config):
        """

        :param model:
        :param train_loader:
        :param val_loader:
        :param s3fs:
        :param config:
        :return:
        """
        criterion = ModelTraining.bce_dice_loss
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

        # with mlflow.start_run():
        ModelTraining.train_model(model, train_loader, val_loader, s3fs, config, criterion, optimizer, scheduler)

    def train_model(self,model, train_loader, val_loader, s3fs, config, criterion, optimizer, scheduler, experiment):
        """

        :param model:
        :param train_loader:
        :param val_loader:
        :param s3fs:
        :param config:
        :param criterion:
        :param optimizer:
        :param scheduler:
        :return:
        """

        model.to(config.device)
        best_val_loss = float('inf')
        epochs_no_improve = 0
        train_loss_history = []


        if config.mlflow_logging:
            mlflow.start_run()
            mlflow.pytorch.autolog()
        if config.comet_logging:
            experiment.log_parameters(
                {"number_of_epochs": config.number_of_epochs,
                 "learning_rate": optimizer.param_groups[0]['lr'],
                 "l2": config.weight_decay,
                 "patience": config.patience,
                 "scheduler": scheduler,
                 "dropout": config.dropout,
                 "training_path": config.training_images_path,
                 "validation_path": config.validation_images_path,
                 "model_name": config.model_name
                 })

        for epoch in range(config.number_of_epochs):
            model.train()
            running_loss = 0.0

            for images, masks in train_loader:
                images, masks = images.to(config.device), masks.to(config.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)

            train_loss = running_loss / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(config.device), masks.to(config.device)
                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)
            logger.info(
                f'Epoch {epoch + 1}/{config.number_of_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time {datetime.datetime.now()}')

            # scheduler.step()
            scheduler.step(val_loss)
            current_lr = scheduler.get_last_lr()[0]  # get_last_lr() returns a list
            logger.info(f'Current learning rate: {current_lr:.6f}')

            if config.comet_logging:
                experiment.log_metric("train_loss", train_loss, step=epoch)
                experiment.log_metric("val_loss", val_loss, step=epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                with s3fs.open(config.model_path, 'wb') as f:
                    torch.save(model.state_dict(), f)
                logger.info(f"Saving model in epoch {epoch + 1}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == config.patience:
                    logger.info("Early stopping")
                    break

            self.val_loss_history.append(val_loss)
            train_loss_history.append(train_loss)
        logger.info('Finished Training')

        if config.mlflow_logging:
            mlflow.end_run()

        # if config.comet_logging:
        #     config.comet_experiment.end()

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Train Loss')
        plt.plot(range(1, len(self.val_loss_history) + 1), self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss During Training')
        plt.legend()
        plt.grid(True)

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        with s3fs.open(config.plot_path, 'wb') as f:
            f.write(buf.getvalue())

        plt.show()
        buf.close()

    @staticmethod
    def dice_coef_loss(inputs, target):
        """"""
        smooth = 1.0
        intersection = 2.0 * ((target * inputs).sum()) + smooth
        union = target.sum() + inputs.sum() + smooth

        return 1 - (intersection / union)

    @staticmethod
    def bce_dice_loss(inputs, target):
        """"""
        inputs = inputs.float()
        target = target.float()

        dice_score = ModelTraining.dice_coef_loss(inputs, target)
        bce_score = nn.BCELoss()
        bce_loss = bce_score(inputs, target)

        return bce_loss + dice_score
