import os
import cv2
import optuna
import torchvision.transforms as transforms
import numpy as np

from comet_ml import Experiment
from PIL import Image
from torch.optim import Adam
from torch.optim import lr_scheduler
from processing import ModelTraining
from model import UNetPlusPlus
from model_att import AttentionUNet


class Utils:
    @staticmethod
    def objective(trial, train_loader, val_loader, s3fs, config):
        experiment = Experiment(api_key=os.environ.get("COMET_SECRET_KEY"),
                                project_name=config.project_name,
                                workspace=config.workspace)

        experiment.set_name(f"{config.model_name}-{config.dataset_name}-{config.date}")

        lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.2, 0.4)
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64, 128, 256])
        batch_size = trial.suggest_categorical('batch_size', [6, 8, 16])

        config.learning_rate = lr
        config.weight_decay = weight_decay
        config.dropout = dropout_rate
        config.hidden_size = hidden_size
        config.batch_size = batch_size

        if config.model_name == "UnetPlusPlus":
            model = UNetPlusPlus(hidden_size)
        else:
            model = AttentionUNet(hidden_size)

        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler1 = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)
        scheduler2 = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

        scheduler = trial.suggest_categorical('scheduler', [scheduler1, scheduler2])

        experiment.log_parameters({"lr": lr,
                                   "weight_decay": weight_decay,
                                   "dropout_rate": dropout_rate,
                                   "hidden_size": hidden_size})

        training = ModelTraining()

        training.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            s3fs=s3fs,
            config=config,
            criterion=ModelTraining.bce_dice_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            experiment=experiment,
        )

        # experiment.log_metric("val_loss", training.val_loss)
        experiment.end()
        return min(training.val_loss_history)

    @staticmethod
    def run_optimization(train_loader, val_loader, s3fs, config):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: Utils.objective(trial, train_loader, val_loader, s3fs, config), n_trials=70)

        print("Best hyperparameters: ", study.best_params)
        return study.best_params


class CLAHETransform:
    """ Transformacija koja primenjuje CLAHE na PIL Image """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        #TODO add option for RGB aside L
        img_np = np.array(img.convert('L'))
        img_np = self.clahe.apply(img_np)
        return Image.fromarray(img_np, 'L')


transform_input = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),  # Resize slike na 512x512 piksela
    CLAHETransform(),
    #transforms.RandomHorizontalFlip(p=0.5),  # Nasumično horizontalno prevrtanje slike
    #transforms.RandomRotation(degrees=10),  # Nasumična rotacija do 10 stepeni
    transforms.ToTensor()  # Konvertovanje slike u PyTorch Tensor
    #transforms.Normalize([0.5], [0.5])
])



