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


class Utils:
    @staticmethod
    def objective(trial, train_loader, val_loader, s3fs, config):
        experiment = Experiment(api_key=os.environ.get("COMET_SECRET_KEY"),
                                project_name=config.project_name,
                                workspace=config.workspace)

        experiment.set_name(f"{config.model_name}-{config.dataset_name}-{config.date}")
        # Predlozi hiperparametre koje ćemo optimizovati
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-1)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.1, 0.5)

        # Ažuriranje config objekta sa hiperparametrima
        config.learning_rate = lr
        config.weight_decay = weight_decay
        config.dropout = dropout_rate



        # Kreiranje modela sa određenim dropout stopom
        model = UNetPlusPlus()

        # Kreiranje optimizatora i scheduler-a sa određenim hiperparametrima
        optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.1)

        experiment.log_parameters({"lr": lr, "weight_decay": weight_decay, "dropout_rate": dropout_rate})

        training = ModelTraining()

        # Treniranje modela
        training.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            s3fs=s3fs,
            config=config,
            criterion=ModelTraining.bce_dice_loss,
            optimizer=optimizer,
            scheduler=scheduler,
            experiment=experiment
        )

        # experiment.log_metric("val_loss", training.val_loss)
        experiment.end()
        # Povratna vrednost za optimizaciju - ovde koristimo validation loss
        return min(training.val_loss_history)



    @staticmethod
    def run_optimization(train_loader, val_loader, s3fs, config):
        # Pokretanje Optuna studije
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: Utils.objective(trial, train_loader, val_loader, s3fs, config), n_trials=50)

        # Prikaz najboljih hiperparametara
        print("Best hyperparameters: ", study.best_params)
        return study.best_params


class CLAHETransform:
    """ Transformacija koja primenjuje CLAHE na PIL Image """

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img):
        img_np = np.array(img.convert('L'))  # Konvertuj PIL Image u NumPy array i prevedi u grayscale
        img_np = self.clahe.apply(img_np)  # Primena CLAHE
        return Image.fromarray(img_np, 'L')  # Vrati u PIL Image format u grayscale


transform_input = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((512, 512)),  # Resize slike na 512x512 piksela
    CLAHETransform(),  # Primena CLAHE na grayscale sliku
    #transforms.RandomHorizontalFlip(p=0.5),  # Nasumično horizontalno prevrtanje slike
    #transforms.RandomRotation(degrees=10),  # Nasumična rotacija do 10 stepeni
    transforms.ToTensor()  # Konvertovanje slike u PyTorch Tensor
    #transforms.Normalize([0.5], [0.5])
])



