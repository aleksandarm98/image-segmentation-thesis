from config import Config
from logger import Logger
from utils import Utils
from model import UNetPlusPlus
from model_att import AttentionUNet
from processing import ModelTraining
from torch.utils.data import DataLoader
from storage import CarotidUltrasoundDataset


logger = Logger(__name__).get_logger()

def main():

    logger.info(" This is main ")
    logger.info("Variables initialisation")
    config = Config()
    logger.info("Data loading")
    train_dataset = CarotidUltrasoundDataset(config.s3fs, config.training_images_path, config.training_labels_path, True)
    val_dataset = CarotidUltrasoundDataset(config.s3fs, config.validation_images_path, config.validation_labels_path, True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    logger.info("Running hyperparameter optimization")
    best_params = Utils.run_optimization(train_loader, val_loader, config.s3fs, config)

    # Update config with best hyperparameters
    config.learning_rate = best_params['lr']
    config.weight_decay = best_params['weight_decay']
    config.dropout = best_params['dropout_rate']

    logger.info("Model initialisation")
    #model = AttentionUNet(config.hidden_size)

    logger.info("Model training")
    #ModelTraining.train(model, train_loader, val_loader, config.s3fs, config)


    logger.info("Validation")

    logger.info("Final logging")

    logger.info("Execution time")


if __name__ == '__main__':
    main()

