import os
import s3fs
import yaml
from datetime import datetime
from torch import device, cuda



class Config:
    def __init__(self):
        self.s3fs = s3fs.S3FileSystem(key=os.environ.get('AWS_ACCESS_KEY_ID'),
                                      secret=os.environ.get('AWS_SECRET_ACCESS_KEY'),
                                      client_kwargs={'region_name': os.environ.get('AWS_REGION_NAME')})

        with open(os.environ.get('CONFIG_PATH', 'r')) as config_file:
            self.config = yaml.safe_load(config_file)

        self.project_name = self.config.get('config').get('model').get('logging').get(
                                                   'comet-project-name')
        self.workspace=self.config.get('config').get('model').get('logging').get(
                                                   'comet-workspace')

        self.bucket_name = os.environ.get('AWS_BUCKET_NAME')
        self.device = device("cuda" if cuda.is_available() else "cpu")
        self.date = datetime.today().strftime('%Y-%m-%d')
        self.datetime = datetime.today().strftime('%Y-%m-%d-%H-%m')

        # Data
        if self.device == "cuda":
            self.num_workers = 4
        else:
            self.num_workers = 0

        self.dataset_name = self.config.get('config').get('general').get('dataset-name')
        self.training_images_path = self.config.get('config').get('input-paths').get('train-images').format(
                                                                                    bucket_name=self.bucket_name)#,
                                                                                    #dataset_name=self.dataset_name)
        self.training_labels_path = self.config.get('config').get('input-paths').get('train-labels').format(
                                                                                    bucket_name=self.bucket_name)#,
                                                                                    # dataset_name=self.dataset_name)
        self.validation_images_path = self.config.get('config').get('input-paths').get('validation-images').format(
                                                                                    bucket_name=self.bucket_name)#,
                                                                                    # dataset_name=self.dataset_name)
        self.validation_labels_path = self.config.get('config').get('input-paths').get('validation-labels').format(
                                                                                    bucket_name=self.bucket_name)#,
                                                                                    # dataset_name=self.dataset_name)

        # Model parameters
        self.model_name = self.config.get('config').get('general').get('model-name')
        self.weight_decay = self.config.get('config').get('model').get('parameters').get('weight-decay')
        self.hidden_size = self.config.get('config').get('model').get('parameters').get('hidden-size')
        self.learning_rate = self.config.get('config').get('model').get('parameters').get('learning-rate')
        self.dropout = self.config.get('config').get('model').get('parameters').get('dropout')
        self.number_of_epochs = self.config.get('config').get('model').get('parameters').get('number-of-epochs')
        self.patience = self.config.get('config').get('model').get('parameters').get('patience')
        self.batch_size = self.config.get('config').get('model').get('parameters').get('batch-size')

        # Model logging
        self.comet_logging = self.config.get('config').get('model').get('logging').get('comet')
        self.mlflow_logging = self.config.get('config').get('model').get('logging').get('ml-flow')

        # Paths
        self.model_path = self.config.get('config').get('output-paths').get('model-path').format(
                                                                                        bucket_name=self.bucket_name,
                                                                                        dataset_name=self.dataset_name,
                                                                                        date=self.date,
                                                                                        model_name=self.model_name,
                                                                                        datetime=self.date)
        self.plot_path = self.config.get('config').get('output-paths').get('loss-plot-path').format(
                                                                                        bucket_name=self.bucket_name,
                                                                                        dataset_name=self.dataset_name,
                                                                                        date=self.date,
                                                                                        model_name=self.model_name,
                                                                                        datetime=self.date)
