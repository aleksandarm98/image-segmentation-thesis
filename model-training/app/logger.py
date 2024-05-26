import logging


class Logger:
    def __init__(self, name, level=logging.INFO):
        # Kreiranje logger-a
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Kreiranje handler-a za logovanje na konzolu
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # Definisanje formata logova
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Dodavanje handler-a logger-u
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# Instanciranje logger-a






# import logging
# import os
#
# class Logger:
#     def __init__(self, name, log_file, level=logging.INFO):
#         # Kreiranje logger-a
#         self.logger = logging.getLogger(name)
#         self.logger.setLevel(level)
#
#         # Kreiranje handler-a za logovanje u fajl
#         file_handler = logging.FileHandler(log_file)
#         file_handler.setLevel(level)
#
#         # Kreiranje handler-a za logovanje na konzolu
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(level)
#
#         # Definisanje formata logova
#         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
#
#         # Dodavanje handler-a logger-u
#         self.logger.addHandler(file_handler)
#         self.logger.addHandler(console_handler)
#
#     def get_logger(self):
#         return self.logger
#
# # Funkcija za kreiranje direktorijuma za logove
# def create_log_dir(log_dir):
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#
# # Kreiranje objekta logger-a i direktorijuma za logove
# log_dir = 'logs'
# create_log_dir(log_dir)
# logger = Logger(__name__, os.path.join(log_dir, 'app.log')).get_logger()
