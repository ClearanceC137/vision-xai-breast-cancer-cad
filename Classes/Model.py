# -*- coding: utf-8 -*-
"""
Created on Mon May 19 22:40:12 2025

@author: clear
"""

class Model:
    def __init__(self):
        self.CNNModel = None                     # Instance of the compiled or uncompiled CNN model
        self.model_path = None                # Path for saving/loading model weights or architecture
        self.model_weights = None

    def load_model(self, filepath): pass
    def summary(self): pass


class ModelEvaluator(Model):
    def __init__(self):
        super().__init__()
        self.test_results = {}                # Stores computed evaluation metrics
        self.y_true = None                    # Ground truth labels
        self.y_pred = None                    # Predicted labels from the model
        self.confusion_matrix = None          # Confusion matrix of true vs. predicted
        self.classification_report = None     # Detailed classification report (e.g., from sklearn)

    def evaluate(self, x_test, y_test): pass
    def compute_accuracy(self): pass
    def compute_precision(self): pass
    def compute_recall(self): pass
    def compute_f1_score(self): pass
    def calculate_all_metrics(self): pass
    def plot_confusion_matrix(self): pass
    def generate_classification_report(self): pass

class ModelPredictor(Model):
    def __init__(self):
        super().__init__()
        self.preprocessed_images = None       # List or array of preprocessed images ready for prediction
        self.predictions = None               # Model predictions (probabilities or labels)

    def predict(self, image): pass
    def predict_batch(self, images): pass

class ModelTrainer(Model):
    def __init__(self):
        super().__init__()
        self.history = None          # Keras training history object
        self.optimizer = None        # Optimization algorithm used
        self.loss = None             # Loss function
        self.metrics = None          # Evaluation metrics list

    def compile_model(self, optimizer, loss, metrics): pass
    def train(self, x_train, y_train, epochs, batch_size, validation_data=None): pass
    def cross_validate(self, data, labels, n_folds): pass
    def save_model(self, filepath): pass
