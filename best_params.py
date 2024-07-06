"""
@File    : best_params.py
@Date    : 2024-06-13
@Author  : LiuTianSheng
@Software : yolo-learn
"""
import optuna
from ultralytics import YOLO


def objective(trial):
    # Define the hyperparameters to be optimized
    batch_size = trial.suggest_int('batch_size', 8, 64)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True)
    momentum = trial.suggest_float('momentum', 0.85, 0.99)

    # Load model and set hyperparameters
    model = YOLO('yolov8n.pt')
    model.batch_size = batch_size
    model.learning_rate = learning_rate
    model.momentum = momentum

    # Train the model
    results = model.train(data='coco.yaml', epochs=10)

    # Return the metric to be optimized (e.g., validation loss)
    return results['metrics']['val_loss']


# Create study and optimize
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

# Print best hyperparameters
print('Best hyperparameters: ', study.best_params)
