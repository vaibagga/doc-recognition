from enum import Enum

SEG_MODELS = ["UNET"]
NUM_EPOCHS = 30

seg_model_configs = {
    "learning_rate": [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1, 1, 10],
    "batch_size": [4, 8, 16, 32, 64, 128, 256],
    "optimizers": ["Adam", "SGD", "RMSPROP"]
}

seg_model_configs_test = {
    "learning_rate": [1e-5],
    "batch_size": [64],
    "optimizers": ["Adam"]
}