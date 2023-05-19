from sklearn.metrics import mean_squared_error
import numpy as np

def add_noise(data, noise_level=0.0):
    rmse = mean_squared_error(data, np.zeros_like(data), squared=False)
    return data + np.random.normal(0, noise_level * rmse, size=data.shape)