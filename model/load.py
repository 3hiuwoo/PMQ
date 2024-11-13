from .cnn3 import CNNContrast, CNNSupervised, CNN3

def load_model(model_name, task, embeddim):
    if task == 'contrast':
        if model_name == 'cnn3':
            return CNNContrast(network=CNN3, embeddim=embeddim)
    elif task == 'supervised':
        if model_name == 'cnn3':
            return CNNSupervised(network=CNN3, embeddim=embeddim)