import pickle
from framework.environment.definitions import CONFIG_PATH

def load_configuration():
    configuration = pickle.load(open(CONFIG_PATH, "rb"))

    return configuration