
from climateai.processing.process import Process
from climateai.processing.data_loading import DataLoader

from climateai.models.neural_net import NeuralNet

if __name__ == "__main__":
    dl = DataLoader()
    full_df = dl.load()
    
    X_train, X_test, y_train, y_test, X_wins_train, X_wins_test, X_scaled_train, X_scaled_test = Process.split_data(full_df)
    nn = NeuralNet()
    nn.train(X_scaled_train, y_train)
    nn.evaluate(X_scaled_test, y_test)


    nn.find_best_parameters()


    