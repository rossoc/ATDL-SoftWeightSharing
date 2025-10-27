from src.ATDLSoftWeightSharing.data import get_mnist_data
(x_train, y_train), (x_test, y_test) = get_mnist_data()
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('y_train first 3:', y_train[:3])