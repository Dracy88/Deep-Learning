from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas
from keras import optimizers

# Creazione dataframe di training
training = pandas.read_csv("training_data.csv", header=None)
training = training.values
X_train = training[1:, 1:22]
Y_train = training[1:, 23]

# Creazione dataframe di testing
testing = pandas.read_csv("testing_data.csv", header=None)
testing = testing.values
X_test = testing[1:, 1:22]

# Standardizzazione dei dataframe di input
scalar = MinMaxScaler()
scalar.fit(X_train)
X_train = scalar.transform(X_train)
X_test = scalar.transform(X_test)

# Iper parametri
hidden_layers_number = 2
neurons = [32, 8, 1]  # Number of neurons for layers
kernel_initializer = ['uniform', 'uniform', 'uniform']  # Weight initializer technique
loss_funtion = 'binary_crossentropy'
activation = ['tanh', 'tanh', 'sigmoid']
epochs = 100
batch_size = 1024
lr = 0.001  # Learning Rate
drop = [0, 0.1, 0]

# Creazione del modello
model = Sequential()
model.add(Dense(neurons[0], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer[0], activation=activation[0]))
for layer in range(hidden_layers_number):
    model.add(Dense(neurons[layer+1], kernel_initializer=kernel_initializer[layer+1], activation=activation[layer+1]))
    model.add(Dropout(drop[layer+1]))

# Compilazione del modello
adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss=loss_funtion, optimizer=adam, metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=epochs, verbose=1)

# Predizioni
ynew = model.predict_classes(X_test)

with open('result.txt', 'w') as f:
    for i in range(len(X_test)):
        f.write(str(ynew[i][0]))
        f.write('\n')
