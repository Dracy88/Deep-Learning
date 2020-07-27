import pandas
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.cross_validation import cross_val_score, StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime

timerStart = datetime.datetime.now()

# Fissiamo un random seed per la riproducibilità del esperimento
seed = 1

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
scalar = MinMaxScaler()
scalar.fit(X_test)
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
def create_baseline():
    ''' Creazione di un modello Dense(32) -> Dense(8) -> Dense(1) '''
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=X_train.shape[1], kernel_initializer=kernel_initializer[0], activation=activation[0]))
    for layer in range(hidden_layers_number):
        model.add(Dense(neurons[layer+1], kernel_initializer=kernel_initializer[layer+1], activation=activation[layer+1]))
        model.add(Dropout(drop[layer+1]))

    # Compilazione del modello
    adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss_funtion, optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model

# Valutazione del modello
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=1)))
pipeline = Pipeline(estimators)

kfold = StratifiedKFold(y=Y_train, n_folds=10, shuffle=True, random_state=seed)

results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)

print("Accuratezza:", round(results.mean(), 6) * 100, '%')
print("Standard Dev:", round(results.std(), 6) * 100, '%')

timerStop = datetime.datetime.now()
print("Process ended in", (timerStop - timerStart).total_seconds())

