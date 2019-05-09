# wykorzytsanie generatora liczb pseud.
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy



#  model

i = 2
for i in [2, 10, 20, 50, 100]:

    numpy.random.seed(19940818)
    # ladowanie danych
    dataset = numpy.loadtxt("dane.csv", delimiter=",")
    # split into input (X) and output (Y) variables
    X = dataset[:, 0:8]
    Y = dataset[:, 8]
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # optymalizacja
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    epochs = 300
    history = model.fit(X, Y, epochs=epochs, batch_size=i)
    print(history.history.keys())
    plt.plot(history.history['acc'])
    #plt.plot(history.history['val_acc'])
    plt.title('Wykres dokładności')
    plt.ylabel('dokładność')
    plt.xlabel('epoka')
    plt.legend(['batch_2', 'batch_10', 'batch_20', 'batch_50', 'batch_100'], loc='upper left')
    plt.xticks(numpy.arange(10, epochs, step=10))
    plt.savefig('sgd-%s.png' % i)
    plt.show()
    # ealuacja
    scores = model.evaluate(X, Y)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

