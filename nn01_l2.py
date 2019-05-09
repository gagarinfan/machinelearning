# wykorzytsanie generatora liczb pseud.
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy



#  model

batch_size = 10
epochs = 1000
neurons = 12 #default 12
print(neurons)
s = 1994-8-18
numpy.random.seed(s)
# ladowanie danych
dataset = numpy.loadtxt("dane.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:, 0:8]
Y = dataset[:, 8]
model = Sequential()

model.add(Dense(int(neurons*4), input_dim=8, activation='relu'))
#71,74 sigmoid, mbinary_error
#71,88 sigmoid mean_squared_error
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# optymalizacja
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, validation_split=0.33)

plt.plot(history.history['acc'])
#plt.plot(history.history['val_acc'])
plt.title('Wykres dokładności')
plt.ylabel('dokładność')
plt.xlabel('epoka')
plt.legend(['train', 'test'], loc='upper left')
#plt.xticks(numpy.arange(10, epochs, step=10))
#plt.savefig('sgd-%s.png' % i)
plt.show()

print(model.predict(X).round())
print(history.history.keys())
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

