import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
# (x_train, x_test), (y_train, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)

# plt.imshow(x_train[0])
# plt.show()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy'
              )
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

model.save('mnist_model')
new_model = tf.keras.models.load_model('mnist_model')


predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()