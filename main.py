from get_data import *
from DenseNet import *

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras


dense_block_size = 3
layers_in_block = 4

growth_rate = 12
classes = 10
model = dense_net(growth_rate * 2, growth_rate, classes, dense_block_size, layers_in_block)
model.summary()


# training
batch_size = 256
epochs = 10
optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit(images,labels, epochs=epochs, batch_size=batch_size, shuffle=True)


# testing
label_pred = model.predict(images_test)

pred = []
for i in range(len(label_pred)):
    pred.append(np.argmax(label_pred[i]))

count = 0
for i in range(len(pred)):
    if pred[i] == labels_test[i]:
        count += 1

print('Accuracy in test set is: ', count / 100)