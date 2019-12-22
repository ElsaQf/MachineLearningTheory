### tensorflow 2.0 练习
1. tensorflow.keras搭建基本神经网络模型
'''
model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
             loss='categorical_crossentropy',
             metrics=\['acc'])
history = model.fit(train_image, 
                    train_label_onehot, 
                    epochs=10, 
                    validation_data=(test_image, test_label_onehot))
'''
2. CNN模型搭建
'''
model = keras.Sequential()
model.add(keras.layers.Conv2D(128,
                            kernel_size=3,
                            activation='relu',
                            input_shape=(64,64,1)))
model.add(keras.layers.Conv2D(64,
                            kernel_size=3,
                            activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(40,
                            activation='softmax'))
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=\['acc'])
'''
