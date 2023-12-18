from data.preprocess_pma import  *
from modules.baseline import *

import tensorflow as tf

model = baseline(input_shape=(64, 32, 1))

datasets = load_datasets(path="datasets/experiment-i", type_load="17class", preproc=True)

x_train, y_train, x_test, y_test = preprocess(datasets=datasets, num_cls=17, sub="S1", hist_equal=True, apply_color_map=False, normalize=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=[tf.keras.metrics.CategoricalAccuracy()])
model.fit(x_train, y_train,
          batch_size=256,
          epochs=50,
          verbose=1,
          # callbacks=[model_checkpoint_callback],
          validation_data=(x_test, y_test))
