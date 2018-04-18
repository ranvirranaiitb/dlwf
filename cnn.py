import numpy as np
import tensorflow as tf
import shutil
import os

LEARNING_RATE = 0.0011
DECAY = 0.0
BATCH_SIZE = 256
EPOCHS = 4
LAYERS = 8
INPUT_UNITS = 3000
DROPOUT = 0.1
KERNELS = 32
KERNEL_SIZE = 5
POOL_SIZE = 4
N_LABELS = None

def build_model(features, labels, mode):
    n, dims, _ = features['x'].shape
    input_layer = features['x']

    layers = []
    for i in range(LAYERS):
        if i == 0:
            inputs = input_layer
        else:
            inputs = layers[i-1]
        if i % 2 == 0:
            conv = tf.layers.conv1d(
                inputs=inputs,
                filters=KERNELS,
                kernel_size=KERNEL_SIZE,
                activation=tf.nn.relu)
            layers.append(tf.layers.dropout(
                inputs=conv,
                rate=DROPOUT,
                training=mode == tf.estimator.ModeKeys.TRAIN))
        else:
            layers.append(tf.layers.max_pooling1d(
                inputs=inputs,
                pool_size=POOL_SIZE,
                strides=POOL_SIZE))

    _, val1, val2 = layers[-1].get_shape()
    out_flat = tf.reshape(layers[-1], [-1, val1 * val2])
    logits = tf.layers.dense(inputs=out_flat, units=N_LABELS)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.reduce_sum(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits), name='loss')

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=DECAY)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # EVAL mode
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train(x_train, x_test, y_train, y_test):
    config=tf.contrib.learn.RunConfig(save_checkpoints_secs=5)
    model_dir = 'models/cnn0'
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    classifier = tf.estimator.Estimator(model_fn=build_model, model_dir=model_dir, config=config)
    tf.logging.set_verbosity(tf.logging.INFO)

    # eval during training
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_test},
        y=y_test,
        shuffle=False)
    validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(input_fn=test_input_fn, every_n_steps=2000)
    hooks = tf.contrib.learn.monitors.replace_monitors_with_hooks([validation_monitor], classifier)

    # train
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_train},
        y=y_train,
        batch_size=BATCH_SIZE,
        num_epochs=EPOCHS,
        shuffle=False)
    print 'training...'
    classifier.train(
        input_fn=train_input_fn,
        steps=None,
        hooks=hooks)

    # eval
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': x_test},
        y=y_test,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print 'FINAL EVAL'
    print eval_results


def load_data(filename):
    print 'loading data...'
    path = 'data/' + filename
    f = np.load(path)
    x = f['data'][:,:INPUT_UNITS]
    x = x.astype(np.float16, copy=False)
    x = np.expand_dims(x, axis=3) # tf requires this for some reason
    y_labels = f['labels']
    n, dims, _ = x.shape

    # convert labels to ints
    y = np.empty(n, dtype=int)
    label_to_int = {}
    cur = 0
    for i in range(n):
        if y_labels[i] in label_to_int:
            val = label_to_int[y_labels[i]]
        else:
            label_to_int[y_labels[i]] = cur
            val = cur
            cur += 1
        y[i] = val
    global N_LABELS
    N_LABELS = np.amax(y) + 1

    # split dataset
    indices = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[train_idx,:], x[test_idx,:]
    y_train, y_test = y[train_idx], y[test_idx]
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = load_data('tor_100w_2500tr.npz')
    train(x_train, x_test, y_train, y_test)


































#
