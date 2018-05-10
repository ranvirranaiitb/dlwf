import numpy as np
import keras as k
from dlwf.kerasdlwf.data import split_dataset, DataGenerator
from keras.models import model_from_json
from keras.optimizers import RMSprop
from dlwf.kerasdlwf import tor_cnn, tor_lstm
from configobj import ConfigObj
import time
import sys
import random

N_GENS = 100
# SEQ_LEN = 3000
POP_SIZE = 100
HALF_MUTATE_RANGE = 0.1
MIN_HALF_MUTATE_RANGE = 0.01
SHRINK_MUT_RANGE_OVER_N_GENS = 9
MUT_STEP = (HALF_MUTATE_RANGE - MIN_HALF_MUTATE_RANGE) / float(SHRINK_MUT_RANGE_OVER_N_GENS)
OVERHEAD_FITNESS_MULTIPLIER = 2.
SAMPLES_PER_GEN = 500
LOAD_DISCRIMINATOR = True
INDEX = '0'
LSTM_UNITS = 10
LSTM_DISCRIM = True
if LSTM_DISCRIM:
    SEQ_LEN = 150
else:
    SEQ_LEN = 3000

def load_data(filepath):
    with np.load(filepath) as f:
        data = f['data']
        labels = f['labels']
    return data, labels

def create_model():
    model = k.models.Sequential()
    model.add(k.layers.LSTM(LSTM_UNITS, batch_input_shape=(SAMPLES_PER_GEN, SEQ_LEN, 1), return_sequences=True))
    model.add(k.layers.Dense(3))
    return model

def load_generator(path):
    model = create_model()
    model.load_weights(path)
    return model

def mutate(model):
    new_model = create_model()
    weights_list = model.get_weights()
    for weights in weights_list:
        for w in np.nditer(weights, op_flags=['readwrite']):
            w[...] = np.random.uniform(w - HALF_MUTATE_RANGE, w + HALF_MUTATE_RANGE)
    new_model.set_weights(weights_list)
    return new_model


def mutate_fast(population, best):
    best_weights = population[best].get_weights()
    for i in range(POP_SIZE):
        new_weights = []
        for weights in best_weights:
            mut = np.random.uniform(-HALF_MUTATE_RANGE, HALF_MUTATE_RANGE, weights.shape)
            new_weights.append(weights + mut)
        population[i].set_weights(new_weights)
    return population


def insert_data(generator_output, original_data):
    new_data = np.empty(original_data.shape)
    generator_output = np.argmax(generator_output, axis=2)
    insertions = 0
    for i in range(original_data.shape[0]):
        x = 0
        for j in range(original_data.shape[1]):
            if x == original_data.shape[1]:
                break
            new_data[i][x] = original_data[i][j]
            x += 1
            if x == original_data.shape[1]:
                break
            if generator_output[i][j] == 1:
                # insert 1
                new_data[i][x] = [1.]
                x += 1
                insertions += 1
            elif generator_output[i][j] == 2:
                # insert -1
                new_data[i][x] = [-1.]
                x += 1
                insertions += 1
    return new_data, insertions


def insert_data_randomly(original_data, percent):
    new_data = np.empty(original_data.shape)
    insertions = 0
    for i in range(original_data.shape[0]):
        x = 0
        for j in range(original_data.shape[1]):
            if x == original_data.shape[1]:
                break
            new_data[i][x] = original_data[i][j]
            x += 1
            if x == original_data.shape[1]:
                break
            if random.random() < percent:
                if random.random() < 0.5:
                    # insert 1
                    new_data[i][x] = [1.]
                    x += 1
                    insertions += 1
                else:
                    # insert -1
                    new_data[i][x] = [-1.]
                    x += 1
                    insertions += 1
    return new_data, insertions


def eval_random_insertion(discriminator):
    if LSTM_DISCRIM:
        f = open('random_insertion_results_lstm.txt', 'w')
        test_data, test_labels = load_data('data/test_onehot_lstm.npz')
    else:
        test_data, test_labels = load_data('data/test_onehot.npz')
        f = open('random_insertion_results.txt', 'w')
    N_TEST_SAMPLES = test_data.shape[0]
    base_accuracy = eval(discriminator, test_data, test_labels, 256)
    # base_accuracy = discriminator.evaluate(test_data, test_labels, batch_size=256)[1]
    print('BASE TEST ACC: {}'.format(base_accuracy))

    for p in range(1, 100):
        percent = float(p) / float(100)
        new_data, insertions = insert_data_randomly(test_data, percent)
        acc = eval(discriminator, new_data, test_labels, batch_size=SAMPLES_PER_GEN)
        overhead = float(insertions) / float(N_TEST_SAMPLES * SEQ_LEN)
        output = 'RANDOM INSERTION CHANCE {}: test overhead = {:.4f}, test acc = {:.4f}'.format(percent, overhead, acc)
        print(output)
        f.write(output + '\n')
    f.close()


def train(datapath):
    torconf = "dlwf/kerasdlwf/tor.conf"
    config = ConfigObj(torconf)
    traces = config.as_int('traces')
    # dnn = config['dnn']
    if LSTM_DISCRIM:
        dnn = 'lstm'
    else:
        dnn = 'cnn'
    seed = config.as_int('seed')
    minlen = config.as_int('minlen')
    nb_epochs = config[dnn].as_int('nb_epochs')
    batch_size = config[dnn].as_int('batch_size')
    val_split = config[dnn].as_float('val_split')
    test_split = config[dnn].as_float('test_split')
    optimizer = config[dnn]['optimizer']
    nb_layers = config[dnn].as_int('nb_layers')
    layers = [config[dnn][str(x)] for x in range(1, nb_layers + 1)]
    lr = config[dnn][optimizer].as_float('lr')
    decay = config[dnn][optimizer].as_float('decay')
    maxlen = config[dnn].as_int('maxlen')

    nb_features = 1

    print('Loading data... ')
    if LSTM_DISCRIM:
        data, labels = load_data('data/train_onehot_lstm.npz')
    else:
        data, labels = load_data('data/train_onehot.npz')

    nb_instances = data.shape[0]
    nb_cells = data.shape[1]
    nb_classes = labels.shape[1]
    nb_traces = int(nb_instances / nb_classes)

    print('Loaded data {} instances for {} classes: {} traces per class, {} Tor cells per trace'.format(nb_instances,
                                                                 nb_classes,
                                                                 nb_traces,
                                                                 nb_cells))

    # CROSS-VALIDATION
    indices = np.arange(nb_instances)
    np.random.shuffle(indices)
    num = nb_instances

    split = int(num * (1 - test_split))
    ind_test = np.array(indices[split:])

    num = indices.shape[0] - ind_test.shape[0]
    split = int(num * (1 - val_split))

    ind_val = np.array(indices[split:num])
    ind_train = np.array(indices[:split])

    # Generators
    train_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_train)
    val_gen = DataGenerator(batch_size=batch_size).generate(data, labels, ind_val)
    test_gen = DataGenerator(batch_size=1).generate(data, labels, ind_test)

    data_params = {'train_gen': train_gen,
                   'val_gen': val_gen,
                   # 'test_data': (x_test, y_test),
                   'nb_instances': nb_instances,
                   'nb_classes': nb_classes,
                   'nb_traces': nb_traces}

    learn_params = {'dnn_type': dnn,
                    'epochs': nb_epochs,
                    'train_steps': ind_train.shape[0] // batch_size,
                    'val_steps': ind_val.shape[0] // batch_size,
                    'nb_features': nb_features,
                    'batch_size': batch_size,
                    'optimizer': optimizer,
                    'nb_layers': nb_layers,
                    'layers': layers,
                    'lr': lr,
                    'decay': decay,
                    'maxlen': maxlen}

    nb_instances = data_params["nb_instances"]
    nb_classes = data_params["nb_classes"]
    nb_traces = data_params["nb_traces"]

    print('Building model...')

    if LSTM_DISCRIM:
        model = tor_lstm.build_model(learn_params, nb_classes)
    else:
        model = tor_cnn.build_model(learn_params, nb_classes)


    print(model.summary())

    # Train model on dataset
    if LOAD_DISCRIMINATOR:
        if LSTM_DISCRIM:
            model.load_weights('lstm_discrim_weights.h5')
        else:
            model.load_weights('best_discriminator_weights.h5')
    else:
        metrics = ['accuracy']
        optimizer = RMSprop(lr=learn_params['lr'],
        decay=learn_params['decay'])
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)
        history = model.fit_generator(generator=data_params['train_gen'],
                                      steps_per_epoch=learn_params['train_steps'],
                                      validation_data=data_params['val_gen'],
                                      validation_steps=learn_params['val_steps'],
                                      epochs=learn_params['epochs'])
        if INDEX == '0':
            if LSTM_DISCRIM:
                model.save_weights('lstm_discrim_weights.h5')
            else:
                model.save_weights('best_discriminator_weights.h5')

    return model


def evaluate(generator, discriminator):
    if LSTM_DISCRIM:
        test_data, test_labels = load_data('data/test_onehot_lstm.npz')
    else:
        test_data, test_labels = load_data('data/test_onehot.npz')
    N_TEST_SAMPLES = test_data.shape[0]
    base_accuracy = eval(discriminator, test_data, test_labels, 256)
    # base_accuracy = discriminator.evaluate(test_data, test_labels, batch_size=256)[1]
    print('BASE TEST ACC: {}'.format(base_accuracy))

    results = generator.predict(test_data, batch_size=SAMPLES_PER_GEN)
    new_data, insertions = insert_data(results, test_data)
    acc = eval(discriminator, new_data, test_labels, batch_size=SAMPLES_PER_GEN)
    overhead = float(insertions) / float(N_TEST_SAMPLES * SEQ_LEN)
    print('generator test overhead = {:.4f}, test acc = {:.4f}'.format(overhead, acc))


def eval(model, data, labels, batch_size):
    results = model.predict(data, batch_size=batch_size)
    idxs1 = np.argmax(results, axis=1)
    idxs2 = np.argmax(labels, axis=1)
    return np.sum(idxs1 == idxs2) / float(labels.shape[0])


def run(discriminator):
    global N_SAMPLES, HALF_MUTATE_RANGE
    f = open('log{}.txt'.format(INDEX), 'w')

    # load data
    if LSTM_DISCRIM:
        all_data, all_labels = load_data('data/val_onehot_lstm.npz')
    else:
        all_data, all_labels = load_data('data/val_onehot.npz')
    N_SAMPLES = all_data.shape[0]
    base_accuracy = eval(discriminator, all_data, all_labels, 256)
    print('BASE VAL ACC: {}'.format(base_accuracy))

    if LSTM_DISCRIM:
        test_data, test_labels = load_data('data/test_onehot_lstm.npz')
    else:
        test_data, test_labels = load_data('data/test_onehot.npz')
    N_TEST_SAMPLES = test_data.shape[0]
    base_accuracy = eval(discriminator, test_data, test_labels, 256)
    print('BASE TEST ACC: {}'.format(base_accuracy))


    # create initial population
    population = []
    for _ in range(POP_SIZE):
        population.append(create_model())

    # generational loop
    for gen in range(N_GENS):
        start = time.time()

        # select subset of data
        idxs = np.random.permutation(N_SAMPLES)
        idxs = idxs[:SAMPLES_PER_GEN]
        data = all_data[idxs, :]
        labels = all_labels[idxs]

        # determine fitness of population
        fitness = []
        for model in population:
            results = model.predict(data, batch_size=SAMPLES_PER_GEN)
            new_data, insertions = insert_data(results, data)
            acc = eval(discriminator, new_data, labels, batch_size=SAMPLES_PER_GEN)
            overhead_fitness = 1. - (float(insertions) / float(SAMPLES_PER_GEN * SEQ_LEN))
            acc_fitness = 1. - acc
            total_fitness = OVERHEAD_FITNESS_MULTIPLIER * overhead_fitness + acc_fitness
            output = 'total fitness: {:.4f}, overhead fitness: {:.4f}, acc fitness: {:.4f}'.format(total_fitness, overhead_fitness, acc_fitness)
            print(output)
            f.write(output + '\n')
            fitness.append(total_fitness)
        best = fitness.index(max(fitness))

        # eval best on test data
        results = population[best].predict(test_data, batch_size=SAMPLES_PER_GEN)
        new_data, insertions = insert_data(results, test_data)
        acc = eval(discriminator, new_data, test_labels, batch_size=SAMPLES_PER_GEN)
        overhead = float(insertions) / float(N_TEST_SAMPLES * SEQ_LEN)
        output = 'BEST OF GEN {}: test overhead = {:.4f}, test acc = {:.4f}'.format(gen, overhead, acc)
        print(output)
        f.write(output + '\n')

        # save best
        if LSTM_DISCRIM:
            population[best].save_weights('best_generator_weights{}_lstm.h5'.format(INDEX))
        else:
            population[best].save_weights('best_generator_weights{}.h5'.format(INDEX))

        # mutate
        # new_pop = []
        # for _ in range(POP_SIZE):
        #     new_pop.append(mutate_fast(population[best]))
        # population = new_pop
        new_pop = mutate_fast(population, best)

        # shrink mutation range
        HALF_MUTATE_RANGE = max(MIN_HALF_MUTATE_RANGE, HALF_MUTATE_RANGE - MUT_STEP)

        end = time.time()
        output = 'GEN {}: best fitness = {:.4f} (took {:.2f} sec)'.format(gen, fitness[best], end-start)
        print(output)
        f.write(output + '\n')

    f.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        INDEX = sys.argv[1]
        print('index: {}'.format(INDEX))
    idx = int(INDEX)
    # if idx == 0:
    #     MIN_HALF_MUTATE_RANGE = 0.005
    # elif idx == 1 or idx == 2:
    #     LSTM_UNITS = 40
    # elif idx == 3 or idx == 4:
    #     LSTM_UNITS = 80
    # elif idx == 5:
    #     LSTM_UNITS = 40
    #     MIN_HALF_MUTATE_RANGE = 0.005
    # elif idx == 6:
    #     LSTM_UNITS = 80
    #     MIN_HALF_MUTATE_RANGE = 0.005
    # elif idx == 7:
    #     LSTM_UNITS = 160
    # elif idx == 8:
    #     LSTM_UNITS = 160
    #     MIN_HALF_MUTATE_RANGE = 0.005
    discriminator = train('data/train.npz')
    # run(discriminator)
    eval_random_insertion(discriminator)
    # generator = load_generator('best_generator_weights0.h5')
    # evaluate(generator, discriminator)
