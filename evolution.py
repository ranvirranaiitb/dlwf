import numpy as np
import keras as k
from dlwf.kerasdlwf.data import load_data, split_dataset, DataGenerator
from keras.models import model_from_json
from keras.optimizers import RMSprop
from dlwf.kerasdlwf import tor_cnn
from configobj import ConfigObj
import time

INPUT_SIZE = 10 + 1 # add 1 for bias
HIDDEN_SIZE = 100
OUTPUT_SIZE = 3
N_GENS = 10000
SEQ_LEN = 3000
POP_SIZE = 100
HALF_MUTATE_RANGE = 0.01
DISCRIMINATOR_PATH = '/home/calvin/projects/web-fingerprinting/dlwf/kerasdlwf/models/2904_181830_cnn'
OVERHEAD_FITNESS_MULTIPLIER = 2.
SAMPLES_PER_GEN = 500

# class MLP:
#     def __init__(self):
#         w0 = np.random.rand(INPUT_SIZE, HIDDEN_SIZE)
#         w1 = np.random.rand(HIDDEN_SIZE, OUTPUT_SIZE)
#
#     def forward(self, x):
#         hidden_out = 1. / (1. + np.exp(-np.dot(x, w0)))
#         out = np.dot(hidden_out, w1)

def create_model():
    model = k.models.Sequential()
    model.add(k.layers.LSTM(10, batch_input_shape=(SAMPLES_PER_GEN, SEQ_LEN, 1), return_sequences=True))
    model.add(k.layers.Dense(3))
    return model


def mutate(model):
    new_model = create_model()
    weights_list = model.get_weights()
    for weights in weights_list:
        for w in np.nditer(weights, op_flags=['readwrite']):
            w[...] = np.random.uniform(w - HALF_MUTATE_RANGE, w + HALF_MUTATE_RANGE)
    new_model.set_weights(weights_list)
    return new_model


def insert_data(generator_output, original_data):
    # return original_data, 0
    new_data = np.empty((SAMPLES_PER_GEN, SEQ_LEN, 1))
    generator_output = np.argmax(generator_output, axis=2)
    insertions = 0
    for i in range(SAMPLES_PER_GEN):
        x = 0
        for j in range(SEQ_LEN):
            if x == SEQ_LEN:
                break
            new_data[i][x] = original_data[i][j]
            x += 1
            if x == SEQ_LEN:
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


def train(datapath):
    torconf = "dlwf/kerasdlwf/tor.conf"
    config = ConfigObj(torconf)
    traces = config.as_int('traces')
    dnn = config['dnn']
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

    print('Loading data {}... '.format(datapath))
    data, labels = load_data(datapath,
                             minlen=minlen,
                             maxlen=maxlen,
                             traces=traces,
                             dnn_type=dnn)

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

    model = tor_cnn.build_model(learn_params, nb_classes)

    metrics = ['accuracy']

    optimizer = RMSprop(lr=learn_params['lr'],
                            decay=learn_params['decay'])

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=metrics)

    print(model.summary())

    # Train model on dataset
    history = model.fit_generator(generator=data_params['train_gen'],
                                  steps_per_epoch=learn_params['train_steps'],
                                  validation_data=data_params['val_gen'],
                                  validation_steps=learn_params['val_steps'],
                                  epochs=learn_params['epochs'])

    tr_loss = round(history.history['loss'][-1], 4)
    tr_acc = round(history.history['acc'][-1], 4)

    # print('~~~~~~~~~~~')
    # # print(model.get_weights())
    # data0, labels0 = load_data('/home/calvin/projects/web-fingerprinting/data/test.npz', minlen=0, maxlen=3000, traces=1500, dnn_type='cnn')
    # result = model.evaluate(data0, labels0, batch_size=256)
    # print(result)
    # print('~~~~~~~~~~~')
    return model


def run(discriminator):
    global N_SAMPLES
    f = open('log.txt', 'w')

    # load discriminator
    # json_file = open(DISCRIMINATOR_PATH + '.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # discriminator = k.models.model_from_json(loaded_model_json)
    # discriminator.load_weights(DISCRIMINATOR_PATH + ".h5")
    # discriminator.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=['accuracy'])

    # load data
    all_data, all_labels = load_data('data/test.npz', maxlen=SEQ_LEN, traces=1500, dnn_type='cnn')
    N_SAMPLES = all_data.shape[0]
    base_accuracy = discriminator.evaluate(all_data, all_labels, batch_size=256)[1]
    print('BASE ACC: {}'.format(base_accuracy))

    # create initial population
    population = [create_model()]

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
            acc = discriminator.evaluate(new_data, labels, batch_size=SAMPLES_PER_GEN)[1]
            overhead_fitness = 1. - (float(insertions) / float(SAMPLES_PER_GEN * SEQ_LEN))
            acc_fitness = 1. - acc
            total_fitness = OVERHEAD_FITNESS_MULTIPLIER * overhead_fitness + acc_fitness
            print('total fitness: {:.4f}, overhead fitness: {:.4f}, acc fitness: {:.4f}'.format(total_fitness, overhead_fitness, acc_fitness))
            f.write('total fitness: {:.4f}, overhead fitness: {:.4f}, acc fitness: {:.4f}\n'.format(total_fitness, overhead_fitness, acc_fitness))
            fitness.append(total_fitness)
        best = fitness.index(max(fitness))

        # mutate
        new_pop = []
        for _ in range(POP_SIZE):
            new_pop.append(mutate(population[best]))
        population = new_pop

        end = time.time()
        print('GEN {}: best fitness = {:.4f} (took {:.2f} sec)'.format(gen, fitness[best], end-start))
        f.write('GEN {}: best fitness = {:.4f} (took {:.2f} sec)\n'.format(gen, fitness[best], end-start))

    f.close()

if __name__ == '__main__':
    discriminator = train('/home/calvin/projects/web-fingerprinting/data/train.npz')
    run(discriminator)
