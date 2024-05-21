from __future__ import print_function
import keras
import keras.backend as K
import numpy as np

from six.moves import cPickle
import os
import tensorflow as tf
import utils

class LoggingReporter(keras.callbacks.Callback):
    def __init__(self, cfg, trn, tst, do_save_func=None, *kargs, **kwargs):
        super(LoggingReporter, self).__init__(*kargs, **kwargs)
        self.cfg = cfg # Configuration options dictionary
        self.trn = trn  # Train data
        self.tst = tst  # Test data
        
        if 'FULL_MI' not in cfg:
            self.cfg['FULL_MI'] = False # Whether to compute MI on train and test data, or just test
            
        if self.cfg['FULL_MI']:
            self.full = utils.construct_full_dataset(trn, tst)
        
        # do_save_func(epoch) should return True if we should save on that epoch
        self.do_save_func = do_save_func
        
    def on_train_begin(self, logs=None):
        if not os.path.exists(self.cfg['SAVE_DIR']):
            print("Making directory", self.cfg['SAVE_DIR'])
            os.makedirs(self.cfg['SAVE_DIR'])
            
        # Indexes of the layers which we keep track of. Basically, this will be any layer 
        # which has a 'kernel' attribute, which is essentially the "Dense" or "Dense"-like layers
        self.layerixs = []
    
        # Functions return activity of each layer
        self.layerfuncs = []
        
        # Functions return weights of each layer
        self.layerweights = []
        for lndx, l in enumerate(self.model.layers):
            if hasattr(l, 'kernel'):
                self.layerixs.append(lndx)
                self.layerfuncs.append(K.function(self.model.inputs, [l.output, ]))
                self.layerweights.append(l.kernel)

    def on_epoch_begin(self, epoch, logs=None):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            self._log_gradients = False
        else:
            # We will log this epoch. For each batch in this epoch, we will save the gradients (in on_batch_end)
            # We will then compute means and vars of these gradients
            
            self._log_gradients = True
            self._batch_weightnorm = []
                
            self._batch_gradients = [[] for _ in self.model.layers[1:]]
            
            # Indexes of all the training data samples. These are shuffled and read-in in chunks of SGD_BATCHSIZE
            ixs = list(range(len(self.trn.X)))
            np.random.shuffle(ixs)
            self._batch_todo_ixs = ixs

    def on_train_batch_end(self, batch, logs=None):
        if not self._log_gradients:
            # We are not keeping track of batch gradients, so do nothing
            return
        
        # Sample a batch
        batchsize = self.cfg['SGD_BATCHSIZE']
        cur_ixs = self._batch_todo_ixs[:batchsize]
        # Advance the indexing, so next on_train_batch_end samples a different batch
        self._batch_todo_ixs = self._batch_todo_ixs[batchsize:]
        
        # Get gradients for this batch
        inputs = tf.convert_to_tensor(self.trn.X[cur_ixs, :])
        targets = tf.convert_to_tensor(self.trn.Y[cur_ixs, :])

        with tf.GradientTape() as tape:
            predictions = self.model(inputs, training=True)
            loss = self.model.compiled_loss(targets, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        for lndx, grad in enumerate(gradients):
            if lndx < len(self.layerixs):
                oneDgrad = np.reshape(grad.numpy(), [-1, 1])  # Flatten to one dimensional vector
                self._batch_gradients[lndx].append(oneDgrad)

    def on_epoch_end(self, epoch, logs=None):
        if self.do_save_func is not None and not self.do_save_func(epoch):
            # Don't log this epoch
            return
        
        # Get overall performance
        loss = {}
        for cdata, cdataname, istrain in ((self.trn, 'trn', 1), (self.tst, 'tst', 0)):       
            inputs = tf.convert_to_tensor(cdata.X)
            targets = tf.convert_to_tensor(cdata.Y)
            predictions = self.model(inputs, training=False)
            loss[cdataname] = self.model.compiled_loss(targets, predictions).numpy()
            
        data = {
            'weights_norm': [],   # L2 norm of weights
            'gradmean': [],       # Mean of gradients
            'gradstd': [],        # Std of gradients
            'activity_tst': []    # Activity in each layer for test set
        }
        
        for lndx, layerix in enumerate(self.layerixs):
            clayer = self.model.layers[layerix]
            
            data['weights_norm'].append(np.linalg.norm(K.get_value(clayer.kernel)))
            
            stackedgrads = np.stack(self._batch_gradients[lndx], axis=1)
            data['gradmean'].append(np.linalg.norm(stackedgrads.mean(axis=1)))
            data['gradstd'].append(np.linalg.norm(stackedgrads.std(axis=1)))
            
            if self.cfg['FULL_MI']:
                data['activity_tst'].append(self.layerfuncs[lndx]([self.full.X, ])[0])
            else:
                data['activity_tst'].append(self.layerfuncs[lndx]([self.tst.X, ])[0])
            
        fname = self.cfg['SAVE_DIR'] + "/epoch%08d" % epoch
        print("Saving", fname)
        with open(fname, 'wb') as f:
            cPickle.dump({'ACTIVATION': self.cfg['ACTIVATION'], 'epoch': epoch, 'data': data, 'loss': loss}, f, cPickle.HIGHEST_PROTOCOL)

# Configuration dictionary
cfg = {
    'SGD_BATCHSIZE': 128,
    'SGD_LEARNINGRATE': 0.001,
    'NUM_EPOCHS': 100,
    'ACTIVATION': 'tanh',
    'LAYER_DIMS': [1024, 20, 20, 20],
}

# Get MNIST data
trn, tst = utils.get_mnist()

# Where to save activation and weights data
cfg['SAVE_DIR'] = 'rawdata/' + cfg['ACTIVATION'] + '_' + '-'.join(map(str, cfg['LAYER_DIMS']))

# Define model
input_layer = tf.keras.layers.Input((trn.X.shape[1],))
clayer = input_layer
for n in cfg['LAYER_DIMS']:
    clayer = tf.keras.layers.Dense(n, activation=cfg['ACTIVATION'])(clayer)
output_layer = tf.keras.layers.Dense(trn.nb_classes, activation='softmax')(clayer)

model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
optimizer = tf.keras.optimizers.SGD(learning_rate=cfg['SGD_LEARNINGRATE'])

model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Define reporting function
def do_report(epoch):
    if epoch < 20:  # Log for all first 20 epochs
        return True
    elif epoch < 100:  # Then for every 5th epoch
        return epoch % 5 == 0
    elif epoch < 200:  # Then every 10th
        return epoch % 10 == 0
    else:  # Then every 100th
        return epoch % 100 == 0

# Create and set LoggingReporter callback
reporter = LoggingReporter(cfg=cfg, trn=trn, tst=tst, do_save_func=do_report)
r = model.fit(x=trn.X, y=trn.Y,
              verbose=2,
              batch_size=cfg['SGD_BATCHSIZE'],
              epochs=cfg['NUM_EPOCHS'],
              callbacks=[reporter])
