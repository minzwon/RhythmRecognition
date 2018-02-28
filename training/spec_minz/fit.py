# -*- coding: utf-8 -*-
import os
import fire
import cPickle

from keras.callbacks import ModelCheckpoint, TensorBoard
import warnings

from data import Data
from machine import Network
import numpy as np

class Learner:
    def fit_gen(self,
            tr_filelist,
            val_filelist=None,
            path='/data1/ExBallroom/spec_trp/',
            meta_path='../../metas_9g.cPickle',
            epochs=20,
            batch_size=100,
            validation_split=0.1,
            weight_checkpoint='checkpoints/spec_minz.{epoch:02d}.hdf5',
            out_model_fname='spec_minz.keras',
            gpu_id = 0,
            is_class_weight=False):
        # set gpu id
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        n_classes = 9
        n = Network(num_tags=n_classes)
        self.model = n.model
        self.data = Data()

        metas = cPickle.load(open(meta_path))

        self.classes = ['Chacha', 'Foxtrot', 'Jive','Quickstep', 'Rumba',
                        'Samba', 'Tango', 'Viennesewaltz', 'Waltz']
        valid_gnr = dict.fromkeys(self.classes)
        for i, c in enumerate(self.classes):
            valid_gnr[c] = i
        print(valid_gnr)

        def cluster_getter(metas, song_id):
            m = metas.get('%06d' % int(song_id))
            mm = np.zeros(9)
            mm[self.classes.index(m)] = 1
            return mm
        class_getter = lambda x, y: cluster_getter(x, y)

        generator_tr = self.data.generator(tr_filelist,
                                           batch_size,
                                           path,
                                           metas,
                                           class_getter,
                                           extension='npy',
                                           data_loader=np.load)
        if val_filelist:
            generator_val = self.data.generator(val_filelist,
                                                batch_size,
                                                path,
                                                metas,
                                                class_getter,
                                                extension='npy',
                                                data_loader=np.load)
        checkpointer = MinzCheckpoint(filepath=weight_checkpoint, verbose=1, save_best_only=False)
        tensor_board = TensorBoard(log_dir='tensorboard/spec_minz', histogram_freq = 0, write_graph=True, write_images=True,batch_size=batch_size)
        callbacks = [checkpointer, tensor_board]
        print('')
        #print('Train: %s %s' % (X.shape, Y.shape))
        #    print('Validation: %s %s' % (val_X.shape, val_Y.shape))
        self.model.fit_generator(generator_tr,
                                 steps_per_epoch = int(51871/batch_size),
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=callbacks,
                                 validation_data=generator_val,
                                 validation_steps=int(5639/batch_size),
                                 initial_epoch=0)


        self.model.save(out_model_fname)


class MinzCheckpoint(ModelCheckpoint):
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=self.epoch)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f, '
                                ' saving model to %s'
                                % (self.epoch, self.monitor, self.best,
                                    current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                (self.epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (self.epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


    def on_batch_end(self, batch, logs=None):
        pass



if __name__ == '__main__':
    l = Learner()
    fire.Fire({'fit_gen': l.fit_gen})
