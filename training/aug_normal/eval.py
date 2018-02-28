# -*- coding: utf-8 -*-
import os
import fire
import cPickle
import tqdm
from keras.models import load_model
from machine import Network
import numpy as np
from sklearn.metrics import confusion_matrix

class Eval:
    def __init__(self):
        self.val = np.load('../../val')
        self.metas = cPickle.load(open('../../metas_9g.cPickle'))
        self.classes = ['Chacha', 'Foxtrot', 'Jive', 'Quickstep', 'Rumba',
                        'Samba', 'Tango', 'Viennesewaltz', 'Waltz']
        self._true = []
        self._prd = []

    def predict(self, songid):
        fn = os.path.join(self.path, songid)+'.npy'
        spec = np.load(fn)
        num_chunk = (spec.shape[1]-512)/60 +1
        feature = []
        for i in range(num_chunk):
            feature.append(spec[:, i*60:i*60+512].reshape(96, 512, 1))
        feature = np.array(feature)
        prds = self.model.predict(feature)
        gen = np.argmax(np.mean(prds,axis=0))
        true_gen = self.classes.index(self.metas[songid])
        if gen == true_gen:
            self.corr += 1
        self._prd.append(gen)
        self._true.append(true_gen)


    def get_path(self, path_type):
        if path_type == 0:
            self.path = '/data1/ExBallroom/spec_16k/'
        else:
            self.path = '/data1/ExBallroom/'+str(path_type)+'p_spec/'


    def run(self, gpu_id=0, path_type=0):
        self.get_path(path_type)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        #n = Network()
        #self.model = n.model
        #self.model.load_weights('checkpoints/aug_normal.79.hdf5')
        self.model = load_model('aug_normal.keras')
        self.corr = 0
        for songid in tqdm.tqdm(self.val):
            self.predict(songid)
        acc = float(self.corr) / len(self.val)
        print('accuracy: %.06f'%acc)
        cm = confusion_matrix(self._true, self._prd)
        self._true = []
        self._prd = []
        return cm


if __name__ == '__main__':
    e = Eval()
    fire.Fire({'run': e.run})
