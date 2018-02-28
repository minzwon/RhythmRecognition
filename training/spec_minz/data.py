# -*- coding: utf-8 -*-
import os

import numpy as np
from keras.utils import np_utils


class Data:
    def __init__(self):
        pass

    @staticmethod
    def generator(filelist,
                  batch_size,
                  path,
                  metas,
                  class_getter,
                  extension='npy',
                  data_loader=np.load):  # batch_size should be a multiple of 10
        # get filelist
        fl = list(np.load(filelist))
        np.random.shuffle(fl)

        cur_i = 0

        while True:
            features, labels = [], []
            for i in range(batch_size):
                _idx = cur_i + i
                if len(fl) <= _idx:
                    cur_i = 0
                    np.random.shuffle(fl)
                    _idx = 0
                    # break
                song_id = fl[_idx][:6]
                chunk_id = int(fl[_idx][7:])
                #cur_i += 1

                fname = '%06d.npy' % (int(song_id))
                subpath = fname[:3]
                flname = os.path.join(path, fname)

                try:
                    x = data_loader(flname)
                except Exception as e:
                    print(repr(e))
                    continue
                y = class_getter(metas, song_id)
                if not isinstance(y, np.ndarray):
                    print('No meta %s' % song_id)
                    continue


                chunk_size = 512
                hop_size = 60
                x_p = x[:,chunk_id*hop_size:chunk_id*hop_size+chunk_size]
                features.append(x_p.reshape(96, chunk_size, 1))
                #Y = np.zeros(40)
                #Y[np.argmax(y)] = 1
                labels.append(y)

            cur_i += batch_size

            yield np.array(features), np.array(labels).reshape(len(features), 9)

