import os
import numpy as np
import fire
import tqdm


class Split:
    def __init__(self):
        self.spec_path = '/data1/ExBallroom/spec_16k'
        self.tr = np.load('../../tr')
        self.val = np.load('../../val')


    def run(self, hopsize=60):
        trc = []
        valc = []
        for song in tqdm.tqdm(self.tr):
            fn = os.path.join(self.spec_path, song)+'.npy'
            spec = np.load(fn)
            nc = (spec.shape[1]-512) / hopsize
            for i in range(nc):
                trc.append(song+'_'+str(i))

        for song in tqdm.tqdm(self.val):
            fn = os.path.join(self.spec_path, song)+'.npy'
            spec = np.load(fn)
            nc = (spec.shape[1]-512) / hopsize
            for i in range(nc):
                valc.append(song+'_'+str(i))

        np.save(open('trc','w'), trc)
        np.save(open('valc','w'),valc)


if __name__ == '__main__':
    s = Split()
    fire.Fire({'run': s.run})
