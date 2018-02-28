import os
import numpy as np
import librosa
import cPickle
import fire
import tqdm


class Extract:
    def __init__(self):
        self.npy_16k_path = '/data1/ExBallroom/npy_16k'
        self.npy_trp_path = '/data1/ExBallroom/npy_trp'
        self.spec_16k_path = '/data1/ExBallroom/spec_16k'
        self.spec_trp_path = '/data1/ExBallroom/spec_trp'
        self.fs = 16000 # sampling frequency (Hz)
        self.metas = cPickle.load(open('/hanmail/minz/ISMIR-2018/ExBall/metas.cPickle'))


    def run(self, iter=0, isRaw=False):
        if isRaw:
            path = self.npy_16k_path
            spec_path = self.spec_16k_path
        else:
            path = self.npy_trp_path
            spec_path = self.spec_trp_path

        for songid in tqdm.tqdm(self.metas.keys()):
            if songid[-1] == str(iter):
                _fn = os.path.join(path, songid)+'.npy'
                if os.path.exists(_fn):
                    audio = np.load(_fn)
                    fn = os.path.join(spec_path, songid)+'.npy'
                    self.get_melspec(audio, fn)


    def get_melspec(self, audio, fn):
        melspec = librosa.feature.melspectrogram(audio, sr=self.fs, n_fft=512, hop_length=256, n_mels=96)
        logam = librosa.logamplitude(melspec**2)
        np.save(open(fn, 'w'), logam)


if __name__ == '__main__':
    e = Extract()
    fire.Fire({'run': e.run})

