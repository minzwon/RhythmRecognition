import os
import numpy as np
import librosa
import essentia
from essentia.standard import MonoWriter
import cPickle
import fire
import tqdm


class Fool:
    def __init__(self):
        self.npy_path = '/data1/ExBallroom/npy_16k'   # path of the numpy array
        self.fool_audio_path = '/data1/ExBallroom/11p_audio'
        self.fool_npy_path = '/data1/ExBallroom/11p_npy'
        self.fs = 16000 # sampling frequency (Hz)
        self.metas = cPickle.load(open('/hanmail/minz/ISMIR-2018/ExBall/metas.cPickle'))
        self.valid_genre = ['Chacha','Foxtrot','Jive','Quickstep','Rumba','Samba','Tango','Viennesewaltz','Waltz']


    def run(self, iter=0):
        for songid in tqdm.tqdm(self.metas.keys()):
            if songid[-1] == str(iter):
                npy_fn = os.path.join(self.npy_path, songid)+'.npy'
                if int(songid[-1])%2==0:
                    stretch_rate = 0.89
                elif int(songid[-1])%2==1:
                    stretch_rate = 1.11
                if os.path.exists(npy_fn):
                    self.get_fool(songid, stretch_rate)


    def get_fool(self, songid, stretch_rate):
        audio = np.load(os.path.join(self.npy_path, songid)+'.npy')
        t_audio = librosa.effects.time_stretch(audio, stretch_rate)
        MonoWriter(filename=os.path.join(self.fool_audio_path, songid)+'.wav', format='wav', sampleRate=self.fs)(t_audio)
        np.save(os.path.join(self.fool_npy_path, songid)+'.npy', t_audio)



if __name__ == '__main__':
    f = Fool()
    fire.Fire({'run': f.run})

