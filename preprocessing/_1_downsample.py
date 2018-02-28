import os
import numpy as np
import librosa
import cPickle
import fire
import tqdm


class Downsample:
    def __init__(self):
        self.audio_path = '/data1/ExBallroom/audio_raw'   # path of the raw audio
        self.target_path = '/data1/ExBallroom/audio_16k'    # path of the downsampled audio
        self.npy_path = '/data1/ExBallroom/npy_16k'   # path of the numpy array
        self.fs = 16000 # sampling frequency (Hz)
        self.metas = cPickle.load(open('/hanmail/minz/ISMIR-2018/ExBall/metas.cPickle'))
        self.valid_genre = ['Chacha','Foxtrot','Jive','Quickstep','Rumba','Samba','Tango','Viennesewaltz','Waltz']


    def run(self, iter=0):
        for songid in tqdm.tqdm(self.metas.keys()):
            if songid[-1] == str(iter):
                genre = self.metas[songid]
                if genre in self.valid_genre:
                    self.down_sample(songid, genre)


    def down_sample(self, songid, genre):
        audio, fs = librosa.load(os.path.join(self.audio_path, genre, songid)+'.mp3', self.fs)
        librosa.output.write_wav(os.path.join(self.target_path, songid)+'.wav', audio, fs)
        np.save(open(os.path.join(self.npy_path, songid)+'.npy', 'w'), audio)



if __name__ == '__main__':
    d = Downsample()
    fire.Fire({'run': d.run})

