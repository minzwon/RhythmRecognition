import os
import numpy as np
from scipy import signal
from madmom.features import beats
import librosa
import cPickle
import fire
import tqdm


class Transpose:
    def __init__(self):
        self.audio_path = '/data1/ExBallroom/audio_16k'   # path of the raw audio
        self.npy_path = '/data1/ExBallroom/npy_16k'   # path of the numpy array
        self.target_audio_path = '/data1/ExBallroom/audio_trp'    # path of the transposed audio
        self.target_npy_path = '/data1/ExBallroom/npy_trp'  # path of the transposed numpy array
        self.target_bpm_path = '/data1/ExBallroom/bpm_vec'  # path of the bpm vector
        self.fs = 16000 # sampling frequency (Hz)
        self.tatum = 2048
        self.metas = cPickle.load(open('/hanmail/minz/ISMIR-2018/ExBall/metas.cPickle'))
        self.valid_genre = ['Chacha','Foxtrot','Jive','Quickstep','Rumba','Samba','Tango','Viennesewaltz','Waltz']


    def run(self, iter=0):
        for songid in tqdm.tqdm(self.metas.keys()):
            if songid[-1] == str(iter):
                npy_fn = os.path.join(self.npy_path, songid)+'.npy'
                if os.path.exists(npy_fn):
                    audio = np.load(npy_fn)
                    b = self.get_beats(songid)
                    bts, bpm_vec = self.get_transposed_beats(b)
                    transposed_audio = self.transpose(bts, audio)
                    # save files
                    librosa.output.write_wav(os.path.join(self.target_audio_path, songid)+'.wav', transposed_audio, self.fs)
                    np.save(open(os.path.join(self.target_npy_path, songid)+'.npy', 'w'), transposed_audio)
                    np.save(open(os.path.join(self.target_bpm_path, songid)+'.npy', 'w'), bpm_vec)


    def transpose(self, bts, audio):
        if len(bts)>0:
            transposed_audio = []
            for beat in bts:
                if beat+self.tatum-1 < len(audio):
                    chunk = audio[beat:beat+self.tatum]
                    for frame in chunk:
                        transposed_audio.append(frame)
            transposed_audio = np.array(transposed_audio)
        else:
            transposed_audio = audio
        return transposed_audio


    def get_transposed_beats(self, b):
        bpm_vec = np.zeros(220-50)
        if len(b) > 1:
            bpm = int(60.0/np.mean(np.diff(b)))
            bpm_vec[bpm-50-3:bpm-50+4] = signal.gaussian(7, 1)

            tatum_sec = float(self.tatum)/self.fs

            current_tatum = np.mean(np.diff(b))
            while current_tatum / 2 >= tatum_sec:
                n_beats = []
                for i in range(len(b)-1):
                    n_beats.append(b[i])
                    n_beats.append((b[i]+b[i+1])/2)
                n_beats.append(b[-1])
                b = n_beats
                current_tatum /= 2
            b = np.array(b)
            bts = np.array(b*self.fs, dtype=int)
            return bts, bpm_vec
        else:
            return [], bpm_vec


    def get_beats(self, songid):
        act = beats.RNNDownBeatProcessor()(os.path.join(self.audio_path, songid)+'.wav')
        proc = beats.DBNDownBeatTrackingProcessor(beats_per_bar=[3,4], fps=100)(act)
        b = []
        for item in proc:
            b.append(item[0])
        return b


if __name__ == '__main__':
    t = Transpose()
    fire.Fire({'run': t.run})

