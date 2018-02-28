import os
import numpy as np
import tqdm

tr = np.load('../../tr')
val = np.load('../../val')


trc = []
valc = []

path = '/data1/ExBallroom/spec_16k/'
for item in tqdm.tqdm(tr):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            trc.append('0_'+item+'_'+str(nc))
for item in tqdm.tqdm(val):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            valc.append('0_'+item+'_'+str(nc))

path = '/data1/ExBallroom/3p_spec/'
for item in tqdm.tqdm(tr):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            trc.append('3_'+item+'_'+str(nc))
for item in tqdm.tqdm(val):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            valc.append('3_'+item+'_'+str(nc))

path = '/data1/ExBallroom/5p_spec/'
for item in tqdm.tqdm(tr):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            trc.append('5_'+item+'_'+str(nc))
for item in tqdm.tqdm(val):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            valc.append('5_'+item+'_'+str(nc))

path = '/data1/ExBallroom/7p_spec/'
for item in tqdm.tqdm(tr):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            trc.append('7_'+item+'_'+str(nc))
for item in tqdm.tqdm(val):
    spec = np.load(os.path.join(path,item)+'.npy')
    if spec.shape[1]>512:
        nc = (spec.shape[1]-512)/60
        for i in range(nc):
            valc.append('7_'+item+'_'+str(nc))


trc = np.array(trc)
valc = np.array(valc)

np.random.shuffle(trc)
np.random.shuffle(valc)

np.save(open('trc','w'),trc)
np.save(open('valc','w'),valc)
