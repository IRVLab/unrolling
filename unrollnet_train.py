from __future__ import print_function, division
import os
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from data_loader import dataLoader
from unrollnet import UnrollNet

# load data
# data_dir = '/mnt/data2/unRolling_Shutter/Rolling_Shutter_TUM'
data_dir = os.path.join(os.getcwd(), "data")
seqs = [1,2,3,4,5] 
data_loader = dataLoader(data_path=data_dir, seqs=seqs) 
imgs_rs, flows_gs2rs = data_loader.loadAll()

# checkpoint
checkpoint_dir = os.path.join(os.getcwd(), "checkpoints")
if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)
ckpt_name = os.path.join(checkpoint_dir, ("model.hdf5"))
checkpoint = ModelCheckpoint(ckpt_name, monitor='loss', save_weights_only=True, save_best_only=True)

# trainning
model_loader = UnrollNet(data_loader.getImgShape())
model_loader.model.compile(optimizer=Adam(), loss=model_loader.flowLoss)
# model_loader.model.load_weights(ckpt_name)
model_loader.model.fit(imgs_rs, flows_gs2rs, batch_size=5, epochs=1000, callbacks=[checkpoint])
