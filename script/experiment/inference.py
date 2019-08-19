"""Train with optional Global Distance, Local Distance, Identification Loss."""
from __future__ import print_function

import sys
sys.path.insert(0, '.')
import argparse
import torch
from torch.nn.parallel import DataParallel
from aligned_reid.utils.utils import load_state_dict
from torch.autograd import Variable

from .PreProcessImage import *
from aligned_reid.utils.utils import str2bool
import time
import os.path as osp
from aligned_reid.utils.utils import set_devices
import numpy as np
from .distance import normalize


from aligned_reid.model.Model import Model


from .extract_feat_image import input

class Config(object):
  def __init__(self):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--sys_device_ids', type=eval, default=(-1,))
    parser.add_argument('-llw', '--l_loss_weight', type=float, default=0.)
    parser.add_argument('--only_test', type=str2bool, default=True)
   
    parser.add_argument('--model_weight_file', type=str, default='model_weight.pth')


    args = parser.parse_known_args()[0]

  
    self.sys_device_ids = args.sys_device_ids

    self.local_conv_out_channels = 128
    
    self.l_loss_weight = args.l_loss_weight

    self.only_test = args.only_test

    

    self.model_weight_file = args.model_weight_file


class ExtractFeature(object):
  """A function to be called in the val/test set, to extract features.
  Args:
    TVT: A callable to transfer images to specific device.
  """

  def __init__(self, model, TVT):
    self.model = model
    self.TVT = TVT

  def __call__(self, ims):
    old_train_eval_model = self.model.training
    # Set eval mode.
    # Force all BN layers to use global mean and variance, also disable
    # dropout.
    self.model.eval()
    ims = Variable(self.TVT(torch.from_numpy(ims).float()))
    global_feat, local_feat = self.model(ims)[:2]
    global_feat = global_feat.data.cpu().numpy()
    local_feat = local_feat.data.cpu().numpy()
    # Restore the model to its old train/eval mode.
    self.model.train(old_train_eval_model)
    return global_feat, local_feat

class Aligned_reid():
  def __init__(self):
    self.cfg = Config()
    self.inp=input()



    TVT, TMO = set_devices(self.cfg.sys_device_ids)

  # if cfg.seed is not None:
  #   set_seed(cfg.seed)


    self.model = Model(local_conv_out_channels=self.cfg.local_conv_out_channels)
  # Model wrapper
    self.model_w = DataParallel(self.model)
    self.load_model()
    self.extract_features=ExtractFeature(self.model_w,TVT)
    self.inp.set_feat_func(ExtractFeature(self.model_w, TVT))

  

  def load_model(self):
    if self.cfg.model_weight_file != '':
      map_location = (lambda storage, loc: storage)
      sd = torch.load(self.cfg.model_weight_file, map_location=map_location)
      load_state_dict(self.model, sd)
      print('Loaded model weights from {}'.format(self.cfg.model_weight_file))
    else:
      load_ckpt(modules_optims, self.cfg.ckpt_file)

    
  
  
  def infer(self,images,normalize_feat):
    # ppi=PreProcessIm(resize_h_w=(256, 128))
    # preprocessed_images=[]
    # for im in images:
    #   im, _ = ppi.pre_process_im(im)
    #   preprocessed_images.append(im)
    # preprocessed_images=np.stack(preprocessed_images)
    
    global_feats,local_feats=self.extract_features(images)
    if normalize_feat:
      global_feats = normalize(global_feats, axis=-1)
    return global_feats,local_feats
    # self.inp.eval(images)
