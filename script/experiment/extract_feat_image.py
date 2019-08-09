from __future__ import print_function
import sys
import time
import os.path as osp
from PIL import Image
import matplotlib.image as img
import numpy as np
import os
from sklearn.cluster import KMeans
from PreProcessImage import *
import shutil
from scipy.spatial import distance
# import Dataset
# from utils.utils.utils import measure_time
# from utils.utils.re_ranking import re_ranking
# from utils.utils.metric import cmc, mean_ap
# from utils.utils.dataset_utils import parse_im_name
from distance import normalize
# from ..utils.distance import compute_dist
# from ..utils.distance import local_dist
# from ..utils.distance import low_memory_matrix_op

class input: 

  def __init__(self,extract_feat_func=None):
    self.normalize_featurct_feat_func=extract_feat_func
    self.directory_path=r'D:\DeepLearning\New folder\Aligned_ReId\data'

  def getstackedimages(self):
      print('heher')
      ims=[]
      ims_names=[]
      ppi=PreProcessIm(resize_h_w=(256, 128))
      for f in os.listdir(self.directory_path):
          im_path = osp.join(self.directory_path, f)
          ims_names.append(f)
          print(im_path)
          im1 = img.imread(im_path)
          im = np.asarray(Image.open(im_path))
          # plt.imshow(im1)
          # plt.show()
          # return
          # cv2.waitKey(0)
          im, _ = ppi.pre_process_im(im)
          ims.append(im)
      ims=np.stack(ims,axis=0)
      # print(ims.shape)
      return ims, True,ims_names

  def eval(self,normalize_feat=True,
      use_local_distance=False,
      to_re_rank=True,
      pool_type='average'):
      return self.extract_feat(normalize_feat)


    
  def set_feat_func(self, extract_feat_func):
    self.extract_feat_func = extract_feat_func

  def extract_feat(self, normalize_feat,image=None):
    """Extract the features of the whole image set.
    Args:
      normalize_feat: True or False, whether to normalize global and local 
        feature to unit length
    Returns:
      global_feats: numpy array with shape [N, C]
      local_feats: numpy array with shape [N, H, c]
      ids: numpy array with shape [N]
      cams: numpy array with shape [N]
      im_names: numpy array with shape [N]
      marks: numpy array with shape [N]
    """
    
    global_feats, local_feats = [], []
    done = False
    step = 0
    printed = False
    st = time.time()
    last_time = time.time()
    while not done:
      ims_,done,ims_names= self.getstackedimages()
      print(ims_.shape)
      global_feat, local_feat = self.extract_feat_func(ims_)
      global_feats.append(global_feat)
      local_feats.append(local_feat)

    global_feats = np.vstack(global_feats)
    local_feats = np.concatenate(local_feats)
    if normalize_feat:
      global_feats = normalize(global_feats, axis=-1)
      # local_feats = normalize(local_feats, axis=-1)

    print(ims_names[1],"  ",ims_names[2]," ",distance.euclidean(global_feats[1],global_feats[2]))
    print(ims_names[1], "  ", ims_names[0], " ", distance.euclidean(global_feats[1], global_feats[0]))
    # print(" ", distance.euclidean(global_feats[1], global_feats[2]))
    # print(" ", distance.euclidean(global_feats[1], global_feats[0]))


    # local_feats = local_feats.reshape((local_feats.shape[0],-1))
    # conc_feats = np.append(global_feats,local_feats,axis=1)
    # print(conc_feats.shape)
    #
    # km=KMeans(n_clusters=6)
    # km.fit(global_feats)
    # list_of_labels = km.labels_
    # print(len(list_of_labels))
    # uni = np.unique(list_of_labels)
    # for i in range(len(uni)):
    #   os.mkdir(osp.join(self.directory_path,str(uni[i])))
    # for i in range(len(list_of_labels)):
    #   # path = osp.join(self.directory_path+"/"+str(list_of_labels[i]),ims_names[i])
    #   shutil.copyfile(osp.join(self.directory_path,ims_names[i]),osp.join(self.directory_path+"/"+str(list_of_labels[i]),ims_names[i]))

    return global_feats, local_feats

# def main():
#   inp=input()
#   ims=inp.getstackedimages(None)
#   # cv2.imshow("abc",ims[0])
#   # plt.imshow(ims[0].transpose(1,2,0))
#   # plt.show()
#
# if __name__ == '__main__':
#   main()
