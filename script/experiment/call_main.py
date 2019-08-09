from train import *
from extract_feat_image import input
import os
from PIL import Image
from PreProcessImage import *

def stacking(images):
    inp=input()
    stacked_images,_=inp.getstackedimages()
    stacked_images, _ = getstackedimages(stacked_images)
    global_feats, local_feats = main_2(stacked_images)
    print(global_feats.shape)

def getstackedimages(st_imgs):
  ims=[]
  ims_names=[]
  ppi=PreProcessIm(resize_h_w=(256, 128))
  for f in range(st_imgs.shape[0]):
      # im_path = osp.join(self.directory_path, f)
      # ims_names.append(f)
      # print(im_path)
      # im1 = img.imread(im_path)
      # im = np.asarray(Image.open(im_path))
      # plt.imshow(im1)
      # plt.show()
      # return
      # cv2.waitKey(0)
      im, _ = ppi.pre_process_im(st_imgs[f])
      ims.append(im)
  ims=np.stack(ims,axis=0)
  print(ims.shape)
  return ims, True

def prepImagefetaures():
  ims=[]
  ims_names=[]
  ppi=PreProcessIm(resize_h_w=(256, 128))
  directory_path = "/home/developer/Desktop/reid/market1501/images"
  for f in os.listdir(path):
      im_path = osp.join(directory_path, f)
      # ims_names.append(f)
      # print(im_path)
      # im1 = img.imread(im_path)
      im = np.asarray(Image.open(im_path))
      # plt.imshow(im1)
      # plt.show()
      # return
      # cv2.waitKey(0)
      im, _ = ppi.pre_process_im(im)
      np.expand_dims(im,axis=0)
      print(im.shape)
      ims.append(im)
  ims=np.stack(ims,axis=0)
  print(ims.shape)
  return ims, True
if __name__ == '__main__':
  main()

