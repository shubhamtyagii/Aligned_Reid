from script.experiment.inference import Aligned_reid
import os
import os.path as osp
import matplotlib.image as img
import numpy as np
from PIL import Image
from script.experiment.PreProcessImage import *
from sklearn.neighbors import KNeighborsClassifier
class Person_Reid():
    def __init__(self,threshold):
        self.inference=Aligned_reid()
        self.id_feature_dict=dict()
        self.knn_model=None
        self.largest_id=0
        self.threshold=threshold
    def getstackedimages(self,images):
        # directory_path=r'D:\DeepLearning\New folder\Aligned_ReId\data'
        # ims=[]
        # ims_names=[]
        ppi=PreProcessIm(resize_h_w=(256, 128))
        pre_processed=[]
        ims=[]
        for im in images:
            # print(im_path)
            # im1 = img.imread(im_path)
            # im = np.asarray(Image.open(im_path))
            # plt.imshow(im1)
            # plt.show()
            # return
            # cv2.waitKey(0)
            im, _ = ppi.pre_process_im(im)
            ims.append(im)
        ims=np.stack(ims,axis=0)
        # print(ims.shape)
        return ims #, True,ims_names
    def get_person_id(self,images):
        
        images=self.getstackedimages(images)
        
        global_feats,local_feats=self.inference.infer(images,True)

        return self.match_features(global_feats,local_feats)

        #return global_feats,local_feats
    
    def match_features(self,global_feats,local_feats):
        if(len(self.id_feature_dict)==0):
            for i in range(global_feats.shape[0]):
                self.id_feature_dict[i]=global_feats[i]
            self.largest_id=i
        results={}
        if(self.knn_model==None and len(self.id_feature_dict)>0):
            self.knn_model=KNeighborsClassifier(n_neighbors=1)
            X_train=np.array(list(self.id_feature_dict.values()))
            print(X_train.shape)
            #X_train=X_train.reshape((-1,204))
            print(X_train.shape)
            y_train=np.array(list(self.id_feature_dict.keys()))
            self.knn_model.fit(X_train,y_train)
            for i in range(y_train.shape[0]):
                results[i]=y_train[i]
            
        else:
            predictions=self.knn_model.kneighbors(global_feats)
            distances=predictions[0]
            predicted_id=predictions[1]
            dict_changed=False
            
            print(distances)
            for i in range (len(distances)):
                if(distances[i]>self.threshold):
                    self.largest_id+=1
                   
                    self.id_feature_dict[self.largest_id]=global_feats[i]
                    results[i]=self.largest_id
                    
                    dict_changed=True
                    
                else:
                    results[i]=predicted_id[i][0]
                    
            
            if(dict_changed):
                X_train=np.array(list(self.id_feature_dict.values()))
                
                y_train=np.array(list(self.id_feature_dict.keys()))
                self.knn_model.fit(X_train,y_train)
        return results
            





directory_path=r'D:\DeepLearning\Aligned_ReId\data'
ims=[]
predict=[]
ims_names=[]
for f in os.listdir(directory_path):
    im_path = osp.join(directory_path, f)
    im1 = img.imread(im_path)
    im = np.asarray(Image.open(im_path))
    if(len(ims)==3):
        print(im_path,'predict')
        predict.append(im)
    else:
        print(im_path)
        ims.append(im)
        



obj=Person_Reid(0.3)


res=obj.get_person_id(ims)
print(res)

res=obj.get_person_id(predict)  
print(res)
