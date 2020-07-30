import torch
import torchvision
from torchvision import transforms
from PIL import Image
import torchvision.transforms as transforms
import os
from random import sample
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from skimage import io

import random
import collections
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import glob
import cv2
import time
import glob
from multiprocessing import Pool, RawArray
import multiprocessing
from scipy.cluster.vq import *

from sklearn import preprocessing
import h5py
from sklearn.model_selection import train_test_split
import pickle
from scipy.cluster.vq import *
import sklearn


class SIFT():     
    def get_sift_features(self,filename,for_bow=True):
        img = cv2.imread(filename,0)
        sift_obj = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift_obj.detectAndCompute(img,None)
        if for_bow:
            if desc is not None:
                return (filename,desc)
            else:
                return None
        else:
            if desc is not None:
                if desc.shape[0]>100:
                    filtered_desc = desc[np.random.randint(desc.shape[0], size=100)]
                elif  0 < desc.shape[0] <=100:
                    filtered_desc = desc
                final_feature = np.mean(filtered_desc,axis=0).reshape(1,-1)
                return (class_id,fp,class_name,final_feature)    
            else:
                return None   
        
def get_bovw(image_list,ret=False,cluster_centres=None,idf=None):
    result_dummy = SIFT().get_sift_features(image_list[0])
    with Pool(processes = multiprocessing.cpu_count()) as pool:
        result = pool.map(SIFT().get_sift_features, image_list)
    result = [i for i in result if i is not None ]
    image_names = [i[0] for i in result ] 
    emb_list = [i[1] for i in result]
    embeddings = np.concatenate( emb_list, axis=0 )
    if ret==False:
        reqd_index = np.random.randint(len(embeddings), size=(100000,))
        embeddings_subset = embeddings[reqd_index]

        voc, variance = kmeans(embeddings_subset, 10000, 1) 
    else:
        voc = cluster_centres
    im_features = np.zeros((len(emb_list), len(voc)), "float32")
    for i in range(len(emb_list)):
        words, distance = vq(emb_list[i],voc)
        for w in words:
            im_features[i][w] += 1

    nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
    normalized_im_features = im_features/np.sum(im_features,axis=1).reshape(-1,1)
    if ret:
        idf = idf
    else:
        idf = np.array(np.log((1.0*len(image_names)+1) / (1.0*nbr_occurences + 1)), 'float32')
    normalized_im_features_idf = normalized_im_features*idf
    #im_features = preprocessing.normalize(im_features, norm='l2')
    im_features_idf = im_features*idf
    return im_features_idf,normalized_im_features_idf,image_names,voc,idf

def find_scene(query,db_emb,db_fp,cluster_centres,idf):
    im_features_idf,normalized_features_idf,fp,_,_ =  get_bovw(query,ret=True,cluster_centres=cluster_centres,idf=idf)
    try:
        cosine_sim = sklearn.metrics.pairwise.cosine_similarity(normalized_features_idf.reshape(1,-1),db_emb)
    except:
        cosine_sim = sklearn.metrics.pairwise.cosine_similarity(normalized_features_idf,db_emb)
    sorted_indices = cosine_sim.reshape(-1).argsort()[::-1]
    sorted_scores = cosine_sim.reshape(-1)[sorted_indices]
    ret_results = np.array(db_fp)[sorted_indices]
    return ret_results,sorted_scores