#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 16:35:00 2023

@author: whitaker-160
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import bisect
import matplotlib.pyplot as plt
import numpy as np
import scipy.io,random

class pCRDataset(Dataset):

    def __init__(self,datatype, info_file, root_dir, cyc_num, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if datatype=='US' or datatype=='DOT':
            self.datatype = datatype
            self.us_frame = pd.read_excel(info_file,sheet_name=self.datatype)
            self.us_num = [0]
            for i in range(len(self.us_frame['Cyc'+cyc_num])):
                if self.us_num==[]: self.us_num=[self.us_frame['Pre_chemo'][i]*self.us_frame['Cyc'+cyc_num][i]]
                else:
                    self.us_num.append(self.us_num[-1]+self.us_frame['Pre_chemo'][i]*self.us_frame['Cyc'+cyc_num][i])
            
            self.cyc = cyc_num
            self.root_dir = root_dir
            self.transform = transform
        elif datatype=='USDOT':
            self.datatype = datatype
            self.us_frame = pd.read_excel(info_file,sheet_name='US')
            self.dot_frame = pd.read_excel(info_file,sheet_name='DOT')
            self.us_num = []
            self.dot_num = []
            self.usdot_num=[0]
            for i in range(len(self.us_frame['Cyc'+cyc_num])):
                if self.us_num==[]: 
                    self.us_num=[self.us_frame['Pre_chemo'][i]*self.us_frame['Cyc'+cyc_num][i]]
                    self.dot_num=[self.dot_frame['Pre_chemo'][i]*self.dot_frame['Cyc'+cyc_num][i]]
                    self.usdot_num+=[self.us_num[-1]*self.dot_num[-1]]
                else:
                    self.us_num.append(self.us_frame['Pre_chemo'][i]*self.us_frame['Cyc'+cyc_num][i])
                    self.dot_num.append(self.dot_frame['Pre_chemo'][i]*self.dot_frame['Cyc'+cyc_num][i])
                    self.usdot_num.append(self.usdot_num[-1]+self.us_num[-1]*self.dot_num[-1])
            self.cyc = cyc_num
            self.root_dir = root_dir
            self.transform = transform
        
    def __len__(self):
        if self.datatype=='US' or self.datatype=='DOT':
            return self.us_num[-1]
        if self.datatype=='USDOT':
            return self.usdot_num[-1]//100

    def __getitem__(self, idx):
        

        if self.datatype=='US':
            if torch.is_tensor(idx):
                idx = idx.tolist()
            # find image index
            p_id = bisect.bisect_right(self.us_num, idx)-1
            prechemo_id = (idx-self.us_num[p_id])//self.us_frame['Cyc'+self.cyc][p_id]
            cyc_id = (idx-self.us_num[p_id])%self.us_frame['Cyc'+self.cyc][p_id]
            # locate folder path
            p_folder = os.path.join(self.root_dir, self.us_frame['P_ID'][p_id])
            #print(self.us_frame['P_ID'][p_id], prechemo_id, cyc_id)
            
            pre_chemo_folder = [f for f in sorted(os.listdir(p_folder)) if 'pre' in f and 'chemo' in f]
            pre_chemo_path = os.path.join(p_folder, pre_chemo_folder[0])
            cyc_folder = [f for f in sorted(os.listdir(p_folder)) if 'cyc'+self.cyc in f or 'cycle'+self.cyc in f]
            cyc_path = os.path.join(p_folder, cyc_folder[0])
            
            # get image
            pre_chemo_image_path = os.path.join(pre_chemo_path, sorted(os.listdir(pre_chemo_path))[prechemo_id])
            cyc_image_path = os.path.join(cyc_path, sorted(os.listdir(cyc_path))[cyc_id])
            # read images
            pre_chemo_image = plt.imread(pre_chemo_image_path)
            cyc_image = plt.imread(cyc_image_path)
            if self.transform:
                pre_chemo_image_tensor=self.transform(pre_chemo_image)
                cyc_image_tensor=self.transform(cyc_image)
            # label
            labels = self.us_frame['pCR'][p_id]
            #labels = np.eye(2)[labels]
            sample = {'image': [pre_chemo_image_tensor,cyc_image_tensor], 'labels': labels}
            
            

            return sample
            
        elif self.datatype=='DOT':        
            if torch.is_tensor(idx):
                idx = idx.tolist()
            # find image index
            p_id = bisect.bisect_right(self.us_num, idx)-1
            prechemo_id = (idx-self.us_num[p_id])//self.us_frame['Cyc'+self.cyc][p_id]
            cyc_id = (idx-self.us_num[p_id])%self.us_frame['Cyc'+self.cyc][p_id]
            # locate folder path
            p_folder = os.path.join(self.root_dir, self.us_frame['P_ID'][p_id])
            #print(self.us_frame['P_ID'][p_id], prechemo_id, cyc_id)
            
            pre_chemo_folder = [f for f in sorted(os.listdir(p_folder)) if 'pre' in f and 'chemo' in f]
            pre_chemo_path = os.path.join(p_folder, pre_chemo_folder[0], 'saved_data')
            cyc_folder = [f for f in sorted(os.listdir(p_folder)) if 'cyc'+self.cyc in f or 'cycle'+self.cyc in f]
            cyc_path = os.path.join(p_folder, cyc_folder[0], 'saved_data')   
            
            # get image
            pre_chemo_image_path = os.path.join(pre_chemo_path, sorted(os.listdir(pre_chemo_path))[prechemo_id])
            cyc_image_path = os.path.join(cyc_path, sorted(os.listdir(cyc_path))[cyc_id])
            # read images
            pre_chemo_image = scipy.io.loadmat(pre_chemo_image_path)['hemo']
            pre_chemo_image=pre_chemo_image
            cyc_image = scipy.io.loadmat(cyc_image_path)['hemo']
            cyc_image=cyc_image
            if self.transform:
                pre_chemo_image_tensor=self.transform(pre_chemo_image)
                cyc_image_tensor=self.transform(cyc_image)
            # label
            labels = self.us_frame['pCR'][p_id]
            #labels = np.eye(2)[labels]
            sample = {'image': [pre_chemo_image_tensor,cyc_image_tensor], 'labels': labels}
            
            

            return sample
        elif self.datatype=='USDOT': 
            if torch.is_tensor(idx):
                idx = idx.tolist()
            # find image index
            idx=max(0,idx-1)*100#+random.randint(0, 99)
            
            p_id = bisect.bisect_right(self.usdot_num, idx)-1
            
            us_idx= (idx-self.usdot_num[p_id])//self.dot_num[p_id]
            dot_idx= (idx-self.usdot_num[p_id])%self.dot_num[p_id]
            
            
            ## US Data
            us_prechemo_id = us_idx//self.us_frame['Cyc'+self.cyc][p_id]
            us_cyc_id = us_idx%self.us_frame['Cyc'+self.cyc][p_id]
            # locate folder path
            us_p_folder = os.path.join(self.root_dir, self.us_frame['P_ID'][p_id])
            #print(self.us_frame['P_ID'][p_id], prechemo_id, cyc_id)
            
            us_pre_chemo_folder = [f for f in sorted(os.listdir(us_p_folder)) if 'pre' in f and 'chemo' in f]
            us_pre_chemo_path = os.path.join(us_p_folder, us_pre_chemo_folder[0])
            us_cyc_folder = [f for f in sorted(os.listdir(us_p_folder)) if 'cyc'+self.cyc in f or 'cycle'+self.cyc in f]
            us_cyc_path = os.path.join(us_p_folder, us_cyc_folder[0])
            
            # get image
            us_pre_chemo_image_path = os.path.join(us_pre_chemo_path, sorted(os.listdir(us_pre_chemo_path))[us_prechemo_id])
            us_cyc_image_path = os.path.join(us_cyc_path, sorted(os.listdir(us_cyc_path))[us_cyc_id])
            # read images
            us_pre_chemo_image = plt.imread(us_pre_chemo_image_path)
            us_cyc_image = plt.imread(us_cyc_image_path)
            if self.transform:
                us_pre_chemo_image_tensor=self.transform(us_pre_chemo_image)
                us_cyc_image_tensor=self.transform(us_cyc_image)

            
            ## DOT data
            
            dot_prechemo_id = dot_idx//self.dot_frame['Cyc'+self.cyc][p_id]
            dot_cyc_id = dot_idx%self.dot_frame['Cyc'+self.cyc][p_id]
            # locate folder path
            dot_p_folder = os.path.join(self.root_dir, self.dot_frame['P_ID'][p_id])
            #print(self.us_frame['P_ID'][p_id], prechemo_id, cyc_id)
            
            
            dot_pre_chemo_folder = [f for f in sorted(os.listdir(dot_p_folder)) if 'pre' in f and 'chemo' in f]
            dot_pre_chemo_path = os.path.join(dot_p_folder, dot_pre_chemo_folder[0], 'saved_data')
            dot_cyc_folder = [f for f in sorted(os.listdir(dot_p_folder)) if 'cyc'+self.cyc in f or 'cycle'+self.cyc in f]
            dot_cyc_path = os.path.join(dot_p_folder, dot_cyc_folder[0], 'saved_data')   
            
            # get image
            dot_pre_chemo_image_path = os.path.join(dot_pre_chemo_path, sorted(os.listdir(dot_pre_chemo_path))[dot_prechemo_id])
            dot_cyc_image_path = os.path.join(dot_cyc_path, sorted(os.listdir(dot_cyc_path))[dot_cyc_id])
            # read images
            dot_pre_chemo_image = scipy.io.loadmat(dot_pre_chemo_image_path)['hemo']
            dot_cyc_image = scipy.io.loadmat(dot_cyc_image_path)['hemo']
                    
            if self.transform:
                dot_pre_chemo_image_tensor=self.transform(dot_pre_chemo_image)
                dot_cyc_image_tensor=self.transform(dot_cyc_image)
            # label
            labels = self.us_frame['pCR'][p_id]
            #labels = np.eye(2)[labels]
            sample = {'image': [dot_pre_chemo_image_tensor,dot_cyc_image_tensor,us_pre_chemo_image_tensor,us_cyc_image_tensor], 'labels': labels}
            
            
    
            return sample