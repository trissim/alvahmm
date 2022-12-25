#!/usr/bin/env python
# coding: utf-8

# # Home-made machinery for image analysis including:
# ## A. parting and branching

# In[1]:


'''
author: Alvason Zhenhua Li
date:   from 11/21/2016 to 02/27/2018
Home-made machinery
'''
### 02/27/2018, updated for using AlvaHmm_class 
###############################################
### open_package +++
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import sys
import os
import logging
logging.basicConfig(format = '%(asctime)s %(message)s',
                    level = logging.INFO, stream = sys.stdout)
if __name__ == '__main__': logging.info('(previous_run_time)')
### open_package ---


# ## branching_connected_map_by_AlvaHmm

# ## branch_HMM_connect_way

# ## pixel_connect_way

# In[5]:


'''
author: Alvason Zhenhua Li
date:   11/21/2016
Home-made machinery
'''
### find_connect_way aaa ##########
import queue
def find_connect_way(img, i, j):
    dx = [0, 0, 1, 1, 1, -1, -1, -1]
    dy = [1, -1, 0, 1, -1, 0, 1, -1]
    x = []
    y = []
    q = queue.Queue()
    if img[i][j] == 1:
        q.put((i, j))
    while q.empty() == False:
        u, v = q.get()
        x.append(u)
        y.append(v)
        for k in range(8):
            xx = u + dx[k]
            yy = v + dy[k]
            if img[xx][yy] == 1:
                ### set checked pixel
                img[xx][yy] = 2
                q.put((xx, yy))
    ###
    connect_xx = np.asarray(x)
    connect_yy = np.asarray(y)
    return (connect_xx, connect_yy)
### find_connect_way zzz ##########
###
### part_connect_way +++ ##########
def part_connect_way(connect_xx, connect_yy):
    ### sorting_xx
    sx = np.argsort(connect_xx)
    sx_xx = connect_xx[sx]
    sx_yy = connect_yy[sx]
    ### branching_way
    img = np.zeros([sx_yy.max() + 2, sx_xx.max() + 2])
    img[sx_yy, sx_xx] = 1
    ###
    xx = sx_xx
    yy = sx_yy
    way_xx_all = []
    way_yy_all = []
    way_rr_all = []
    x_range = xx.max() - xx.min()
    y_range = yy.max() - yy.min()
    total_scan_pixel = int(y_range * x_range / 2)
    ### nearest_neighbor from low_priority (dx = 0, dy = -1) to high_priority (dx = 1, dy = 0)
    dx = [0, 0, 1, 1, 1] 
    dy = [-1, 1, -1, 1, 0]
    dr = [1, 1, 2**(0.5), 2**(0.5), 1]
    total_neighbor = len(dx)
    for i in range(len(xx)):
        way_xx = []
        way_yy = []
        way_rr = []
        if (img[yy[i]][xx[i]] == 1):
            way_xx.append(xx[i])
            way_yy.append(yy[i])
            for i in range(total_scan_pixel):
                xc = way_xx[-1]
                yc = way_yy[-1]
                x_next = None
                y_next = None
                for n in range(total_neighbor):
                    xn = xc + dx[n]
                    yn = yc + dy[n]
                    if (img[yn][xn] == 1):
                        x_next = xn
                        y_next = yn
                        r_next = dr[n]
                ### checking connection
                if (x_next == way_xx[-1] or x_next == way_xx[-1] + 1): 
                    way_xx.append(x_next)
                    way_yy.append(y_next)
                    way_rr.append(r_next)
                    ### marking checked pixel 
                    img[y_next][x_next] = 2
            ### only append connection_case (avoiding '[]' non_connection_case)
            if way_rr != []:
                way_xx = np.asarray(way_xx)
                way_yy = np.asarray(way_yy)
                way_rr = np.asarray(way_rr)
                #print(way_xx)
                way_xx_all.append(way_xx)
                way_yy_all.append(way_yy)
                way_rr_all.append(way_rr)
        ###
        branch_xx_all = np.asarray(way_xx_all)
        branch_yy_all = np.asarray(way_yy_all)
        branch_rr_all = np.asarray(way_rr_all)
    return (branch_xx_all, branch_yy_all, branch_rr_all)
### part_connect_way --- ##########
###
def connect_way(chain_mmm_fine,
                line_length_min = None,
                free_zone_from_y0 = None,
               ):
    ###
    mmm = np.copy(chain_mmm_fine)
    ### image_size +++
    total_pixel_y, total_pixel_x = mmm.shape
    ### image_size ---
    if line_length_min is None:
        line_length_min = 16 
    if free_zone_from_y0 is None:
        free_zone_from_y0 = 4
    ### avoiding boundary +++
    boundary = 4
    mmm[0:boundary, :] = 0
    mmm[-boundary:, :] = 0
    mmm[:, 0:boundary] = 0
    mmm[:, -boundary:] = 0
    ### avoiding boundary ---
    print('total all_connect_way_pixel =', mmm.sum())
    ####################
    ### root_connect_way
    ####################
    tree_xx = []
    tree_yy = []
    ### avoiding changing the original image
    binary_image = np.copy(mmm)
    for yn in range(mmm.shape[0]):
        for xn in range(mmm.shape[1]):
            connect_yy, connect_xx = find_connect_way(binary_image, yn, xn)
            if len(connect_xx) > line_length_min:
                tree_xx.append(connect_xx)
                tree_yy.append(connect_yy)
    #####################################
    ### find the tip of connect_way +++ ###
    tip_xx = []
    tip_yy = []
    ### 
    for i in range(len(tree_xx)):
        ###
        connect_xx = tree_xx[i]
        connect_yy = tree_yy[i]
        ###
        index_max = np.argmax(connect_xx)
        tip_xx.append(connect_xx[index_max])
        tip_yy.append(connect_yy[index_max])
    ###
    tip_xx = np.asarray(tip_xx)
    tip_yy = np.asarray(tip_yy)
    ### find the tip of connect_way --- ###
    ### filter out non_root_tip +++
    root_tree_xx = []
    root_tree_yy = []
    for i in range(len(tree_yy)):
        y_min_index = np.argmax(tree_yy[i]) 
        if (tree_yy[i].max() >= free_zone_from_y0):
            connect_xx = tree_xx[i]
            connect_yy = tree_yy[i]  
            root_tree_xx.append(connect_xx)
            root_tree_yy.append(connect_yy)
    ### filter out tip ---
    ### find the tip of connect_way +++ ###
    root_tip_xx = []
    root_tip_yy = []
    ### 
    for i in range(len(root_tree_xx)):
        ###
        connect_xx = root_tree_xx[i]
        connect_yy = root_tree_yy[i]
        ###
        index_max = np.argmax(connect_xx)
        root_tip_xx.append(connect_xx[index_max])
        root_tip_yy.append(connect_yy[index_max])
    ###
    root_tip_xx = np.asarray(root_tip_xx)
    root_tip_yy = np.asarray(root_tip_yy)
    ### find the tip of connect_way --- 
    return(root_tree_yy,
           root_tree_xx,
           root_tip_yy,
           root_tip_xx)
    ################################
###
###########################################################################
'''#####################################################################'''
###########################################################################

