#!/usr/bin/env python
# coding: utf-8

# # Home-made machinery for image analysis including:
# ## A. Neurite\Cell determination and branching based on Markov_Chain

# In[1]:


'''
author: Alvason Zhenhua Li
date:   from 01/13/2017 to 02/27/2018
Home-made machinery
'''
### 02/27/2018, updated for pair_seed and AlvaHmm_class
### 02/21/2018, updated for normalized_image
############################################
### open_package +++
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys
import os
import logging
logging.basicConfig(format = '%(asctime)s %(message)s',
                    level = logging.INFO, stream = sys.stdout)
if __name__ == '__main__': logging.info('(previous_run_time)')
### open_package ---


# ## pair_HMM_chain

# In[2]:


'''
author: Alvason Zhenhua Li
date:   02/27/2018
Home-made machinery
'''
class AlvaHmm(object):
    def __init__(cell,
                 likelihood_mmm, #image
                 total_node = None, #nodes of HMM_chain
                 total_path = None, #possible paths of each node
                 node_r = None, #radial_distance between nodes
                 node_angle_max = None, #maximum searching_angle_range between starting and ending node_path
                ):
        ###
        if likelihood_mmm.min() < 0 or likelihood_mmm.max() > 1:
            ### normalize +++
            likelihood_mmm = likelihood_mmm - likelihood_mmm.min()
            likelihood_mmm = likelihood_mmm / likelihood_mmm.max()
            print('normalization_likelihood_mmm =', likelihood_mmm.min(), likelihood_mmm.max())
            ### normalize ---
        if total_node is None:
            total_node = 16 #likelihood_mmm.shape[0] / 5
        if total_path is None:
            total_path = 8
        if node_r is None:
            node_r = 5
        if node_angle_max is None:
            node_angle_max = 90 * (np.pi / 180)
        ###
        ###
        cell.mmm = likelihood_mmm
        cell.total_node = int(total_node)
        cell.total_path = int(total_path)
        cell.node_r = int(node_r)
        cell.node_angle_max = node_angle_max
        ### possible paths starting from seed +++
        ### setting 8x paths is good enough for practical cases
        ### additional 1 in (1+8x) is for symmetric computation: angle_range / (total_seed_path -1)
        cell.total_path_seed = int(1 + 8 + 8 * np.floor(total_path / 8))
        ### possible paths starting from seed ---
    ###
    def _prob_sum_state(cell, x0, y0, node_angle):
        ### image_size +++
        total_pixel_y, total_pixel_x = cell.mmm.shape
        ### image_size ---
        x1 = int(cell.node_r * np.cos(node_angle) + x0)
        y1 = int(cell.node_r * np.sin(node_angle) + y0)
        ###
        if (y1 < 0 or y1 >= total_pixel_y - 1 or             x1 < 0 or x1 >= total_pixel_x - 1 or             y0 < 0 or y0 >= total_pixel_y - 1 or             x0 < 0 or x0 >= total_pixel_x - 1):
            prob = -np.inf
        else:
            prob = 0
            ### prob_sum of the linear_interpolation_points between two nodes
            for rn in range(1, cell.node_r + 1):
                rx = int(rn * np.cos(node_angle) + x0)
                ry = int(rn * np.sin(node_angle) + y0)
                ### avoiding log0 problem
                if cell.mmm[ry, rx] == 0:
                    prob = prob + 0
                else:
                    ### 255 is avoiding negative log_value of normalized_mmm whose range is 1
                    prob = prob + np.log(cell.mmm[ry, rx] * 255)
        ###
        return (prob, x1, y1)
    #########################
    ###
    def _node_link_intensity(cell, node_A, node_B):
        ### image_size +++
        total_pixel_y, total_pixel_x = cell.mmm.shape
        ### image_size ---
        ### np.int64() is making sure is int because it is used for pixel_index of mmm
        node_A_x, node_A_y = np.int64(node_A)
        node_B_x, node_B_y = np.int64(node_B)
        ###
        ox = (node_B_x - node_A_x)
        oy = (node_B_y - node_A_y)
        link_r = int((ox * ox + oy * oy)**(0.5))
        link_zone = []
        link_path = []
        for zn in np.append(np.arange(-link_r, link_r, 1), np.int64([0])):
            try:
                zn_xn = int(-oy * zn / link_r)
            except:
                zn_xn = 0
            try:
                zn_yn = int(ox * zn / link_r)
            except:
                zn_yn = 0
            ###
            for rn in range(link_r):
                link_xn = int(node_A_x + ox * (rn / link_r)) + zn_xn
                link_yn = int(node_A_y + oy * (rn / link_r)) + zn_yn
                ### boundary +++
                if link_xn < 0:
                    link_xn = 0
                if link_xn >= total_pixel_x:
                    link_xn = total_pixel_x - 1
                if link_yn < 0:
                    link_yn = 0
                if link_yn >= total_pixel_y:
                    link_yn = total_pixel_y - 1
                ### boundary ---
                link_zone.append(cell.mmm[link_yn, link_xn])
                ### three adjacent lines for better evaluation
                if zn in [0]:
                    link_path.append(cell.mmm[link_yn, link_xn])
        ###
        zone_median = np.median(link_zone)
        ###
        link_mean = np.mean(link_path)
        return (link_mean, zone_median)
    ###################################
    ###
    def node_HMM_path(cell,
                      seed_x,
                      seed_y,
                      seed_angle = None, #seed_angle of the starting seed_path
                      seed_angle_max = None, #maximum angle_range between ending seed_path(-ending, starting, +ending)
                     ):
        ###
        if seed_angle is None:
            seed_angle = 0
        if seed_angle_max is None:
            seed_angle_max = 360 * (np.pi / 180)
        ###
        ### image_size +++
        total_pixel_y, total_pixel_x = cell.mmm.shape
        ### image_size ---
        ###
        ### node +++
        node_aa = np.zeros([cell.total_node], dtype = np.float64)
        node_xx = np.zeros([cell.total_node], dtype = np.int64)
        node_yy = np.zeros([cell.total_node], dtype = np.int64)
        ### node ---
        ### node_path +++
        node_path_aa = np.zeros([cell.total_node, cell.total_path], dtype = np.float64)
        node_path_xx = np.zeros([cell.total_node, cell.total_path], dtype = np.int64)
        node_path_yy = np.zeros([cell.total_node, cell.total_path], dtype = np.int64)
        node_path_pp = np.zeros([cell.total_node, cell.total_path], dtype = np.float64)
        ###
        node_path_path0max = np.zeros([cell.total_node, cell.total_path], dtype = np.int64)
        ### node_path ---
        ### node_path_path0 +++
        node_path_path0_aa = np.zeros([cell.total_node, cell.total_path, cell.total_path], dtype = np.float64)
        node_path_path0_xx = np.zeros([cell.total_node, cell.total_path, cell.total_path], dtype = np.int64)
        node_path_path0_yy = np.zeros([cell.total_node, cell.total_path, cell.total_path], dtype = np.int64)
        node_path_path0_pp = np.zeros([cell.total_node, cell.total_path, cell.total_path], dtype = np.float64)
        ### node_path_path0 ---
        ###
        dAngle = cell.node_angle_max / (cell.total_path - 1)
        ####################################################
        ### setting initial present_node_0 +++
        ### for every path_Pn
        Nn = 0
        ### seed_path +++
        ###
        node_path_path0_aa_seed = np.zeros([cell.total_path_seed], dtype = np.float64)
        node_path_path0_xx_seed = np.zeros([cell.total_path_seed], dtype = np.int64)
        node_path_path0_yy_seed = np.zeros([cell.total_path_seed], dtype = np.int64)
        node_path_path0_pp_seed = np.zeros([cell.total_path_seed], dtype = np.float64)
        ### seed_path ---
        ### seed_path in symmetric distribution of all directions (part or whole 360_degree) +++
        dAngle_seed = seed_angle_max / (cell.total_path_seed - 1)
        ###
        for Pn in range(cell.total_path_seed):
            node_angle = (Pn * dAngle_seed) + (seed_angle - seed_angle_max / 2)
            ###
            prob, x1, y1 = cell._prob_sum_state(seed_x,
                                                seed_y,
                                                node_angle)
            ###
            node_path_path0_aa_seed[Pn] = node_angle
            node_path_path0_xx_seed[Pn] = x1
            node_path_path0_yy_seed[Pn] = y1
            node_path_path0_pp_seed[Pn] = prob
            ###
        top_path_from_seed = np.argsort(node_path_path0_pp_seed)[-cell.total_path:]
        ### seed_path in symmetric distribution of all directions (part or whole 360_degree) ---
        for Pn in range(cell.total_path):
            Pn_seed = top_path_from_seed[Pn]
            Pn_now = 0
            ###
            node_path_path0_aa[Nn, Pn, Pn_now] = node_path_path0_aa_seed[Pn_seed]
            node_path_path0_xx[Nn, Pn, Pn_now] = node_path_path0_xx_seed[Pn_seed]
            node_path_path0_yy[Nn, Pn, Pn_now] = node_path_path0_yy_seed[Pn_seed]
            node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp_seed[Pn_seed]
            ###
            Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
            node_path_path0max[Nn, Pn] = Pn_now_max
            ###
            node_path_aa[Nn, Pn] = node_path_path0_aa[Nn, Pn, Pn_now_max]
            node_path_xx[Nn, Pn] = node_path_path0_xx[Nn, Pn, Pn_now_max]
            node_path_yy[Nn, Pn] = node_path_path0_yy[Nn, Pn, Pn_now_max]
            node_path_pp[Nn, Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]
        ###
        node_aa[0] = seed_angle
        node_xx[0] = seed_x
        node_yy[0] = seed_y
        ### setting initial present_node_0 ---
        ### future_node +++
        for Nn in range(1, cell.total_node):
            ###
            Nn_now = Nn - 1
            ### for every path_Pn
            for Pn in range(cell.total_path):
                ### for every path_state_Sn
                for Pn_now in range(cell.total_path):
                    Pn_now_max = node_path_path0max[Nn_now, Pn_now]
                    node_angle = (node_path_path0_aa[Nn_now, Pn_now, Pn_now_max]
                                  - cell.node_angle_max / 2) + (Pn * dAngle)
                    prob, x1, y1 = cell._prob_sum_state(node_path_path0_xx[Nn_now, Pn_now, Pn_now_max],
                                                        node_path_path0_yy[Nn_now, Pn_now, Pn_now_max],
                                                        node_angle)
                    ###
                    node_path_path0_aa[Nn, Pn, Pn_now] = node_angle
                    node_path_path0_xx[Nn, Pn, Pn_now] = x1
                    node_path_path0_yy[Nn, Pn, Pn_now] = y1
                    node_path_path0_pp[Nn, Pn, Pn_now] = node_path_path0_pp[Nn_now, Pn_now, Pn_now_max] + prob
                ###
                Pn_now_max = np.argmax(node_path_path0_pp[Nn, Pn, :])
                node_path_path0max[Nn, Pn] = Pn_now_max
                ###
                node_path_aa[Nn, Pn] = node_path_path0_aa[Nn, Pn, Pn_now_max]
                node_path_xx[Nn, Pn] = node_path_path0_xx[Nn, Pn, Pn_now_max]
                node_path_yy[Nn, Pn] = node_path_path0_yy[Nn, Pn, Pn_now_max]
                node_path_pp[Nn, Pn] = node_path_path0_pp[Nn, Pn, Pn_now_max]
            ###
        ###
        ###
        for Nn in np.arange(cell.total_node - 1, 0, -1):
            Nn_now = Nn - 1
            if Nn == cell.total_node - 1:
                Pn_max = np.argmax(node_path_pp[Nn, :])
            else:
                Pn_max = Pn_max_Pn_now_max
            ###
            Pn_max_Pn_now_max = node_path_path0max[Nn, Pn_max]
            ###
            node_aa[Nn] = node_path_aa[Nn_now, Pn_max_Pn_now_max]
            node_xx[Nn] = node_path_xx[Nn_now, Pn_max_Pn_now_max]
            node_yy[Nn] = node_path_yy[Nn_now, Pn_max_Pn_now_max]
        ###
        return (node_aa, node_xx, node_yy)
    ######################################
    ###
    def chain_HMM_node(cell,
                       seed_xx,
                       seed_yy,
                       seed_aa = None, #seed_angle of the starting seed_path
                       seed_angle_max = None, #maximum angle_range between ending seed_path(-end, start, +end)
                       chain_level = None,
                      ):
        ###
        total_seed = len(seed_xx)
        ###
        if chain_level is None:
            chain_level = 1
        ###
        if (seed_aa is None):
            seed_aa = np.zeros(total_seed)
        ###
        if (seed_angle_max is None):
            seed_angle_max = 360 * (np.pi / 180)
        ### node +++
        seed_node_aa = np.zeros([total_seed, cell.total_node], dtype = np.float64)
        seed_node_xx = np.zeros([total_seed, cell.total_node], dtype = np.int64)
        seed_node_yy = np.zeros([total_seed, cell.total_node], dtype = np.int64)
        ### node ---
        ###
        real_chain_ii_list = []
        real_chain_aa_list = []
        real_chain_xx_list = []
        real_chain_yy_list = []
        ###
        #####################################
        for i in range(total_seed):
            ###
            seed_a = seed_aa[i]
            seed_x = seed_xx[i]
            seed_y = seed_yy[i]
            #############################
            node_HMM = cell.node_HMM_path(seed_x,
                                          seed_y,
                                          seed_angle = seed_a,
                                          seed_angle_max = seed_angle_max,
                                         )
            seed_node_aa[i], seed_node_xx[i], seed_node_yy[i] = node_HMM
            ###
            #############################
            ### node_chain_intensity +++
            high_node = []
            for Nn in range(cell.total_node):
                if Nn == 0:
                    Nn_A = Nn
                    Nn_B = Nn + 1
                    cut_level = 4 * chain_level
                else:
                    Nn_A = Nn - 1
                    Nn_B = Nn
                    cut_level = chain_level
                node_A = np.array([seed_node_xx[i, Nn_A], seed_node_yy[i, Nn_A]])
                node_B = np.array([seed_node_xx[i, Nn_B], seed_node_yy[i, Nn_B]])
                link_mean, zone_median = cell._node_link_intensity(node_A, node_B)
                if link_mean > cut_level * zone_median:
                     high_node.append(Nn)
            ### node_chain_intensity ---
            ### real_chain (continuous chain) +++
            if len(high_node) >= 3:
                real_chain = []
                j = high_node[0]
                real_chain.append(high_node[0])
                for k in high_node[1:]:
                    if k == j + 1:
                        real_chain.append(k)
                    j = j + 1
                ###
                real_chain_ii_list.append(real_chain)
                real_chain_aa_list.append(seed_node_aa[i])
                real_chain_xx_list.append(seed_node_xx[i])
                real_chain_yy_list.append(seed_node_yy[i])
            ### real_chain (continuous chain) +++
        ####
        real_chain_ii = np.array(real_chain_ii_list)
        real_chain_aa = np.array(real_chain_aa_list)
        real_chain_xx = np.array(real_chain_xx_list)
        real_chain_yy = np.array(real_chain_yy_list)
        ###
        return(real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy,
               seed_node_xx, seed_node_yy)
    ######################################################################
    ###
    def pair_HMM_chain(cell,
                       seed_xx,
                       seed_yy,
                       seed_aa = None, #seed_angle of the starting seed_path
                       seed_angle_max = None, #maximum angle_range between ending seed_path(-end, start, +end)
                       chain_level = None,
                      ):
        ### first chain +++
        chain_HMM = cell.chain_HMM_node(seed_xx,
                                        seed_yy,
                                        seed_aa = seed_aa,
                                        seed_angle_max = seed_angle_max,
                                        chain_level = chain_level,
                                       )
        ###
        real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
        ###########################################################################
        ### first chain ---
        pair_seed_aa = []
        pair_seed_xx = []
        pair_seed_yy = []
        for ri in range(real_chain_ii.shape[0]):
            chain_aa = real_chain_aa[ri][real_chain_ii[ri]]
            if len(chain_aa) >= 2:
                pair_seed_aa.append(chain_aa[1] + 180 * (np.pi / 180)) #opposite direction (180 degree difference)
                chain_xx = real_chain_xx[ri][real_chain_ii[ri]]
                pair_seed_xx.append(chain_xx[0])
                chain_yy = real_chain_yy[ri][real_chain_ii[ri]]
                pair_seed_yy.append(chain_yy[0])
        ### secondary chain +++
        ###
        seed_angle_max = 180 * (np.pi / 180) #only half of 360_degree
        ###
        pair_chain_HMM = cell.chain_HMM_node(pair_seed_xx,
                                             pair_seed_yy,
                                             seed_aa = pair_seed_aa,
                                             seed_angle_max = seed_angle_max,
                                             chain_level = chain_level,
                                            )
        ### secondary chain ---
        ###########################################################################
        return(chain_HMM,
               pair_chain_HMM,
               pair_seed_xx, pair_seed_yy)
    ######################################
    ###########################################################################
    ### processing chain_node from AlvaHmm
    ###########################################################################
    ### connecting node_point which has constant radial_distance between points)
    def connecting_point_by_pixel(cell,
                                  point_xx,
                                  point_yy,
                                  point_r = None,
                                 ):
        ###
        if point_r is None:
            point_r = cell.node_r
        ###
        pixel_line_xx = np.array([], dtype = np.int64)
        pixel_line_yy = np.array([], dtype = np.int64)
        for i in range(len(point_xx) - 1):
            dx = point_xx[i+1] - point_xx[i]
            dy = point_yy[i+1] - point_yy[i]
            if (dx == 0):
                # Handle vertical line (dx = 0)
                step_size = abs(dy) / point_r if dy != 0 else 1  # Avoid division by zero
                step_list = np.arange(0, dy, step_size) if dy != 0 else np.zeros(point_r)
                step_dy = point_yy[i] + np.array(step_list, dtype = np.int64)
                step_dx = point_xx[i] + np.zeros(len(step_list), dtype = np.int64)
            else:
                # Handle non-vertical line
                step_size = abs(dx) / point_r if dx != 0 else 1  # Avoid division by zero
                step_list = np.arange(0, dx, step_size) if dx != 0 else np.zeros(point_r)
                step_dx = point_xx[i] + np.array(step_list, dtype = np.int64)
                # Avoid division by zero when calculating slope
                step_dy = point_yy[i] + np.array((dy / dx if dx != 0 else 0) * step_list, dtype = np.int64)
            pixel_line_xx = np.append(pixel_line_xx, step_dx)
            pixel_line_yy = np.append(pixel_line_yy, step_dy)
        ###
        return (pixel_line_yy, pixel_line_xx)
        #####################################
    ###
    def chain_image(cell,
                    chain_HMM_1st,
                    pair_chain_HMM,):
        ###
        ### image_size +++
        total_pixel_y, total_pixel_x = cell.mmm.shape
        ### image_size ---
        ###
        ### connect_chain_image +++
        connect_chain_xx = np.array([], dtype = np.int64)
        connect_chain_yy = np.array([], dtype = np.int64)
        ### chain_HMM +++
        for chain_i in [0, 1]:
            chain_HMM = [chain_HMM_1st, pair_chain_HMM][chain_i]
            ###
            real_chain_ii, real_chain_aa, real_chain_xx, real_chain_yy = chain_HMM[0:4]
            seed_node_xx, seed_node_yy = chain_HMM[4:6]
        ### chain_HMM ---
            for i in range(len(real_chain_ii)):
                point_xx = np.array(real_chain_xx[i][real_chain_ii[i]], dtype = np.int64)
                point_yy = np.array(real_chain_yy[i][real_chain_ii[i]], dtype = np.int64)
                ###
                pixel_line_yy, pixel_line_xx = cell.connecting_point_by_pixel(point_xx, point_yy)
                connect_chain_xx = np.append(connect_chain_xx, pixel_line_xx)
                connect_chain_yy = np.append(connect_chain_yy, pixel_line_yy)
        ###
        chain_mmm_draft = np.zeros([total_pixel_y, total_pixel_x], dtype = np.int64)
        chain_mmm_draft[connect_chain_yy, connect_chain_xx] = 1
        ###
        ### dilation_skeletonize +++
        from skimage.morphology import dilation, disk, square, skeletonize
        mmmD = dilation(chain_mmm_draft, disk(cell.node_r / 2))
        bool_mmm = skeletonize(mmmD)
        ### dilation_skeletonize ---
        ### converting bool(True, False) into number(1, 0)
        chain_mmm_fine = np.int64(bool_mmm)
        return (chain_mmm_fine)
        ########################################
###########################################################################
'''#####################################################################'''
###########################################################################

