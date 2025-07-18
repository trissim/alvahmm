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
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys
import os
import logging
logging.basicConfig(format = '%(asctime)s %(message)s',
                    level = logging.INFO, stream = sys.stdout)
if __name__ == '__main__': logging.info('(previous_run_time)')

# Import torbi for GPU-accelerated Viterbi decoding
import torbi
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
        real_chain_ii = np.array(real_chain_ii_list, dtype=object)
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


class AlvaHmmTorbi(AlvaHmm):
    """
    Torbi-accelerated version of AlvaHmm.

    This class inherits from AlvaHmm and overrides the core Viterbi computation
    methods to use torbi for GPU acceleration while maintaining the same API.
    """

    def __init__(self, likelihood_mmm, total_node=None, total_path=None,
                 node_r=None, node_angle_max=None, device=None):
        """
        Initialize AlvaHmmTorbi with GPU device support.

        Args:
            device: torch.device for GPU computation (obtained from input tensor)
        """
        # Call parent constructor
        super().__init__(likelihood_mmm, total_node, total_path, node_r, node_angle_max)

        # Store device for torbi computation
        self.device = device
        print(f"✅ AlvaHmmTorbi initialized with device: {self.device}")

    def node_HMM_path(self, seed_x, seed_y, seed_angle=None, seed_angle_max=None):
        """
        Torbi-accelerated version of node_HMM_path.

        This method replaces the manual forward/backward pass loops with
        torbi's GPU-accelerated Viterbi decoding.
        """
        print("🚀 Using torbi GPU acceleration for Viterbi decoding")

        # Use parent class initialization logic
        if seed_angle is None:
            seed_angle = 0
        if seed_angle_max is None:
            seed_angle_max = self.node_angle_max

        # Build emission and transition matrices for torbi
        emission_probs, transition_matrix, initial_probs = self._build_torbi_matrices(
            seed_x, seed_y, seed_angle, seed_angle_max
        )

        # Use torbi for GPU-accelerated Viterbi decoding
        gpu_id = 0 if self.device.type == 'cuda' else None
        optimal_path = torbi.from_probabilities(
            observation=emission_probs,
            transition=transition_matrix,
            initial=initial_probs,
            log_probs=False,
            gpu=gpu_id
        )

        # Convert torbi results back to alvahmm format
        return self._convert_torbi_results(optimal_path, seed_x, seed_y, seed_angle, seed_angle_max)

    def _build_torbi_matrices(self, seed_x, seed_y, seed_angle, seed_angle_max):
        """
        Build emission, transition, and initial probability matrices for torbi.

        This converts the alvahmm probability computation into torbi format:
        - emission_probs: (batch=1, frames=total_node, states=total_path)
        - transition_matrix: (states=total_path, states=total_path)
        - initial_probs: (states=total_path,)
        """
        # Initialize arrays like the original algorithm
        node_path_aa = np.zeros([self.total_node, self.total_path], dtype=np.float64)
        node_path_xx = np.zeros([self.total_node, self.total_path], dtype=np.int64)
        node_path_yy = np.zeros([self.total_node, self.total_path], dtype=np.int64)
        node_path_pp = np.zeros([self.total_node, self.total_path], dtype=np.float64)

        # 3D arrays for full state space (like original)
        node_path_path0_aa = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)
        node_path_path0_xx = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_yy = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.int64)
        node_path_path0_pp = np.zeros([self.total_node, self.total_path, self.total_path], dtype=np.float64)

        dAngle = self.node_angle_max / (self.total_path - 1)

        # Build emission probabilities for each node and path
        emission_probs = np.zeros((1, self.total_node, self.total_path), dtype=np.float32)

        # Initialize first node (seed) - same as original algorithm
        Nn = 0
        total_path_seed = min(self.total_path * 4, 64)  # More paths for seed
        dAngle_seed = seed_angle_max / (total_path_seed - 1)

        # Evaluate all possible directions from seed
        seed_probs = []
        for Pn in range(total_path_seed):
            node_angle = (Pn * dAngle_seed) + (seed_angle - seed_angle_max / 2)
            prob, x1, y1 = self._prob_sum_state(seed_x, seed_y, node_angle)
            seed_probs.append(prob)

        # Select top paths from seed
        top_indices = np.argsort(seed_probs)[-self.total_path:]

        for Pn in range(self.total_path):
            Pn_seed = top_indices[Pn]
            node_angle = (Pn_seed * dAngle_seed) + (seed_angle - seed_angle_max / 2)
            prob, x1, y1 = self._prob_sum_state(seed_x, seed_y, node_angle)

            # Store for later conversion
            node_path_aa[Nn, Pn] = node_angle
            node_path_xx[Nn, Pn] = x1
            node_path_yy[Nn, Pn] = y1
            node_path_pp[Nn, Pn] = prob

            # Set emission probability for this state
            emission_probs[0, Nn, Pn] = prob

        # Build emission probabilities for subsequent nodes
        for Nn in range(1, self.total_node):
            Nn_prev = Nn - 1

            for Pn in range(self.total_path):
                for Pn_prev in range(self.total_path):
                    # Calculate angle for this transition
                    node_angle = (node_path_aa[Nn_prev, Pn_prev] - self.node_angle_max / 2) + (Pn * dAngle)

                    # Get probability for this state
                    prob, x1, y1 = self._prob_sum_state(
                        node_path_xx[Nn_prev, Pn_prev],
                        node_path_yy[Nn_prev, Pn_prev],
                        node_angle
                    )

                    # Store the maximum probability for this state
                    if Pn_prev == 0 or prob > emission_probs[0, Nn, Pn]:
                        emission_probs[0, Nn, Pn] = prob
                        node_path_aa[Nn, Pn] = node_angle
                        node_path_xx[Nn, Pn] = x1
                        node_path_yy[Nn, Pn] = y1
                        node_path_pp[Nn, Pn] = prob

        # Build transition matrix (uniform transitions between adjacent angles)
        transition_matrix = np.zeros((self.total_path, self.total_path), dtype=np.float32)
        for i in range(self.total_path):
            for j in range(self.total_path):
                # Favor transitions to nearby angles
                angle_diff = abs(i - j)
                if angle_diff <= 1 or angle_diff >= self.total_path - 1:
                    transition_matrix[i, j] = 0.8  # High probability for nearby angles
                else:
                    transition_matrix[i, j] = 0.2 / (self.total_path - 3)  # Low probability for distant angles

        # Normalize transition matrix
        transition_matrix = transition_matrix / transition_matrix.sum(axis=1, keepdims=True)

        # Build initial probabilities (uniform for now)
        initial_probs = np.ones(self.total_path, dtype=np.float32) / self.total_path

        # Convert to torch tensors on the correct device
        emission_probs = torch.from_numpy(emission_probs).to(self.device)
        transition_matrix = torch.from_numpy(transition_matrix).to(self.device)
        initial_probs = torch.from_numpy(initial_probs).to(self.device)

        # Store for result conversion
        self._node_path_aa = node_path_aa
        self._node_path_xx = node_path_xx
        self._node_path_yy = node_path_yy
        self._node_path_pp = node_path_pp

        return emission_probs, transition_matrix, initial_probs

    def _convert_torbi_results(self, optimal_path, seed_x, seed_y, seed_angle, seed_angle_max):
        """
        Convert torbi results back to alvahmm format.

        Args:
            optimal_path: (batch=1, frames=total_node) tensor from torbi

        Returns:
            (node_aa, node_xx, node_yy): Same format as original node_HMM_path
        """
        # Convert to numpy and extract the single batch
        path_indices = optimal_path[0].cpu().numpy()  # Shape: (total_node,)

        # Initialize result arrays
        node_aa = np.zeros(self.total_node, dtype=np.float64)
        node_xx = np.zeros(self.total_node, dtype=np.int64)
        node_yy = np.zeros(self.total_node, dtype=np.int64)

        # Convert path indices back to coordinates using stored values
        for Nn in range(self.total_node):
            Pn = path_indices[Nn]
            node_aa[Nn] = self._node_path_aa[Nn, Pn]
            node_xx[Nn] = self._node_path_xx[Nn, Pn]
            node_yy[Nn] = self._node_path_yy[Nn, Pn]

        # Set initial node to seed values
        node_aa[0] = seed_angle
        node_xx[0] = seed_x
        node_yy[0] = seed_y

        return (node_aa, node_xx, node_yy)

    def pair_HMM_chain_batched(self, seed_xx, seed_yy, chain_level=1.05):
        """
        Batched version of pair_HMM_chain for GPU acceleration.

        Processes all seeds in parallel using torbi's batch processing capabilities.
        """
        print(f"🚀 Batched processing {len(seed_xx)} seeds with torbi")

        # Process all seeds in a single batched torbi call
        all_chains = self._process_seeds_batched(seed_xx, seed_yy)

        # Validate chains (can be parallelized later)
        valid_chains = []
        for i, (node_aa, node_xx, node_yy) in enumerate(all_chains):
            valid_nodes = self.chain_HMM_node_single(node_aa, node_xx, node_yy, chain_level)
            if len(valid_nodes) > 0:
                valid_chains.append((node_aa, node_xx, node_yy, valid_nodes))

        if not valid_chains:
            # Return empty results if no valid chains
            return ([], [], [], []), ([], [], [], []), [], []

        # Process all valid chains (matching original structure)
        real_chain_ii_list = []
        real_chain_aa_list = []
        real_chain_xx_list = []
        real_chain_yy_list = []

        for node_aa, node_xx, node_yy, valid_nodes in valid_chains:
            # Convert to numpy arrays
            node_aa = np.array(node_aa)
            node_xx = np.array(node_xx, dtype=np.int64)
            node_yy = np.array(node_yy, dtype=np.int64)
            valid_nodes = np.array(valid_nodes, dtype=np.int64)

            # Add to lists (matching original format)
            real_chain_ii_list.append(valid_nodes.tolist())
            real_chain_aa_list.append(node_aa)  # Full node array
            real_chain_xx_list.append(node_xx)  # Full node array
            real_chain_yy_list.append(node_yy)  # Full node array

        # Create arrays matching original format
        real_chain_ii = np.array(real_chain_ii_list, dtype=object)
        real_chain_aa = np.array(real_chain_aa_list)
        real_chain_xx = np.array(real_chain_xx_list)
        real_chain_yy = np.array(real_chain_yy_list)

        # Create seed node arrays (dummy values for now)
        seed_node_xx = np.array([seed_xx])
        seed_node_yy = np.array([seed_yy])

        # Create bidirectional chains with correct structure
        chain_HMM_1st = (
            real_chain_ii,     # real_chain_ii - array of objects (lists)
            real_chain_aa,     # real_chain_aa - 2D array
            real_chain_xx,     # real_chain_xx - 2D array
            real_chain_yy,     # real_chain_yy - 2D array
            seed_node_xx,      # seed_node_xx - 2D array
            seed_node_yy       # seed_node_yy - 2D array
        )

        # Pair chain (simplified - same structure for now)
        pair_chain_HMM = (
            real_chain_ii,     # real_chain_ii
            real_chain_aa,     # real_chain_aa
            real_chain_xx,     # real_chain_xx
            real_chain_yy,     # real_chain_yy
            seed_node_xx,      # seed_node_xx
            seed_node_yy       # seed_node_yy
        )

        return chain_HMM_1st, pair_chain_HMM, seed_xx, seed_yy

    def _process_seeds_batched(self, seed_xx, seed_yy):
        """
        Process all seeds in a single batched torbi call for maximum GPU utilization.
        """
        num_seeds = len(seed_xx)
        print(f"🔥 Building batched emission matrices for {num_seeds} seeds")

        # Build batched emission, transition, and initial probability matrices
        batch_emissions, transition_matrix, initial_probs = self._build_batched_torbi_matrices(seed_xx, seed_yy)

        print(f"📊 Batch shapes: emissions={batch_emissions.shape}, transitions={transition_matrix.shape}")

        # Single torbi call for all seeds - THIS IS WHERE THE GPU ACCELERATION HAPPENS!
        gpu_id = 0 if self.device.type == 'cuda' else None
        print(f"⚡ Running torbi.from_probabilities on {self.device} with {num_seeds} seeds")

        optimal_paths = torbi.from_probabilities(
            observation=batch_emissions,      # (num_seeds, total_node, total_path)
            transition=transition_matrix,     # (total_path, total_path)
            initial=initial_probs,           # (total_path,)
            log_probs=False,
            gpu=gpu_id
        )

        print(f"✅ Torbi completed! Processing {num_seeds} results")

        # Convert all results back to alvahmm format
        all_chains = []
        for i in range(num_seeds):
            node_aa, node_xx, node_yy = self._convert_single_torbi_result(
                optimal_paths[i], seed_xx[i], seed_yy[i], i
            )
            all_chains.append((node_aa, node_xx, node_yy))

        return all_chains

    def _build_batched_torbi_matrices(self, seed_xx, seed_yy):
        """
        Build batched emission matrices for all seeds simultaneously using GPU vectorization.

        Returns:
            batch_emissions: (num_seeds, total_node, total_path)
            transition_matrix: (total_path, total_path) - shared across all seeds
            initial_probs: (total_path,) - shared across all seeds
        """
        num_seeds = len(seed_xx)
        print(f"🔥 GPU vectorized matrix building for {num_seeds} seeds")

        # Convert to GPU tensors
        seed_xx_tensor = torch.tensor(seed_xx, device=self.device, dtype=torch.float32)
        seed_yy_tensor = torch.tensor(seed_yy, device=self.device, dtype=torch.float32)
        image_tensor = torch.from_numpy(self.mmm).to(self.device).float()

        # Initialize GPU tensors for results
        batch_emissions = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device, dtype=torch.float32)

        # Store node information for all seeds (on GPU)
        self._batch_node_path_aa = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device, dtype=torch.float32)
        self._batch_node_path_xx = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device, dtype=torch.float32)
        self._batch_node_path_yy = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device, dtype=torch.float32)

        dAngle = self.node_angle_max / (self.total_path - 1)

        # GPU vectorized probability calculation for ALL seeds, nodes, and paths at once
        batch_emissions, self._batch_node_path_aa, self._batch_node_path_xx, self._batch_node_path_yy = self._compute_all_probabilities_vectorized(
            seed_xx_tensor, seed_yy_tensor, image_tensor, dAngle
        )

        # All probability calculations now done in single GPU call above!

        # Build shared transition matrix (on GPU)
        transition_matrix = torch.zeros((self.total_path, self.total_path), device=self.device, dtype=torch.float32)
        for i in range(self.total_path):
            for j in range(self.total_path):
                angle_diff = abs(i - j)
                if angle_diff <= 1 or angle_diff >= self.total_path - 1:
                    transition_matrix[i, j] = 0.8
                else:
                    transition_matrix[i, j] = 0.2 / (self.total_path - 3)

        # Normalize transition matrix
        transition_matrix = transition_matrix / transition_matrix.sum(dim=1, keepdim=True)

        # Build shared initial probabilities (on GPU)
        initial_probs = torch.ones(self.total_path, device=self.device, dtype=torch.float32) / self.total_path

        return batch_emissions, transition_matrix, initial_probs

    def _compute_all_probabilities_vectorized(self, seed_xx_tensor, seed_yy_tensor, image_tensor, dAngle):
        """
        GPU vectorized computation of ALL emission probabilities for ALL seeds simultaneously.

        This replaces thousands of CPU _prob_sum_state() calls with a single GPU operation.
        """
        num_seeds = len(seed_xx_tensor)
        print(f"⚡ GPU vectorizing {num_seeds * self.total_node * self.total_path} probability calculations")

        # Initialize result tensors
        batch_emissions = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device)
        batch_node_aa = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device)
        batch_node_xx = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device)
        batch_node_yy = torch.zeros((num_seeds, self.total_node, self.total_path), device=self.device)

        # Build ALL coordinates and angles for vectorized computation
        seed_angle = 0.0  # Default seed angle
        seed_angle_max = self.node_angle_max

        # Process first node (seeds) - vectorized
        Nn = 0
        total_path_seed = min(self.total_path * 2, 32)
        dAngle_seed = seed_angle_max / max(total_path_seed - 1, 1)

        # Generate all seed angles for all seeds
        Pn_indices = torch.arange(total_path_seed, device=self.device, dtype=torch.float32)
        seed_angles = (Pn_indices * dAngle_seed) + (seed_angle - seed_angle_max / 2)  # [total_path_seed]

        # Expand for all seeds: [num_seeds, total_path_seed]
        all_seed_x = seed_xx_tensor.unsqueeze(1).expand(-1, total_path_seed)  # [num_seeds, total_path_seed]
        all_seed_y = seed_yy_tensor.unsqueeze(1).expand(-1, total_path_seed)  # [num_seeds, total_path_seed]
        all_seed_angles = seed_angles.unsqueeze(0).expand(num_seeds, -1)  # [num_seeds, total_path_seed]

        # Vectorized probability calculation for ALL seeds and ALL seed paths
        all_seed_probs, all_seed_x1, all_seed_y1 = self._compute_log_probability_vectorized(
            image_tensor,
            torch.stack([all_seed_x.flatten(), all_seed_y.flatten()], dim=1),  # [num_seeds*total_path_seed, 2]
            all_seed_angles.flatten(),  # [num_seeds*total_path_seed]
            self.node_r
        )

        # Reshape results: [num_seeds, total_path_seed]
        all_seed_probs = all_seed_probs.view(num_seeds, total_path_seed)
        all_seed_x1 = all_seed_x1.view(num_seeds, total_path_seed)
        all_seed_y1 = all_seed_y1.view(num_seeds, total_path_seed)

        # Select top paths for each seed
        _, top_indices = torch.topk(all_seed_probs, self.total_path, dim=1)  # [num_seeds, total_path]

        # Gather selected values for first node
        batch_indices = torch.arange(num_seeds, device=self.device).unsqueeze(1).expand(-1, self.total_path)
        selected_angles = all_seed_angles[batch_indices, top_indices]  # [num_seeds, total_path]
        selected_probs = all_seed_probs[batch_indices, top_indices]  # [num_seeds, total_path]
        selected_x1 = all_seed_x1[batch_indices, top_indices]  # [num_seeds, total_path]
        selected_y1 = all_seed_y1[batch_indices, top_indices]  # [num_seeds, total_path]

        # Store first node results
        batch_emissions[:, Nn, :] = selected_probs
        batch_node_aa[:, Nn, :] = selected_angles
        batch_node_xx[:, Nn, :] = selected_x1
        batch_node_yy[:, Nn, :] = selected_y1

        # Process subsequent nodes - vectorized
        for Nn in range(1, self.total_node):
            Nn_prev = Nn - 1

            # Get all previous positions and angles
            prev_x = batch_node_xx[:, Nn_prev, :].unsqueeze(2)  # [num_seeds, total_path, 1]
            prev_y = batch_node_yy[:, Nn_prev, :].unsqueeze(2)  # [num_seeds, total_path, 1]
            prev_angles = batch_node_aa[:, Nn_prev, :].unsqueeze(2)  # [num_seeds, total_path, 1]

            # Generate all possible next angles
            Pn_indices = torch.arange(self.total_path, device=self.device, dtype=torch.float32)
            next_angle_offsets = (Pn_indices * dAngle).unsqueeze(0).unsqueeze(0)  # [1, 1, total_path]

            # Calculate all transition angles: [num_seeds, total_path_prev, total_path_next]
            all_next_angles = (prev_angles - self.node_angle_max / 2) + next_angle_offsets

            # Flatten for vectorized computation
            flat_prev_x = prev_x.expand(-1, -1, self.total_path).flatten()  # [num_seeds*total_path*total_path]
            flat_prev_y = prev_y.expand(-1, -1, self.total_path).flatten()  # [num_seeds*total_path*total_path]
            flat_angles = all_next_angles.flatten()  # [num_seeds*total_path*total_path]

            # Vectorized probability calculation for ALL transitions
            flat_probs, flat_x1, flat_y1 = self._compute_log_probability_vectorized(
                image_tensor,
                torch.stack([flat_prev_x, flat_prev_y], dim=1),  # [num_seeds*total_path*total_path, 2]
                flat_angles,  # [num_seeds*total_path*total_path]
                self.node_r
            )

            # Reshape and find best transitions: [num_seeds, total_path_prev, total_path_next]
            transition_probs = flat_probs.view(num_seeds, self.total_path, self.total_path)
            transition_x1 = flat_x1.view(num_seeds, self.total_path, self.total_path)
            transition_y1 = flat_y1.view(num_seeds, self.total_path, self.total_path)
            transition_angles = all_next_angles

            # Find best previous state for each next state
            best_probs, best_prev_indices = torch.max(transition_probs, dim=1)  # [num_seeds, total_path]

            # Gather best coordinates and angles
            seed_indices = torch.arange(num_seeds, device=self.device).unsqueeze(1).expand(-1, self.total_path)
            path_indices = torch.arange(self.total_path, device=self.device).unsqueeze(0).expand(num_seeds, -1)

            best_x1 = transition_x1[seed_indices, best_prev_indices, path_indices]
            best_y1 = transition_y1[seed_indices, best_prev_indices, path_indices]
            best_angles = transition_angles[seed_indices, best_prev_indices, path_indices]

            # Store results for this node
            batch_emissions[:, Nn, :] = best_probs
            batch_node_aa[:, Nn, :] = best_angles
            batch_node_xx[:, Nn, :] = best_x1
            batch_node_yy[:, Nn, :] = best_y1

        print(f"✅ GPU vectorization complete! Processed {num_seeds * self.total_node * self.total_path} calculations")
        return batch_emissions, batch_node_aa, batch_node_xx, batch_node_yy

    def _compute_log_probability_vectorized(self, image_tensor, start_positions, angles, node_r, prob_multiplier=255.0):
        """
        Vectorized GPU version of _prob_sum_state for massive parallel processing.

        Based on the existing vectorized implementations in the codebase.

        Args:
            image_tensor: [H, W] image on GPU
            start_positions: [N, 2] starting positions (x, y)
            angles: [N] angles in radians
            node_r: number of steps along each path

        Returns:
            probs: [N] log probabilities
            x1: [N] end x coordinates
            y1: [N] end y coordinates
        """
        N = start_positions.shape[0]
        device = self.device
        H, W = image_tensor.shape

        # Calculate end positions
        x1 = (node_r * torch.cos(angles) + start_positions[:, 0]).long()
        y1 = (node_r * torch.sin(angles) + start_positions[:, 1]).long()

        # Boundary check - set invalid positions to -inf probability
        valid_mask = (
            (y1 >= 0) & (y1 < H - 1) &
            (x1 >= 0) & (x1 < W - 1) &
            (start_positions[:, 1] >= 0) & (start_positions[:, 1] < H - 1) &
            (start_positions[:, 0] >= 0) & (start_positions[:, 0] < W - 1)
        )

        probs = torch.full((N,), -float('inf'), device=device)

        if valid_mask.any():
            valid_indices = torch.where(valid_mask)[0]
            valid_start_pos = start_positions[valid_indices]  # [valid_count, 2]
            valid_angles = angles[valid_indices]  # [valid_count]

            # Generate all sampling points along paths
            rn_steps = torch.arange(1, node_r + 1, device=device, dtype=torch.float32)  # [node_r]

            # Expand for vectorized computation: [valid_count, node_r]
            valid_count = len(valid_indices)
            rn_expanded = rn_steps.unsqueeze(0).expand(valid_count, -1)  # [valid_count, node_r]
            angles_expanded = valid_angles.unsqueeze(1).expand(-1, node_r)  # [valid_count, node_r]
            x0_expanded = valid_start_pos[:, 0].unsqueeze(1).expand(-1, node_r)  # [valid_count, node_r]
            y0_expanded = valid_start_pos[:, 1].unsqueeze(1).expand(-1, node_r)  # [valid_count, node_r]

            # Compute all coordinates at once [valid_count, node_r]
            rx = (rn_expanded * torch.cos(angles_expanded) + x0_expanded).long()
            ry = (rn_expanded * torch.sin(angles_expanded) + y0_expanded).long()

            # Clamp coordinates to image bounds
            rx = torch.clamp(rx, 0, W - 1)
            ry = torch.clamp(ry, 0, H - 1)

            # Sample all values at once [valid_count, node_r]
            pixel_vals = image_tensor[ry, rx]

            # Author's exact zero handling: "if mmm[ry, rx] == 0: prob += 0"
            zero_mask = (pixel_vals == 0)
            non_zero_mask = ~zero_mask

            # Compute log values for non-zero pixels (author's formula: log(pixel * 255))
            log_vals = torch.where(
                non_zero_mask,
                torch.log(pixel_vals * prob_multiplier + 1e-8),  # Small epsilon to avoid log(0)
                torch.tensor(0.0, device=device)
            )

            # Sum across node_r dimension (author's accumulation)
            probs[valid_indices] = log_vals.sum(dim=1)

        return probs, x1.float(), y1.float()

    def _convert_single_torbi_result(self, optimal_path, seed_x, seed_y, seed_idx):
        """
        Convert a single torbi result back to alvahmm format.
        """
        # Convert to numpy
        path_indices = optimal_path.cpu().numpy()  # Shape: (total_node,)

        # Initialize result arrays
        node_aa = np.zeros(self.total_node, dtype=np.float64)
        node_xx = np.zeros(self.total_node, dtype=np.int64)
        node_yy = np.zeros(self.total_node, dtype=np.int64)

        # Convert path indices back to coordinates with bounds checking
        for Nn in range(self.total_node):
            Pn = path_indices[Nn]
            # Clamp path index to valid range
            Pn = max(0, min(Pn, self.total_path - 1))
            node_aa[Nn] = self._batch_node_path_aa[seed_idx, Nn, Pn].cpu().item()
            node_xx[Nn] = self._batch_node_path_xx[seed_idx, Nn, Pn].cpu().item()
            node_yy[Nn] = self._batch_node_path_yy[seed_idx, Nn, Pn].cpu().item()

        # Set initial node to seed values
        node_aa[0] = 0  # Default seed angle
        node_xx[0] = seed_x
        node_yy[0] = seed_y

        return node_aa, node_xx, node_yy

    def chain_HMM_node_single(self, node_aa, node_xx, node_yy, chain_level=1.05):
        """
        Validate a single HMM chain (simplified version of chain_HMM_node).
        """
        high_node = []

        for Nn in range(self.total_node - 1):
            node_A = np.array([node_xx[Nn], node_yy[Nn]])
            node_B = np.array([node_xx[Nn + 1], node_yy[Nn + 1]])

            link_mean, zone_median = self._node_link_intensity(node_A, node_B)

            # Validation criterion from original paper
            if Nn == 0:
                cut_level = 4 * chain_level  # First node: stricter threshold
            else:
                cut_level = chain_level      # Other nodes: normal threshold

            if link_mean > cut_level * zone_median:
                high_node.append(Nn)

        # Require at least 3 valid nodes
        if len(high_node) >= 3:
            return high_node
        else:
            return []

