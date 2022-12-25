#!/usr/bin/env python
# coding: utf-8

# # Home-made machinery for T-cell analysis including Circle-packing algorithm

# In[1]:


'''
author: Alvason Zhenhua Li
date:   01/21/2016

Home-made machinery
'''
### open_package +++
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import sys
import os
import datetime
### open_package ---
### local_package +++
import alva_machinery.tool.aBox as alva_tool
iWatcher = alva_tool.iWatcher
### local_package ---
###
if __name__ == '__main__':
    previous_running_time = datetime.datetime.now()
    print ('Previous running time is {:}'.format(previous_running_time))


# In[2]:


'''
author: Alvason Zhenhua Li
date:   01/21/2016

Home-made machinery for circle-packing 
'''
AlvaColorCycle = ['blue', 'green', 'cyan'
                  , 'lightpink', 'magenta', 'yellow'
                  , 'purple', 'black', 'deepskyblue'
                  , 'gray', 'red', 'lime']

class AlvaCirclePacking(object):
    def __init__(cell, data_area, asteroid_size = None, moon_size = None, belt_width = None, **kwargs):
        if asteroid_size is None:
            asteroid_size = 1
        cell.asteroid_size = asteroid_size
        if moon_size is None:
            moon_size = 1
        cell.moon_size = moon_size
        if belt_width is None:
            if cell.asteroid_size == 0 or cell.moon_size == 1:
                belt_width = 0.05
            else:
                belt_width = 0.15
        cell.belt_width = belt_width
        # raw data
        cell.raw_area = data_area
        cell.raw_radii = (cell.raw_area / np.pi)**(1.0/2)
        #Max-min sorting-index of all-object
        cell.raw_radii_index = np.argsort(cell.raw_radii)[::-1]
        cell.sort_radii = cell.raw_radii[cell.raw_radii_index]
        cell.total_galaxy_raw = len(cell.raw_area)
        # initialzing a memory space
        cell.polar_angle = np.zeros(cell.total_galaxy_raw)
        cell.polar_distance = np.zeros(cell.total_galaxy_raw)
        # galaxy_object
        cell.object_area = cell.raw_area[cell.raw_area > 0]
        cell.object_radii = (cell.object_area / np.pi)**(1.0/2)
        cell.total_galaxy_object = len(cell.object_area) 
        # setting galaxy_area = 4 * total_circle_area
        cell.galaxy_diameter = 2 * (4 * cell.object_area.sum() / np.pi)**(1.0/2) 
        ### 1. initializing a random-distribution of all-object within the galaxy-area
        for rn in range(cell.total_galaxy_object): 
            cell.polar_angle[rn] = np.random.random() * (2 * np.pi) 
            cell.polar_distance[rn] = abs((cell.galaxy_diameter / 2) * np.random.random() 
                                          - cell.sort_radii[rn])
        ### 2. selecting big-circle as core (planet) 
        cell.total_coreNum = int(0)
        cell.total_galaxy_asteroid = int(0)
        cell.total_galaxy_moon = int(0)
        for rn in range(cell.total_galaxy_object):
            if np.around(np.pi * (cell.sort_radii[rn])**2) <= cell.asteroid_size:
                # building a asteroid belt around at the boundary of galaxy
                cell.polar_distance[rn] = abs((cell.galaxy_diameter / 2) * 
                                              (1 - cell.belt_width * np.random.random()) 
                                              + cell.sort_radii[rn]) 
                cell.total_galaxy_asteroid = cell.total_galaxy_asteroid + 1
            elif np.around(np.pi * (cell.sort_radii[rn])**2) >= cell.moon_size:
                # counting total core
                cell.total_coreNum = cell.total_coreNum + 1
                cell.polar_distance[rn] = abs((cell.galaxy_diameter / 2) 
                                              * (1 - 2*cell.belt_width) * np.random.random()
                                              - cell.sort_radii[rn])
            else:
                # building moon-belt around at the boundary of galaxy
                cell.polar_distance[rn] = abs((cell.galaxy_diameter / 2) 
                                              * ((1 - cell.belt_width) - 2*cell.belt_width * np.random.random()) 
                                              - cell.sort_radii[rn])
                cell.total_galaxy_moon = cell.total_galaxy_moon + 1
        # setting the top-cicle at the center of galaxy  
        cell.polar_distance[0] = 0.0
        cell.total_galaxy_planet = cell.total_coreNum
        cell.total_galaxy_star = cell.total_galaxy_object - cell.total_galaxy_asteroid
    #######################################################
    ### packing by out-core and fit-core events
    def out_fit_core(cell):
        for core in iWatcher(range(cell.total_coreNum)):
            print('out---core', core)
            # moving inside-circle out of the core
            cell.out_core(core)
            # fitting core-neighbor
            cell.fit_core(core)
        return (cell.polar_angle, cell.polar_distance)
    ###------------------------
    def out_core(cell, core, far_out = False, polar_angle = None, polar_distance = None, raw_radii = None):
        if polar_angle is None:
            polar_angle = cell.polar_angle[0:cell.total_galaxy_star]
        if polar_distance is None:
            polar_distance = cell.polar_distance[0:cell.total_galaxy_star]
        if raw_radii is None:
            raw_radii = cell.sort_radii[0:cell.total_galaxy_star]
        total_circle = len(polar_angle)
        ## moving the whole galaxy (all circles) so that core-circle is located on the-origin
        # the easy way to getting the relative displacement is in XY-unit
        center_X = polarXY(polar_angle, polar_distance)[0]
        center_Y = polarXY(polar_angle, polar_distance)[1]
        relative_x = center_X[core]
        relative_y = center_Y[core]
        center_X = center_X - relative_x
        center_Y = center_Y - relative_y
        ## moving inside-circle out of the core
        # the easy way to detecting inside-circle is in Polar-unit 
        polar_angle = xyPolar(center_X, center_Y)[0]
        polar_distance = xyPolar(center_X, center_Y)[1]
        record_inside_rn = []
        for rn in range(core + 1, total_circle):
            # detecting inside-circle
            if (polar_distance[rn] < raw_radii[core] + raw_radii[rn]):
                # moving to outside of the core
                polar_distance[rn] = (raw_radii[core] + raw_radii[rn])
                if far_out == True:
                    record_inside_rn.append(rn)
        ## moving the newly updated galaxy so that core-cricle is returned to previous-location  
        center_X = polarXY(polar_angle, polar_distance)[0]
        center_Y = polarXY(polar_angle, polar_distance)[1]
        center_X = center_X + relative_x
        center_Y = center_Y + relative_y
        polar_angle = xyPolar(center_X, center_Y)[0]
        polar_distance = xyPolar(center_X, center_Y)[1] 
        ## out_distance
        for rn in record_inside_rn:
            polar_angle[rn] = np.random.random() * (2 * np.pi)
            polar_distance[rn] = abs((cell.galaxy_diameter / 2) 
                                     * ((1 - cell.belt_width) - cell.belt_width * np.random.random()) 
                                     - raw_radii[rn])
            # in case of no asteroid and moon
            if cell.asteroid_size == 0 or cell.moon_size == 1:
                polar_distance[rn] = abs((cell.galaxy_diameter / 2) 
                                         * (1 - 2*cell.belt_width * np.random.random()) 
                                         - raw_radii[rn])                
        ## returning
        cell.polar_angle[0:cell.total_galaxy_star] = polar_angle
        cell.polar_distance[0:cell.total_galaxy_star] = polar_distance
        return (cell.polar_angle, cell.polar_distance)
    ###-------------------
    def fit_core(cell, core, polar_angle = None, polar_distance = None, raw_radii = None):
        if polar_angle is None:
            polar_angle = cell.polar_angle[0:cell.total_galaxy_star]
        if polar_distance is None:
            polar_distance = cell.polar_distance[0:cell.total_galaxy_star]
        if raw_radii is None:
            raw_radii = cell.sort_radii[0:cell.total_galaxy_star]
        total_circle = len(polar_angle)
        ## moving the whole galaxy (all circles) so that core-circle is located on the-origin
        # the easy way to getting the relative displacement is in XY-unit
        center_X = polarXY(polar_angle, polar_distance)[0]
        center_Y = polarXY(polar_angle, polar_distance)[1]
        relative_x = center_X[core]
        relative_y = center_Y[core]
        center_X = center_X - relative_x
        center_Y = center_Y - relative_y
        # the easy way to detect circle is in Polar-unit 
        polar_angle = xyPolar(center_X, center_Y)[0]
        polar_distance = xyPolar(center_X, center_Y)[1]    
        # detecting and recording current neighbor of the core
        nnnCore = []
        for rn in range(total_circle):
            # detecting
            if (raw_radii[core] + raw_radii[rn]*1.01) >             (polar_distance[rn]) >             (raw_radii[core] + raw_radii[rn]*0.99):
                # recording
                nnnCore.append(rn)
        total_neighbor = len(nnnCore)
        # fitting current neighbor without overlapping
        for i in iWatcher(range(total_neighbor)):
            # previous cores are fixed
            if (nnnCore[i] > core):
                # try to fit
                total_gear_teeth = 360
                teeth = 0
                # detecting overlap
                overlap = cell._overlap_detection(nnnCore[i], nnnCore
                                                  , polar_angle = polar_angle, polar_distance = polar_distance)
                while ((teeth < total_gear_teeth) and (overlap == True)):
                    # fit-in-the-loop
                    polar_angle[nnnCore[i]] = polar_angle[nnnCore[i]] + float(teeth)*(2*np.pi)/total_gear_teeth
                    teeth = teeth + 1
                    overlap = cell._overlap_detection(nnnCore[i], nnnCore
                                                      , polar_angle = polar_angle, polar_distance = polar_distance)
                # out-of-the-loop
                if (overlap == True):
                    polar_angle[nnnCore[i]] = polar_angle[core + 1]
                    polar_distance[nnnCore[i]] = polar_distance[core + 1]     
            # testing
            #polar_angle[nnnCore[i]] = i*(2*np.pi)/total_neighbor
        ## moving the newly updated galaxy so that core-cricle is returned to previous-location  
        center_X = polarXY(polar_angle, polar_distance)[0]
        center_Y = polarXY(polar_angle, polar_distance)[1]
        center_X = center_X + relative_x
        center_Y = center_Y + relative_y
        polar_angle = xyPolar(center_X, center_Y)[0]
        polar_distance = xyPolar(center_X, center_Y)[1] 
        ## returning
        cell.polar_angle[0:cell.total_galaxy_star] = polar_angle
        cell.polar_distance[0:cell.total_galaxy_star] = polar_distance
        return (cell.polar_angle, cell.polar_distance)

    ###------------------------
    # overlap-detection is only for local neighborhood of the core
    def _overlap_detection(cell, detected_rn, nnnCore, polar_angle = None, polar_distance = None, raw_radii = None):
        if polar_angle is None:
            polar_angle = cell.polar_angle[0:cell.total_galaxy_star]
        if polar_distance is None:
            polar_distance = cell.polar_distance[0:cell.total_galaxy_star]
        if raw_radii is None:
            raw_radii = cell.sort_radii[0:cell.total_galaxy_star]
        overlap = False
        total_neighbor = len(nnnCore)  
        polar_A = np.zeros(total_neighbor)
        polar_D = np.zeros(total_neighbor) 
        ortho_X = np.zeros(total_neighbor)
        ortho_Y = np.zeros(total_neighbor)   
        for i in range(total_neighbor):
            polar_A[i] = np.copy(polar_angle[nnnCore[i]])
            polar_D[i] = np.copy(polar_distance[nnnCore[i]])
        # the easy way to getting the relative displacement is in XY-unit
        ortho_X = polarXY(polar_A, polar_D)[0]
        ortho_Y = polarXY(polar_A, polar_D)[1]
        relative_x0 = polar_to_xy(polar_angle[detected_rn], polar_distance[detected_rn])[0]
        relative_y0 = polar_to_xy(polar_angle[detected_rn], polar_distance[detected_rn])[1]
        ortho_X = ortho_X - relative_x0
        ortho_Y = ortho_Y - relative_y0
        # the easy way to detect circle is in Polar-unit 
        polar_A = xyPolar(ortho_X, ortho_Y)[0]
        polar_D = xyPolar(ortho_X, ortho_Y)[1] 
        # detecting
        # overlap_time > 1 for avoiding self-detected
        overlap_time = 0
        for i in range(total_neighbor):
            if (polar_D[i] < (raw_radii[detected_rn] + raw_radii[nnnCore[i]])):
                overlap_time = overlap_time + 1
        if (overlap_time > 1):
            overlap = True    
        return (overlap) 
    ###------------------------
    # patching circle by original-index of raw_radii (sort_radii) so that colorings are consistent
    # patching circle with coloring
    def patchingCircle(cell, center_X = None, center_Y = None, raw_radii = None
                       , AlvaColorCycle = AlvaColorCycle, color_alpha = None, line_width = None):
        if line_width is None:
            line_width = 0.01
        if color_alpha is None:
            color_alpha = 0.9
        if center_X is None and center_Y is None:
            aaa = polarXY(cell.polar_angle, cell.polar_distance)
            center_X = aaa[0]
            center_Y = aaa[1]
        if raw_radii is None:
            raw_radii = cell.sort_radii
        sort_radii = cell.sort_radii
        raw_radii_index = cell.raw_radii_index
        # patching circle by original-index of raw_radii so that coloring are consistent
        raw_radii_index_index = np.argsort(raw_radii_index) 
        circle_patch = []
        for xn, yn, rn in zip(center_X[raw_radii_index_index]
                              , center_Y[raw_radii_index_index]
                              , sort_radii[raw_radii_index_index]):
            circle = mpl.patches.Circle((xn, yn), rn)
            circle_patch.append(circle) 
        # coloring 
        color_cycle = mpl.colors.ListedColormap(AlvaColorCycle, 'indexed')
        pCircle = mpl.collections.PatchCollection(circle_patch, cmap = color_cycle
                                                  , alpha = color_alpha, linewidths = line_width)
        pCircle.set_array(np.arange(len(AlvaColorCycle))*0.1)
        return (pCircle)

#####################################################################
# Polar to XY -------------------------------------------------------
def polar_to_xy(angle, distance):
    x = distance*np.cos(angle)
    y = distance*np.sin(angle)
    return (x, y)
#
def polarXY(polar_A, polar_D):
    total_point = len(polar_A)
    ortho_X = np.zeros(total_point)
    ortho_Y = np.zeros(total_point)
    for rn in range(total_point):
        runA = polar_to_xy(polar_A[rn], polar_D[rn])
        ortho_X[rn] = runA[0]
        ortho_Y[rn] = runA[1]
    return (ortho_X, ortho_Y)
##############-------------------------------------------------------
# XY to Polar -------------------------------------------------------
def xy_to_polar(x, y):
    if (x == 0.0 and y == 0.0):
        distance = 0.0
        angle = 0.0
    else: 
        distance = (x**2 + y**2)**(1.0/2)
        angle = np.arccos(x/distance)
    if (y < 0.0): 
        angle = -angle
    return (angle, distance)
#
def xyPolar(ortho_X, ortho_Y):
    total_point = len(ortho_X)
    polar_A = np.zeros(total_point)
    polar_D = np.zeros(total_point)
    for rn in range(total_point):
        runAA = xy_to_polar(ortho_X[rn], ortho_Y[rn])
        polar_A[rn] = runAA[0]
        polar_D[rn] = runAA[1]
    return (polar_A, polar_D)
#############-------------------------------------------------------
def ab_angle(core, neighbor_n):
    if (neighbor_n - 1) == core: neighbor_n = core
    a = raw_radii[core] + raw_radii[neighbor_n - 1]
    b = raw_radii[core] + raw_radii[neighbor_n]
    c = raw_radii[neighbor_n - 1] + raw_radii[neighbor_n]
    angle = np.arccos((a**2 + b**2 - c**2)/(2*a*b))
    return (angle)
##################################################################################################################

