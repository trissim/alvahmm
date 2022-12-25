#!/usr/bin/env python
# coding: utf-8

# # Home-made machinery for general-analysis including:
# ### time_tracking
# ### pixel_line, grid_line
# ### PDF, PMF

# In[1]:


'''
author: Alvason Zhenhua Li
date:   from 10/21/2017 to 08/27/2020
Home-made machinery
'''
### 09/20/2020, updated sky_grid_x00_y00_label with pd_print option
### 08/27/2020, updated sky_grid_positionN_x00_y00_label function with '_' and right view
### 07/27/2020, adding sky_grid_positionN_x00_y00_label function for large_imaging_dataset
### 06/18/2019, updated AlvaSpace_or_LocalSpace (one_saving vs together_saving)
### 03/06/2019, adding AlvaSpace_or_LocalSpace function for large_dataset
### 12/14/2018, adding sky_grid_x00_y00_label function for large_imaging_dataset
### 09/28/2018, adding root_working_path function for convienience
### 03/04/2018, updated for iWatcher (beautifying the output by adding the label 'total_time')
### open_package +++
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time
import sys
import os
import logging
import warnings
logging.basicConfig(format = '%(asctime)s %(message)s',
                    level = logging.INFO, stream = sys.stdout)
if __name__ == '__main__': logging.info('(previous_run_time)')
### open_package ---


# ## sky_grid_x00_y00_label

# In[3]:


'''
author: Alvason Zhenhua Li
date:   from 10/21/2018 to 07/27/2020
Home-made machinery for stitching large imaging dataset
'''
############################################################
def pd_print(data, width = None, column = None, row = None):
    ## setting the display of pd.DataFrame
    if width is None: width = 100
    if column is None: column = 10 
    if row is None: row = 10
    pd.set_option('display.max_colwidth', width)
    pd.set_option('display.max_columns', column)
    pd.set_option('display.max_rows', row)
    import IPython.core.display as core
    pd_table = core.display(pd.DataFrame(data))
    return pd_table
############################################################
###
###################################################
def sky_grid_x00_y00_label(sky_size_x, sky_size_y, table_print = True):
    ###
    aaa = np.arange(sky_size_x * sky_size_y)
    bbb = aaa.reshape(sky_size_x, sky_size_y)
    ###########
    def _ooi(i):
        if i <= 9:
            ooi = '00{:}'.format(i)
        elif (i >= 10) and (i < 100):
            ooi = '0{:}'.format(i)
        return ooi
    ###########
    sky_grid_xy = bbb.astype(object)
    for xi in range(sky_size_x):
        for yi in range(sky_size_y):
            oxi = 'x' + _ooi(xi + 1)
            oyi = 'y' + _ooi(yi + 1)
            sky_grid_xy[xi, yi] = '{:}_{:}'.format(oxi, oyi)
    ###
    if table_print:
        pd_print(sky_grid_xy.T, row = 6, column = 10)
    ###
    return sky_grid_xy
###################################################
###
#############################################################
def sky_grid_positionN_x00_y00_label(sky_size_x, sky_size_y, table_print = True):
    ###
    aaa = np.arange(sky_size_x * sky_size_y)
    ###########
    def _ooi(i):
        if i <= 9:
            ooi = '00{:}'.format(i)
        elif (i >= 10) and (i < 100):
            ooi = '0{:}'.format(i)
        return ooi
    ###########
    #################
    ### positionN +++
    bbb = aaa.reshape(sky_size_x, sky_size_y)
    sky_grid_N = bbb.astype(object)
    N = 0
    for yi in range(sky_size_y):
        for xi in range(sky_size_x):
            sky_grid_N[xi, yi] = 'Position{:}_'.format(N + 1)
            N = N + 1
            ###
    positionN = sky_grid_N[::-1]
    if table_print:
        pd_print(positionN.T, row = 6, column = 10)
    ### positionN ---
    #################
    ###############
    ### x00_y00 +++
    sky_size_x__ = sky_size_x
    sky_size_y__ = sky_size_y
    ccc = aaa.reshape(sky_size_x__, sky_size_y__)
    sky_grid_xy__ = ccc.astype(object)
    for yi in range(sky_size_y__):
        for xi in range(sky_size_x__):
            oxi = 'x' + _ooi(xi + 1)
            oyi = 'y' + _ooi(yi + 1)
            sky_grid_xy__[xi, yi] = '{:}_{:}'.format(oxi, oyi)
    ###
    if table_print:
        pd_print(sky_grid_xy__.T, row = 6, column = 10)
    ###
    ### x00_y00 ---
    ###############
    ######################
    ### positionN_dict +++
    positionN_dict = {}
    for yi in range(sky_size_y__):
        for xi in range(sky_size_x__):
            positionN_dict[positionN[xi, yi]] = sky_grid_xy__[xi, yi]
    ###
    if table_print:
        print('positionN_dict =', positionN_dict)
    ### positionN_dict ---
    ######################
    return positionN, sky_grid_xy__, positionN_dict
#############################################################
if __name__ == '__main__':
    sky_size_x = 16
    sky_size_y = 19
    sky_grid_xy_label = sky_grid_x00_y00_label(sky_size_x, sky_size_y)
    ###
    sky_size_x = 6
    sky_size_y = 9
    positionN, sky_grid_xy, positionN_dict = sky_grid_positionN_x00_y00_label(sky_size_x, sky_size_y)


# ## timeWatcher

# In[5]:


'''
author: Alvason Zhenhua Li
date:   10/21/2017
Home-made machinery
'''
### time-watching and progress-bar +++
from IPython.core.display import clear_output
# import time
class timeWatcher(object):
    def __init__(cell):
        cell.start_time = time.time()
    ###
    def progressBar(cell, starting , current_step, stopping):
        progressing = float(current_step - starting) / (stopping - starting) 
        clear_output(wait = True) 
        current_time = time.time()
        print('[{:6.6f} second {:} {:}% {:}]'.format(current_time - cell.start_time
                                              , int(10 * progressing) * '--'
                                              , int(100 * progressing)
                                              , int(10 - 10 * progressing) * '++'))
        #sys.stdout.write('\n')
    ###
    def runTime(cell):
        current_time = time.time()
        total_time = current_time - cell.start_time
        print('[running time = {:6.6f} second]'.format(total_time))
        return total_time
### time-watching and progress-bar ---

if __name__ == '__main__':   
    timing = timeWatcher()
    for i in range(9):
        timing.progressBar(0, i, 8)
        time.sleep(0.3)
        print(i)
    timing.runTime()    


# ## iWatcher of looping

# In[6]:


'''
author: Alvason Zhenhua Li
date:   12/25/2017
Home-made machinery
'''
# import numpy as np
# import time
# import sys
from IPython.core.display import clear_output
###
class progressBar(object):
    def __init__(cell, total_i):
        cell.new_i = 0
        cell.max_i = total_i
        cell.start_time = time.time()
        cell.total_time = 0   
        cell.bar_width = 20

    def update(cell):
        cell.new_i += 1
        cell.total_time = time.time() - cell.start_time
        bar_time = ' run_time = ' + str(cell.total_time)
        progress = np.floor(round(cell.new_i / cell.max_i * 100, 2) / 100 * cell.bar_width)
        bar_type = '[{:}{:}]'.format('--' * int(progress),
                                     '++' * int(cell.bar_width - progress))
        sys.stdout.write('\r{:}{:}'.format(bar_type, bar_time))
        sys.stdout.flush()
        if cell.new_i >= cell.max_i:
            sys.stdout.write(', total_time = {:}'.format(cell.time_format(cell.total_time)))
            sys.stdout.write('\n')
            sys.stdout.flush() 

    def time_format(cell, _time):
        if (_time > 60):
            time_string = time.strftime('%H:%M:%S', time.gmtime(_time)) + ' (hour: minute: second)'
        else:
            time_string = str(_time) + ' (second)'
        return time_string
###      
def generator_on_the_fly(bar_class, clear_display = False):
    def bbb(loop_list):
        total_i = len(loop_list)
        if clear_display:
            clear_output(wait = True)
        progress_bar = bar_class(total_i)
        for i in loop_list:
            yield i
            progress_bar.update()
    return bbb
###
iWatcher = generator_on_the_fly(progressBar)
###
if __name__ == '__main__': 
    ###
    iWatcher = generator_on_the_fly(progressBar, clear_display = False)
    for k in range(9):
        for i in iWatcher(range(3)): 
            #print('\n', i)
            time.sleep(0.2)
    ###
    iWatcher = generator_on_the_fly(progressBar, clear_display = True)
    for k in range(9):
        for i in iWatcher(range(3)): 
            print('\n', i)
            time.sleep(0.2)

