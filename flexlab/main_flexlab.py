#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:03:22 2019

@author: jasperpakshong
"""

'''

Main for running simulations - used to:
    # set ORT config & act scales by reading in excel sheet from sw_mat folder
    # initialize run SPBC?
    # initialize run LPBC?
    # send commands to load racks?
    # ...
    
'''
# In[imports]:

# import...
import pandas as pd
import numpy as np

from ORT_modbus_energise import *
'''
functions from ORT_modbus_energise:
    
    read_sw_mat(filepath)
        ~ read in the sw matrix excel file that has switch configs for each test case on that feeder
        ~ define relative folder in pathname and filename in filename
    
    sim_start_stop(test_ID,sim_time_min)
        ~ sends switch values of associated test_ID [i.e. 'T1.1']
        ~ along with a simulation flag signal of 1 to begin recording data
        ~ at end of sim_time_min [minutes], sends simulation flag of 0 to stop recording data
        
    sim_stop()
        ~ sends simulation flag signal of 0
        
'''

# In[settings]:

# settings...

test_ID = 'T1.1' # which test case you want to run
sim_time_min = 1 # amount of time to record for in minutes

#should input option for choosing feeder such that the correct excel sheet is selected
#currently only set up for 13NF_bal
# feeder =
# Choose from [13NF_bal, 13NF_unbal]
pathname = 'sw_mat/'
filename = 'HIL_switch_matrix_13NF_bal.xlsx'
filepath = pathname+filename


















