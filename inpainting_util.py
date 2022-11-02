# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 18:33:42 2022

@author: carndt
"""

import numpy as np


def extract_visible_edges(img, start_angle, stop_angle, threshold=0.1):
    g1 = img[2:,1:-1] - img[:-2,1:-1]
    g2 = img[1:-1,2:] - img[1:-1,:-2]

    magn = np.sqrt(g1**2 + g2**2)
    edges = np.zeros([2,magn.shape[0],magn.shape[1]])
    
    #magn[magn<threshold]=0
    magn_bool = magn>=threshold*magn.max()
    g1_bool = (g1==0)
    g2_bool = (g2==0)
    
    angle = np.zeros(magn.shape, dtype=float)
    
    angle[magn_bool*g1_bool] = np.pi + np.sign(g2[magn_bool*g1_bool])*np.pi/2
    angle[magn_bool*g2_bool] = (1-np.sign(g1[magn_bool*g2_bool]))*np.pi/2
    
    rest = magn_bool*(~g1_bool)*(~g2_bool)
    angle[rest] = np.arctan(g2[rest]/g1[rest])
    angle[rest*(g1<0)] += np.pi
    angle[rest*(g1>0)*(g2<0)] += 2*np.pi
    
    angle = angle*180/np.pi
    
    angle_plus = (angle+90)%360
    angle_minus = (angle-90)%360
    
    angle_bool_1 = angle_plus >= start_angle
    angle_bool_1 *= angle_plus <= stop_angle
    
    angle_bool_2 = angle_minus >= start_angle
    angle_bool_2 *= angle_minus <= stop_angle
    angle_bool = (~((~angle_bool_1)*(~angle_bool_2)))*magn_bool
    
    edges[0][angle_bool] = g1[angle_bool]
    edges[1][angle_bool] = g2[angle_bool]
                
    return edges