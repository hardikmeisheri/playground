#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 12:31:29 2019

@author: hardik
"""

from pommerman import agents
import tensorflow as tf
import pickle
from collections import deque
import numpy as np
import action_prune

def decrypt_mess(mess):
    
    approx_team_mate = 0
    approx_enm1 = 0
    approx_enm2 = 0
    
    if ((mess[0] == 1) & (mess[1] == 1)):
        approx_team_mate == 1
        approx_enm1 = 0
        approx_enm2 = 0
    elif ((mess[0] == 1) & (mess[1] == 2)):
        approx_team_mate == 1
        approx_enm1 = 0
        approx_enm2 = 1
    elif ((mess[0] == 1) & (mess[1] == 3)):
        approx_team_mate == 1
        approx_enm1 = 0
        approx_enm2 = 2
    elif ((mess[0] == 1) & (mess[1] == 4)):
        approx_team_mate == 1
        approx_enm1 = 0
        approx_enm2 = 3
    elif ((mess[0] == 1) & (mess[1] == 5)):
        approx_team_mate == 1
        approx_enm1 = 0
        approx_enm2 = 4
    elif ((mess[0] == 1) & (mess[1] == 6)):
        approx_team_mate == 1
        approx_enm1 = 1
        approx_enm2 = 0
    elif ((mess[0] == 1) & (mess[1] == 7)):
        approx_team_mate == 1
        approx_enm1 = 1
        approx_enm2 = 1
    elif ((mess[0] == 1) & (mess[1] == 8)):
        approx_team_mate == 1
        approx_enm1 = 1
        approx_enm2 = 2
    elif ((mess[0] == 2) & (mess[1] == 1)):
        approx_team_mate == 1
        approx_enm1 = 1
        approx_enm2 = 3
    elif ((mess[0] == 2) & (mess[1] == 2)):
        approx_team_mate == 1
        approx_enm1 = 1
        approx_enm2 = 4
    elif ((mess[0] == 2) & (mess[1] == 3)):
        approx_team_mate == 1
        approx_enm1 = 2
        approx_enm2 = 0
    elif ((mess[0] == 2) & (mess[1] == 4)):
        approx_team_mate == 1
        approx_enm1 = 2
        approx_enm2 = 1
    elif ((mess[0] == 2) & (mess[1] == 5)):
        approx_team_mate == 1
        approx_enm1 = 2
        approx_enm2 = 2
    elif ((mess[0] == 2) & (mess[1] == 6)):
        approx_team_mate == 1
        approx_enm1 = 2
        approx_enm2 = 3
    elif ((mess[0] == 2) & (mess[1] == 7)):
        approx_team_mate == 1
        approx_enm1 = 2
        approx_enm2 = 4
    elif ((mess[0] == 2) & (mess[1] == 8)):
        approx_team_mate == 1
        approx_enm1 = 3
        approx_enm2 = 0
    elif ((mess[0] == 3) & (mess[1] == 1)):
        approx_team_mate == 1
        approx_enm1 = 3
        approx_enm2 = 1
    elif ((mess[0] == 3) & (mess[1] == 2)):
        approx_team_mate == 1
        approx_enm1 = 3
        approx_enm2 = 2
    elif ((mess[0] == 3) & (mess[1] == 3)):
        approx_team_mate == 1
        approx_enm1 = 3
        approx_enm2 = 3
    elif ((mess[0] == 3) & (mess[1] == 4)):
        approx_team_mate == 1
        approx_enm1 = 3
        approx_enm2 = 4
    elif ((mess[0] == 3) & (mess[1] == 5)):
        approx_team_mate == 1
        approx_enm1 = 4
        approx_enm2 = 0
    elif ((mess[0] == 3) & (mess[1] == 6)):
        approx_team_mate == 1
        approx_enm1 = 4
        approx_enm2 = 1
    elif ((mess[0] == 3) & (mess[1] == 7)):
        approx_team_mate == 1
        approx_enm1 = 4
        approx_enm2 = 2
    elif ((mess[0] == 3) & (mess[1] == 8)):
        approx_team_mate == 1
        approx_enm1 = 4
        approx_enm2 = 3
    elif ((mess[0] == 4) & (mess[1] == 1)):
        approx_team_mate == 1
        approx_enm1 = 4
        approx_enm2 = 4
    elif ((mess[0] == 4) & (mess[1] == 2)):
        approx_team_mate == 2
        approx_enm1 = 0
        approx_enm2 = 0
    elif ((mess[0] == 4) & (mess[1] == 3)):
        approx_team_mate == 2
        approx_enm1 = 0
        approx_enm2 = 1
    elif ((mess[0] == 4) & (mess[1] == 4)):
        approx_team_mate == 2
        approx_enm1 = 0
        approx_enm2 = 2
    elif ((mess[0] == 4) & (mess[1] == 5)):
        approx_team_mate == 2
        approx_enm1 = 0
        approx_enm2 = 3
    elif ((mess[0] == 4) & (mess[1] == 6)):
        approx_team_mate == 2
        approx_enm1 = 0
        approx_enm2 = 4
    elif ((mess[0] == 4) & (mess[1] == 7)):
        approx_team_mate == 2
        approx_enm1 = 1
        approx_enm2 = 0
    elif ((mess[0] == 4) & (mess[1] == 8)):
        approx_team_mate == 2
        approx_enm1 = 1
        approx_enm2 = 1
    elif ((mess[0] == 5) & (mess[1] == 1)):
        approx_team_mate == 2
        approx_enm1 = 1
        approx_enm2 = 2
    elif ((mess[0] == 5) & (mess[1] == 2)):
        approx_team_mate == 2
        approx_enm1 = 1
        approx_enm2 = 3
    elif ((mess[0] == 5) & (mess[1] == 3)):
        approx_team_mate == 2
        approx_enm1 = 1
        approx_enm2 = 4
    elif ((mess[0] == 5) & (mess[1] == 4)):
        approx_team_mate == 2
        approx_enm1 = 2
        approx_enm2 = 0
    elif ((mess[0] == 5) & (mess[1] == 5)):
        approx_team_mate == 2
        approx_enm1 = 2
        approx_enm2 = 1
    elif ((mess[0] == 5) & (mess[1] == 6)):
        approx_team_mate == 2
        approx_enm1 = 2
        approx_enm2 = 2
    elif ((mess[0] == 5) & (mess[1] == 7)):
        approx_team_mate == 2
        approx_enm1 = 2
        approx_enm2 = 3
    elif ((mess[0] == 5) & (mess[1] == 8)):
        approx_team_mate == 2
        approx_enm1 = 2
        approx_enm2 = 4
    elif ((mess[0] == 6) & (mess[1] == 1)):
        approx_team_mate == 2
        approx_enm1 = 3
        approx_enm2 = 0
    elif ((mess[0] == 6) & (mess[1] == 2)):    
        approx_team_mate == 2
        approx_enm1 = 3
        approx_enm2 = 1
    elif ((mess[0] == 6) & (mess[1] == 3)):
        approx_team_mate == 2
        approx_enm1 = 3
        approx_enm2 = 2
    elif ((mess[0] == 6) & (mess[1] == 4)):
        approx_team_mate == 2
        approx_enm1 = 3
        approx_enm2 = 3
    elif ((mess[0] == 6) & (mess[1] == 5)):
        approx_team_mate == 2
        approx_enm1 = 3
        approx_enm2 = 4
    elif ((mess[0] == 6) & (mess[1] == 6)):
        approx_team_mate == 2
        approx_enm1 = 4
        approx_enm2 = 0
    elif ((mess[0] == 6) & (mess[1] == 7)):
        approx_team_mate == 2
        approx_enm1 = 4
        approx_enm2 = 1
    elif ((mess[0] == 6) & (mess[1] == 8)):
        approx_team_mate == 2
        approx_enm1 = 4
        approx_enm2 = 2
    elif ((mess[0] == 7) & (mess[1] == 1)):
        approx_team_mate == 2
        approx_enm1 = 4
        approx_enm2 = 3
    elif ((mess[0] == 7) & (mess[1] == 2)):
        approx_team_mate == 2
        approx_enm1 = 4
        approx_enm2 = 4

    return approx_team_mate, approx_enm1, approx_enm2

def featurize(obs):

    board = obs['board']
    mess = obs['message']
    approx_team_mate, approx_enm1, approx_enm2 = decrypt_mess(mess)

    board_maps = np.stack([board == i for i in range(10)], axis=2).astype(np.float32)
    bomb_blast_strength = (obs["bomb_blast_strength"].astype(np.float32)/11)
    bomb_life = (obs["bomb_life"].astype(np.float32)/10)
    
    own_position_plane = np.zeros_like(board)
    own_position_plane[obs["position"]] = 1.0
    
    team_mate_position_plane = np.zeros_like(board)
    team_mate_alive_mat = np.zeros_like(board)
    
    team_mate_id = obs['teammate'].value
    
    if team_mate_id in obs['alive']:
        team_mate_alive_mat += 1
        if np.where(board==team_mate_id)[0].shape[0]>0:
            team_mate_position_plane[np.where(board==team_mate_id)[0]] = 1
        else:
            if approx_team_mate == 1:
                team_mate_position_plane[:, 0:5] = 1
            elif approx_team_mate == 2:
                team_mate_position_plane[:, 5:] = 1
    
    enemy1_position_plane = np.zeros_like(board)
    enemy1_alive_mat = np.zeros_like(board)
    
    enemy1_id = obs['enemies'][0].value
    
    if enemy1_id in obs['alive']:
        enemy1_alive_mat += 1
        if np.where(board==enemy1_id)[0].shape[0]>0:
            enemy1_position_plane[np.where(board==enemy1_id)[0]] = 1
        else:
            if approx_enm1 == 1:
                enemy1_position_plane[0:5, 0:5] = 1
            elif approx_enm1 == 2:
                enemy1_position_plane[0:5, 5:] = 1
            elif approx_enm1 == 3:
                enemy1_position_plane[5:, 1:5] = 1
            elif approx_enm1 == 4:
                enemy1_position_plane[5:, 5:] = 1
    
    enemy2_position_plane = np.zeros_like(board)
    enemy2_alive_mat = np.zeros_like(board)
    enemy2_id = obs['enemies'][1].value
    
    if enemy2_id in obs['alive']:
        enemy2_alive_mat += 1
        if np.where(board==enemy2_id)[0].shape[0]>0:
            enemy2_position_plane[np.where(board==enemy2_id)[0]] = 1
        else:
            if approx_enm2 == 1:
                enemy2_position_plane[0:5, 0:5] = 1
            elif approx_enm2 == 2:
                enemy2_position_plane[0:5, 5:] = 1
            elif approx_enm2 == 3:
                enemy2_position_plane[5:, 1:5] = 1
            elif approx_enm2 == 4:
                enemy2_position_plane[5:, 5:] = 1
    
    
    
    time_step_plane = np.zeros_like(board).astype('float64')
    time_step_plane += (obs['step_count']/800)
    
    blast_strength = np.zeros_like(board).astype('float64')
    blast_strength += (obs['blast_strength']/11)
    
    ammo = np.zeros_like(board).astype('float64')
    ammo += (obs['ammo']/10)
    
    kick_power = np.zeros_like(board).astype('float64')
    kick_power += int(obs['ammo'])
    
    new_obs_cust = np.concatenate([board_maps, bomb_blast_strength.reshape((11, 11, 1)), bomb_life.reshape((11, 11, 1)), \
                             own_position_plane.reshape((11, 11, 1)), team_mate_position_plane.reshape((11, 11, 1)), \
                             team_mate_alive_mat.reshape((11, 11, 1)), enemy1_position_plane.reshape((11, 11, 1)), \
                             enemy1_alive_mat.reshape((11, 11, 1)), enemy2_position_plane.reshape((11, 11, 1)), \
                             enemy2_alive_mat.reshape((11, 11, 1)), blast_strength.reshape((11, 11, 1)), \
                             ammo.reshape((11, 11, 1)), kick_power.reshape((11, 11, 1)), time_step_plane.reshape((11, 11, 1))], axis=-1)
    
    return new_obs_cust


def generate_mess(obs):

    message = np.zeros((1, 2))
    i =0
    board = obs['board']
    own_position = obs["position"]
    enemy1_id = obs['enemies'][0].value
    enemy2_id = obs['enemies'][1].value
    
    enem1_known = 0
    enem2_known = 0
    
    if np.where(board==enemy1_id)[0].shape[0]>0:
        enemy_1_pos = np.where(board==enemy1_id)
        enem1_known = 1
    
    if np.where(board==enemy2_id)[0].shape[0]>0:
        enemy_2_pos = np.where(board==enemy2_id)
        enem2_known = 1
    
    if own_position[1] < 5:
        approx_own_pos = 1
    else:
        approx_own_pos = 2
        
    if enem1_known == 1:
        
        if enemy_1_pos[0][0] < 5:
            if enemy_1_pos[1][0] < 5:
                approx_enm1 = 1
            else:
                approx_enm1 = 2
        else:
            if enemy_1_pos[1][0] < 5:
                approx_enm1 = 3
            else:
                approx_enm1 = 4
    else:
        approx_enm1 = 0
    
    if enem2_known == 1:
        
        
        if enemy_2_pos[0][0] < 5:
            if enemy_2_pos[1][0] < 5:
                approx_enm2 = 1
            else:
                approx_enm2 = 2
        else:
            if enemy_2_pos[1][0] < 5:
                approx_enm2 = 3
            else:
                approx_enm2 = 4
    else:
        approx_enm2 = 0


    if ((approx_own_pos ==1) & (approx_enm1 ==0) & (approx_enm2==0)):
        message[i, 0] = 1
        message[i, 1] = 1
    elif ((approx_own_pos ==1) & (approx_enm1 ==0) & (approx_enm2==1)):
        message[i, 0] = 1
        message[i, 1] = 2
    elif ((approx_own_pos ==1) & (approx_enm1 ==0) & (approx_enm2==2)):
        message[i, 0] = 1
        message[i, 1] = 3
    elif ((approx_own_pos ==1) & (approx_enm1 ==0) & (approx_enm2==3)):
        message[i, 0] = 1
        message[i, 1] = 4
    elif ((approx_own_pos ==1) & (approx_enm1 ==0) & (approx_enm2==4)):
        message[i, 0] = 1
        message[i, 1] = 5
    elif ((approx_own_pos ==1) & (approx_enm1 ==1) & (approx_enm2==0)):
        message[i, 0] = 1
        message[i, 1] = 6
    elif ((approx_own_pos ==1) & (approx_enm1 ==1) & (approx_enm2==1)):
        message[i, 0] = 1
        message[i, 1] = 7
    elif ((approx_own_pos ==1) & (approx_enm1 ==1) & (approx_enm2==2)):
        message[i, 0] = 1
        message[i, 1] = 8
    elif ((approx_own_pos ==1) & (approx_enm1 ==1) & (approx_enm2==3)):
        message[i, 0] = 2
        message[i, 1] = 1
    elif ((approx_own_pos ==1) & (approx_enm1 ==1) & (approx_enm2==4)):
        message[i, 0] = 2
        message[i, 1] = 2
    elif ((approx_own_pos ==1) & (approx_enm1 ==2) & (approx_enm2==0)):
        message[i, 0] = 2
        message[i, 1] = 3
    elif ((approx_own_pos ==1) & (approx_enm1 ==2) & (approx_enm2==1)):
        message[i, 0] = 2
        message[i, 1] = 4
    elif ((approx_own_pos ==1) & (approx_enm1 ==2) & (approx_enm2==2)):
        message[i, 0] = 2
        message[i, 1] = 5
    elif ((approx_own_pos ==1) & (approx_enm1 ==2) & (approx_enm2==3)):
        message[i, 0] = 2
        message[i, 1] = 6
    elif ((approx_own_pos ==1) & (approx_enm1 ==2) & (approx_enm2==4)):
        message[i, 0] = 2
        message[i, 1] = 7
    elif ((approx_own_pos ==1) & (approx_enm1 ==3) & (approx_enm2==0)):
        message[i, 0] = 2
        message[i, 1] = 8
    elif ((approx_own_pos ==1) & (approx_enm1 ==3) & (approx_enm2==1)):
        message[i, 0] = 3
        message[i, 1] = 1
    elif ((approx_own_pos ==1) & (approx_enm1 ==3) & (approx_enm2==2)):
        message[i, 0] = 3
        message[i, 1] = 2
    elif ((approx_own_pos ==1) & (approx_enm1 ==3) & (approx_enm2==3)):
        message[i, 0] = 3
        message[i, 1] = 3
    elif ((approx_own_pos ==1) & (approx_enm1 ==3) & (approx_enm2==4)):
        message[i, 0] = 3
        message[i, 1] = 4
    elif ((approx_own_pos ==1) & (approx_enm1 ==4) & (approx_enm2==0)):
        message[i, 0] = 3
        message[i, 1] = 5
    elif ((approx_own_pos ==1) & (approx_enm1 ==4) & (approx_enm2==1)):
        message[i, 0] = 3
        message[i, 1] = 6
    elif ((approx_own_pos ==1) & (approx_enm1 ==4) & (approx_enm2==2)):
        message[i, 0] = 3
        message[i, 1] = 7
    elif ((approx_own_pos ==1) & (approx_enm1 ==4) & (approx_enm2==3)):
        message[i, 0] = 3
        message[i, 1] = 8
    elif ((approx_own_pos ==1) & (approx_enm1 ==4) & (approx_enm2==4)):
        message[i, 0] = 4
        message[i, 1] = 1
    elif ((approx_own_pos ==2) & (approx_enm1 ==0) & (approx_enm2==0)):
        message[i, 0] = 4
        message[i, 1] = 2
    elif ((approx_own_pos ==2) & (approx_enm1 ==0) & (approx_enm2==1)):
        message[i, 0] = 4
        message[i, 1] = 3
    elif ((approx_own_pos ==2) & (approx_enm1 ==0) & (approx_enm2==2)):
        message[i, 0] = 4
        message[i, 1] = 4
    elif ((approx_own_pos ==2) & (approx_enm1 ==0) & (approx_enm2==3)):
        message[i, 0] = 4
        message[i, 1] = 5
    elif ((approx_own_pos ==2) & (approx_enm1 ==0) & (approx_enm2==4)):
        message[i, 0] = 4
        message[i, 1] = 6
    elif ((approx_own_pos ==2) & (approx_enm1 ==1) & (approx_enm2==0)):
        message[i, 0] = 4
        message[i, 1] = 7
    elif ((approx_own_pos ==2) & (approx_enm1 ==1) & (approx_enm2==1)):
        message[i, 0] = 4
        message[i, 1] = 8
    elif ((approx_own_pos ==2) & (approx_enm1 ==1) & (approx_enm2==2)):
        message[i, 0] = 5
        message[i, 1] = 1
    elif ((approx_own_pos ==2) & (approx_enm1 ==1) & (approx_enm2==3)):
        message[i, 0] = 5
        message[i, 1] = 2
    elif ((approx_own_pos ==2) & (approx_enm1 ==1) & (approx_enm2==4)):
        message[i, 0] = 5
        message[i, 1] = 3
    elif ((approx_own_pos ==2) & (approx_enm1 ==2) & (approx_enm2==0)):
        message[i, 0] = 5
        message[i, 1] = 4
    elif ((approx_own_pos ==2) & (approx_enm1 ==2) & (approx_enm2==1)):
        message[i, 0] = 5
        message[i, 1] = 5
    elif ((approx_own_pos ==2) & (approx_enm1 ==2) & (approx_enm2==2)):
        message[i, 0] = 5
        message[i, 1] = 6
    elif ((approx_own_pos ==2) & (approx_enm1 ==2) & (approx_enm2==3)):
        message[i, 0] = 5
        message[i, 1] = 7
    elif ((approx_own_pos ==2) & (approx_enm1 ==2) & (approx_enm2==4)):
        message[i, 0] = 5
        message[i, 1] = 8
    elif ((approx_own_pos ==2) & (approx_enm1 ==3) & (approx_enm2==0)):
        message[i, 0] = 6
        message[i, 1] = 1
    elif ((approx_own_pos ==2) & (approx_enm1 ==3) & (approx_enm2==1)):
        message[i, 0] = 6
        message[i, 1] = 2
    elif ((approx_own_pos ==2) & (approx_enm1 ==3) & (approx_enm2==2)):
        message[i, 0] = 6
        message[i, 1] = 3
    elif ((approx_own_pos ==2) & (approx_enm1 ==3) & (approx_enm2==3)):
        message[i, 0] = 6
        message[i, 1] = 4
    elif ((approx_own_pos ==2) & (approx_enm1 ==3) & (approx_enm2==4)):
        message[i, 0] = 6
        message[i, 1] = 5
    elif ((approx_own_pos ==2) & (approx_enm1 ==4) & (approx_enm2==0)):
        message[i, 0] = 6
        message[i, 1] = 6
    elif ((approx_own_pos ==2) & (approx_enm1 ==4) & (approx_enm2==1)):
        message[i, 0] = 6
        message[i, 1] = 7
    elif ((approx_own_pos ==2) & (approx_enm1 ==4) & (approx_enm2==2)):
        message[i, 0] = 6
        message[i, 1] = 8
    elif ((approx_own_pos ==2) & (approx_enm1 ==4) & (approx_enm2==3)):
        message[i, 0] = 7
        message[i, 1] = 1
    elif ((approx_own_pos ==2) & (approx_enm1 ==4) & (approx_enm2==4)):
        message[i, 0] = 7
        message[i, 1] = 2
    
    return message


def featurize_retrospective(trajectory):

    op_mat = np.zeros((11, 11, 10))
            


    board_list = [obs['board'] for obs in trajectory]
    base_board = board_list[-1]

    if (len(board_list)-1) > 0:
        
        for board_ind in reversed(range(len(board_list)-1)):
            visible_board_idx = np.where((board_list[board_ind] != 5) )
            possible_updation_idx = np.where(base_board[visible_board_idx] == 5)[0]
            
            if possible_updation_idx.shape[0]>0:
                for repl_ind in range(possible_updation_idx.shape[0]):
                    if board_list[board_ind][visible_board_idx[0][possible_updation_idx[repl_ind]], visible_board_idx[1][possible_updation_idx[repl_ind]]] not in [10, 11, 12, 13]:
                        base_board[visible_board_idx[0][possible_updation_idx[repl_ind]], visible_board_idx[1][possible_updation_idx[repl_ind]]]=board_list[board_ind][visible_board_idx[0][possible_updation_idx[repl_ind]], visible_board_idx[1][possible_updation_idx[repl_ind]]]
                    else:
                        temp_value = board_list[board_ind][visible_board_idx[0][possible_updation_idx[repl_ind]], visible_board_idx[1][possible_updation_idx[repl_ind]]]
                        if not np.where(base_board == temp_value)[0].shape[0] > 0:
                            base_board[visible_board_idx[0][possible_updation_idx[repl_ind]], visible_board_idx[1][possible_updation_idx[repl_ind]]] = temp_value




    op_mat = np.stack([base_board == i for i in range(10)], axis=2).astype(np.float32)
    
    return op_mat


class RL_Agent(agents.BaseAgent):
    def __init__(self, sess):
        
        
        self.obs_ph = tf.placeholder(dtype = tf.float32, shape = [None, 11,11,33], name = 'obs_ph')
        self.action_ph = tf.placeholder(dtype = tf.int32, shape = [None], name = 'actions')
        self.look_back = 5
        self.retro_threshold = 50
        self.obs_buffer = deque(maxlen = self.retro_threshold)
        
        with tf.variable_scope('policy_net'):
            cnn1 = tf.layers.conv2d(inputs = self.obs_ph, filters = 128, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            cnn2 = tf.layers.conv2d(inputs = cnn1, filters = 128, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            max_pool1 = tf.layers.max_pooling2d(inputs = cnn2, pool_size = 2, strides = 2)
            dropout_1 = tf.layers.dropout(inputs = max_pool1, rate = 0.3, training = self.policy_dropout)
    
            cnn3 = tf.layers.conv2d(inputs = dropout_1, filters = 64, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            cnn4 = tf.layers.conv2d(inputs = cnn3, filters = 64, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            max_pool2 = tf.layers.max_pooling2d(inputs = cnn4, pool_size = 2, strides = 2)
            dropout_2 = tf.layers.dropout(inputs = max_pool2, rate = 0.3, training = self.policy_dropout)
    
            flatten_1 = tf.layers.flatten(inputs = dropout_2)
            
            
            d1 = tf.layers.dense(inputs = flatten_1, units = 50, activation = tf.nn.relu)
            d1_3 = tf.layers.dropout(inputs = d1, rate = 0.2, training = self.policy_dropout)        
            d2 = tf.layers.dense(inputs = d1_3, units = 10, activation = tf.nn.relu)
    
            self.logits = tf.layers.dense(inputs = d2, units = self.number_of_actions, activation = None)        
            self.one_hot_actions = tf.one_hot(self.action_ph, self.logits.get_shape().as_list()[-1])
            self.pi = tf.random.categorical(self.logits, 1)
            self.det_pi = tf.argmax(self.logits, axis=-1)
            
        
        self.sess.run(tf.global_variables_initializer())
        
        policy_weights_ph = tf.trainable_variables('policy_net')
        
        with open('policy_weights.pkl', 'rb') as outp:
            policy_weights = pickle.load(outp)
        
        for i in range(self.number_of_layers_from_imitation):
            self.sess.run(tf.assign(policy_weights_ph[i], policy_weights[i]))
            
            
    def act(self, obs, action_space):
        self.obs_buffer.append(obs)
        actions_filtered = action_prune.get_filtered_actions(obs,(None,None))
        inp_mat = np.zeros((1, 11, 11, 33))
        temp_inp1_mat = featurize(obs)
        temp_inp2_mat = featurize_retrospective(self.obs_buffer)
        inp_mat[0, :, :, :] = np.concatenate([temp_inp1_mat, temp_inp2_mat], axis=-1)
        
        act, log_prob = self.sess.run([self.det_pi, self.logits], feed_dict = {self.obs_ph: inp_mat, self.policy_dropout: False, self.value_dropout: False})
        act = act[0]
        log_prob_ind = np.argsort(log_prob, axis=-1)[0]
        
        if act not in actions_filtered:
            ind_to_select = -1
            act1 = log_prob_ind[ind_to_select]
            while(act1 not in actions_filtered):
                ind_to_select -=1
                act1 = log_prob_ind[ind_to_select]
                
            act = act1
            act = np.asarray([act])
        
        message = generate_mess(obs)
        if type(act) == int:
            act = np.asarray([act])
        elif np.asarray(act).shape == ():
            act = np.asarray([act])
        else:
            act = np.asarray(act)
            
        if type(act) == int:
            self.action_duplicate = act
            return [act, int(message[0, 0]), int(message[0, 1])]
        else:
            self.action_duplicate = np.asscalar(act)
            return [np.asscalar(act), int(message[0, 0]), int(message[0, 1])]