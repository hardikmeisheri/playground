#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 12:33:51 2018

@author: hardik
"""
import numpy as np
from pommerman import agents
from keras.models import load_model

class RL_learned_agent(agents.BaseAgent):
    def __init__(self):
        super().__init__()
        self.actor_model = load_model('actor_model_RL.h5')
        
    def act(self, obs, action_space):
        
        def featurize(obs):
           
            board = obs['board']
        
            # convert board items into bitmaps
            maps = [board == i for i in range(10)]
            maps.append(obs['bomb_blast_strength'])
            maps.append(obs['bomb_life'])
        
            # duplicate ammo, blast_strength and can_kick over entire map
            maps.append(np.full(board.shape, obs['ammo']))
            maps.append(np.full(board.shape, obs['blast_strength']))
            maps.append(np.full(board.shape, obs['can_kick']))
        
            # add my position as bitmap
            position = np.zeros(board.shape)
            position[obs['position']] = 1
            maps.append(position)
        
            # add teammate
            if obs['teammate'] is not None:
                maps.append(board == obs['teammate'].value)
            else:
                maps.append(np.zeros(board.shape))
        
            # add enemies
            enemies = [board == e.value for e in obs['enemies']]
            maps.append(np.any(enemies, axis=0))
        
            old_state = np.stack(maps, axis=2)
            
            augmented_mat = obs['board'].copy()
            augmented_mat[np.where((obs['board'] == 6) | (obs['board'] == 7) | (obs['board'] == 8))] = 0 # Powerups
            augmented_mat[np.where(obs['board'] == 2)] = 1 # wooden Wall
            augmented_mat[np.where(obs['board'] == 0)] = 2 # Passage
            augmented_mat[np.where(obs['board'] == 5)] = 3 # Fog
            augmented_mat[np.where((obs['board'] == obs['enemies'][0].value) | (obs['board'] == obs['enemies'][1].value))] = 4 # enemy
            augmented_mat[np.where(obs['board'] == 1)] = 5 # Rigid Wall
            augmented_mat[np.where(obs['board'] == obs['teammate'].value)] = 6 # Teammate
            augmented_mat[np.where(obs['board'] == 3)] = 7 # Bomb
            augmented_mat[np.where(obs['board'] == 4)] = 8 # Flames
            augmented_mat[augmented_mat > 8] = 6 # Own value with teammate
            
            new_obs_cust = np.concatenate([old_state, augmented_mat.reshape((augmented_mat.shape[0], augmented_mat.shape[1], 1))], axis=-1)
            
            return new_obs_cust
        
        feat = featurize(obs)
        probs = self.actor_model.predict(feat[np.newaxis])
        
        action = np.argmax(probs)
        return action.item()
