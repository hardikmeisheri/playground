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
from pommerman import utility
from pommerman import constants
import copy

INT_MAX=9999.0
BOMBING_TEST='lookahead'
NO_KICKING=True

FLAME_LIFE=2

EPSILON=0.001

def _opposite_direction(direction):
    if direction == constants.Action.Left:
        return constants.Action.Right
    if direction == constants.Action.Right:
        return constants.Action.Left
    if direction == constants.Action.Up:
        return constants.Action.Down
    if direction == constants.Action.Down:
        return constants.Action.Up
    return None

def move_moving_bombs_to_next_position(prev_obs, obs):
    def is_moving_direction(bomb_pos, direction):
        rev_d=_opposite_direction(direction)
        rev_pos=utility.get_next_position(bomb_pos, rev_d)
        if not utility.position_on_board(prev_obs['board'], rev_pos):
            return False
        if prev_obs['bomb_life'][rev_pos] - 1 == obs['bomb_life'][bomb_pos] \
            and prev_obs['bomb_blast_strength'][rev_pos] == obs['bomb_blast_strength'][bomb_pos] \
                and utility.position_is_passage(prev_obs['board'], bomb_pos):
            return True
        return False

    bombs=zip(*np.where(obs['bomb_life']>1))
    moving_bombs=[]
    for bomb_pos in bombs:
        moving_dir=None
        for d in [constants.Action.Left, constants.Action.Right, \
            constants.Action.Up, constants.Action.Down]:
            if is_moving_direction(bomb_pos, d):
                moving_dir=d
                break
        if moving_dir is not None:
            moving_bombs.append((bomb_pos, moving_dir))
    board=obs['board']
    bomb_life=obs['bomb_life']
    bomb_blast_strength=obs['bomb_blast_strength']
    for bomb_pos, moving_dir in moving_bombs:
        next_pos=utility.get_next_position(bomb_pos, moving_dir)
        if not utility.position_on_board(obs['board'], next_pos):
            continue
        if utility.position_is_passage(obs['board'], next_pos):
            board[next_pos]=board[bomb_pos]
            bomb_life[next_pos]=bomb_life[bomb_pos]
            bomb_blast_strength[next_pos]=bomb_blast_strength[bomb_pos]
            board[bomb_pos]=constants.Item.Passage.value
            bomb_life[bomb_pos]=0
            bomb_blast_strength[bomb_pos]=0
    return obs

def _all_directions(exclude_stop=True):
    dirs=[constants.Action.Left, constants.Action.Right, constants.Action.Up, constants.Action.Down]
    return dirs if exclude_stop else dirs + [constants.Action.Stop]

def _all_bomb_real_life(board, bomb_life, bomb_blast_st):
    def get_bomb_real_life(bomb_position, bomb_real_life):
        """One bomb's real life is the minimum life of its adjacent bomb. 
           Not that this could be chained, so please call it on each bomb mulitple times until
           converge
        """
        dirs=_all_directions(exclude_stop=True)
        min_life=bomb_real_life[bomb_position]
        for d in dirs:
            pos=bomb_position
            last_pos=bomb_position
            while True:
                pos=utility.get_next_position(pos, d)
                if _stop_condition(board, pos):
                    break
                if bomb_real_life[pos] > 0:
                    if bomb_real_life[pos]<min_life and \
                        _manhattan_distance(pos, last_pos) <= bomb_blast_st[pos]-1:
                        min_life = bomb_real_life[pos]
                        last_pos=pos
                    else:
                        break
        return min_life
    bomb_real_life_map=np.copy(bomb_life) 
    sz=len(board)
    while True:
        no_change=[]
        for i in range(sz):
            for j in range(sz):
                if utility.position_is_wall(board, (i,j)) or utility.position_is_powerup(board, (i,j)) \
                    or utility.position_is_fog(board, (i,j)):
                    continue
                if bomb_life[i,j] < 0+EPSILON:
                    continue
                real_life=get_bomb_real_life((i,j), bomb_real_life_map)
                no_change.append(bomb_real_life_map[i,j] == real_life)
                bomb_real_life_map[i,j]=real_life
        if all(no_change):
            break
    return bomb_real_life_map

def _manhattan_distance(pos1, pos2):
    #the manhattan distance here is for the specific case
    assert(pos1[0]==pos2[0] or pos1[1]==pos2[1])
    return abs(pos1[0]-pos2[0]) + abs(pos1[1]-pos2[1])

def _stop_condition(board, pos, exclude_agent=True):
    if not utility.position_on_board(board, pos):
        return True
    if utility.position_is_fog(board, pos):
        return True
    if utility.position_is_wall(board, pos):
        return True
    if not exclude_agent:
        if utility.position_is_agent(board, pos):
            return True
    return False

def _position_covered_by_bomb(obs, pos, bomb_real_life_map):
    #return a tuple (True/False, min_bomb_life_value, max life value)
    min_bomb_pos, max_bomb_pos=None, None
    min_bomb_value, max_bomb_value=INT_MAX, -INT_MAX
    if obs['bomb_life'][pos]>0:
        min_bomb_value, max_bomb_value=bomb_real_life_map[pos],bomb_real_life_map[pos]
        min_bomb_pos, max_bomb_pos=pos, pos
    dirs=_all_directions(exclude_stop=True)
    board=obs['board']
    for d in dirs:
        next_pos=pos
        while True:
            next_pos=utility.get_next_position(next_pos, d)
            if _stop_condition(board, next_pos, exclude_agent=True):
                #here should assume agents are dynamic
                break
            if obs['bomb_life'][next_pos]>0 and obs['bomb_blast_strength'][next_pos] - 1 >= _manhattan_distance(pos, next_pos):
                if bomb_real_life_map[next_pos] < min_bomb_value:
                    min_bomb_value=bomb_real_life_map[next_pos]
                    min_bomb_pos=next_pos
                if bomb_real_life_map[next_pos] > max_bomb_value:
                    max_bomb_value=bomb_real_life_map[next_pos]
                    max_bomb_pos = next_pos
                break
    if min_bomb_pos is not None:
        return True, min_bomb_value, max_bomb_value
    return False, INT_MAX, -INT_MAX
  
def _compute_min_evade_step(obs, parent_pos_list, pos, bomb_real_life):
    flag_cover, min_cover_value, max_cover_value=_position_covered_by_bomb(obs, pos, bomb_real_life)
    if not flag_cover:    
        return 0
    elif len(parent_pos_list) >= max_cover_value:
        if len(parent_pos_list)   > max_cover_value + FLAME_LIFE :
            return 0
        else:
            return INT_MAX
    elif len(parent_pos_list)  >= min_cover_value:
        if len(parent_pos_list)  > min_cover_value + FLAME_LIFE:
            return 0
        else:
            return INT_MAX
    else:
        board=obs['board']
        dirs=_all_directions(exclude_stop=True)
        min_step=INT_MAX
        for d in dirs:
            next_pos=utility.get_next_position(pos, d)
            if not utility.position_on_board(board, next_pos):
                continue
            if not (utility.position_is_passage(board, next_pos) or \
            utility.position_is_powerup(board, next_pos)):
                continue
            if next_pos in parent_pos_list:
                continue
            x=_compute_min_evade_step(obs, parent_pos_list+[next_pos], next_pos, bomb_real_life)
            min_step=min(min_step, x+1)
        return min_step

def _check_if_flame_will_gone(obs, prev_two_obs, flame_pos):
    assert(prev_two_obs[0] is not None)
    assert(prev_two_obs[1] is not None)
    #check the flame group in current obs, see if 
    #the whole group was there prev two obs
    #otherwise, although this flame appears in prev two obs, 
    #it could be a old overlap new, thus will not gone next step
    if not (utility.position_is_flames(prev_two_obs[0]['board'], flame_pos) \
        and utility.position_is_flames(prev_two_obs[1]['board'], flame_pos)):
        return False
    board=obs['board']
    Q=deque(maxlen=121)
    Q.append(flame_pos)
    visited=[flame_pos]
    dirs=_all_directions(exclude_stop=True)
    while len(Q)>0:
        pos=Q.popleft()
        if not (utility.position_is_flames(prev_two_obs[0]['board'], pos) \
            and utility.position_is_flames(prev_two_obs[1]['board'], pos)):
            return False
        for d in dirs:
            next_pos=utility.get_next_position(pos, d)
            if utility.position_on_board(board, next_pos) and utility.position_is_agent(board, next_pos):
                if next_pos not in visited:
                    Q.append(next_pos)
                    visited.append(next_pos)
    return True

def _compute_safe_actions(obs, exclude_kicking=False, prev_two_obs=(None,None)):
    dirs=_all_directions(exclude_stop=True)
    ret=set()
    my_position, board, blast_st, bomb_life, can_kick=obs['position'], obs['board'], obs['bomb_blast_strength'], obs['bomb_life'], obs['can_kick']
    kick_dir=None
    bomb_real_life_map=_all_bomb_real_life(board, bomb_life, blast_st)
    flag_cover_passages=[]
    for direction in dirs:
        position = utility.get_next_position(my_position, direction) 
        if not utility.position_on_board(board, position):
            continue
        if (not exclude_kicking) and utility.position_in_items(board, position, [constants.Item.Bomb]) and can_kick:
            #filter kick if kick is unsafe
            if _kick_test(board, blast_st, bomb_real_life_map, my_position, direction): 
                ret.add(direction.value)
                kick_dir=direction.value
        gone_flame_pos=None
        if (prev_two_obs[0]!=None and prev_two_obs[1]!=None) and _check_if_flame_will_gone(obs, prev_two_obs, position):
            #three consecutive flames means next step this position must be good
            #make this a candidate
            obs['board'][position]=constants.Item.Passage.value
            gone_flame_pos=position

        if utility.position_is_passage(board, position) or utility.position_is_powerup(board, position):
            my_id=obs['board'][my_position]
            obs['board'][my_position]= constants.Item.Bomb.value if bomb_life[my_position]>0  else constants.Item.Passage.value
            flag_cover, min_cover_value , max_cover_value=_position_covered_by_bomb(obs, position, bomb_real_life_map)
            flag_cover_passages.append(flag_cover)
            if not flag_cover:
                ret.add(direction.value)
            else:
                min_escape_step=_compute_min_evade_step(obs, [position], position, bomb_real_life_map)
                assert(min_escape_step>0)
                if min_escape_step  < min_cover_value:
                    ret.add(direction.value)
            obs['board'][my_position]=my_id
        if gone_flame_pos is not None:
            obs['board'][gone_flame_pos]=constants.Item.Flames.value

    # Test Stop action only when agent is covered by bomb, 
    # otherwise Stop is always an viable option 
    my_id=obs['board'][my_position]
    obs['board'][my_position]= constants.Item.Bomb.value if bomb_life[my_position]>0  else constants.Item.Passage.value    
    #REMEMBER: before compute min evade step, modify board accordingly first..
    flag_cover, min_cover_value, max_cover_value=_position_covered_by_bomb(obs, my_position, bomb_real_life_map)
    if flag_cover:
        min_escape_step=_compute_min_evade_step(obs, [None, my_position], my_position, bomb_real_life_map)
        if min_escape_step < min_cover_value:
            ret.add(constants.Action.Stop.value)
    else:
        ret.add(constants.Action.Stop.value)
    obs['board'][my_position]=my_id
    #REMEBER: change the board back

    #Now test bomb action
    if not (obs['ammo'] <=0 or obs['bomb_life'][obs['position']]>0):
        #place bomb might be possible
        assert(BOMBING_TEST in ['simple', 'simple_adjacent', 'lookahead'])
        if BOMBING_TEST == 'simple':
            if not flag_cover:
                ret.add(constants.Action.Bomb.value)
        elif BOMBING_TEST == 'simple_adjacent':
            if (not flag_cover) and (not any(flag_cover_passages)): 
                ret.add(constants.Action.Bomb.value)
        else: #lookahead
            if (constants.Action.Stop.value in ret) and len(ret)>1 and (kick_dir is None):
                obs2=copy.deepcopy(obs)
                my_pos=obs2['position']
                obs2['board'][my_pos]=constants.Item.Bomb.value
                obs2['bomb_life'][my_pos]=min_cover_value if flag_cover else 10
                obs2['bomb_blast_strength'][my_pos]=obs2['blast_strength']
                bomb_life2, bomb_blast_st2, board2=obs2['bomb_life'], obs2['bomb_blast_strength'],obs2['board']
                bomb_real_life_map=_all_bomb_real_life(board2, bomb_life2, bomb_blast_st2)
                min_evade_step=_compute_min_evade_step(obs2, [None, my_position], my_pos, bomb_real_life_map)
                current_cover_value=obs2['bomb_life'][my_pos]
                if min_evade_step  < current_cover_value:
                    ret.add(constants.Action.Bomb.value)
    return ret

def get_filtered_actions(obs, prev_two_obs=None):
    if obs['board'][obs['position']] not in obs['alive']:
        return [constants.Action.Stop.value]
    obs_cpy=copy.deepcopy(obs)
    if prev_two_obs[-1] is not None:
        obs=move_moving_bombs_to_next_position(prev_two_obs[-1], obs)
    ret=_compute_safe_actions(obs,exclude_kicking=NO_KICKING, prev_two_obs=prev_two_obs)
    obs=obs_cpy
    if len(ret)!=0:
        return list(ret)
    else:
        return [constants.Action.Stop.value]

def _kick_test(board, blast_st, bomb_life, my_position, direction):
    def moving_bomb_check(moving_bomb_pos, p_dir, time_elapsed):
        pos2=utility.get_next_position(moving_bomb_pos, p_dir)
        dist=0
        for i in range(10):
            dist +=1
            if not utility.position_on_board(board, pos2):
                break
            if not (utility.position_is_powerup(board, pos2) or utility.position_is_passage(board, pos2)):
                break
            life_now=bomb_life[pos2] - time_elapsed
            if bomb_life[pos2]>0 and life_now>=-2 and life_now <= 0 and dist<blast_st[pos2]:
                return False
            pos2=utility.get_next_position(pos2, p_dir)
        return True
    next_position=utility.get_next_position(my_position, direction)
    assert(utility.position_in_items(board, next_position, [constants.Item.Bomb]))
    life_value=int(bomb_life[next_position])
    strength=int(blast_st[next_position])
    dist=0
    pos=utility.get_next_position(next_position, direction)
    perpendicular_dirs=[constants.Action.Left, constants.Action.Right] 
    if direction == constants.Action.Left or direction == constants.Action.Right:
        perpendicular_dirs=[constants.Action.Down, constants.Action.Up]
    for i in range(life_value): 
        if utility.position_on_board(board, pos) and utility.position_is_passage(board, pos):
            #do a check if this position happens to be in flame when the moving bomb arrives!
            if not (moving_bomb_check(pos, perpendicular_dirs[0], i) and moving_bomb_check(pos, perpendicular_dirs[1], i)):
                break
            dist +=1 
        else:
            break
        pos=utility.get_next_position(pos, direction)
        #can kick and kick direction is valid
    return dist > strength

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
        super().__init__()
        
        self.sess = sess
        self.obs_ph = tf.placeholder(dtype = tf.float32, shape = [None, 11,11,33], name = 'obs_ph')
        self.action_ph = tf.placeholder(dtype = tf.int32, shape = [None], name = 'actions')
        self.look_back = 5
        self.retro_threshold = 50
        self.obs_buffer = deque(maxlen = self.retro_threshold)
        self.number_of_actions = 6
        with tf.variable_scope('policy_net'):
            cnn1 = tf.layers.conv2d(inputs = self.obs_ph, filters = 128, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            cnn2 = tf.layers.conv2d(inputs = cnn1, filters = 128, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            max_pool1 = tf.layers.max_pooling2d(inputs = cnn2, pool_size = 2, strides = 2)
    
            cnn3 = tf.layers.conv2d(inputs = max_pool1, filters = 64, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            cnn4 = tf.layers.conv2d(inputs = cnn3, filters = 64, kernel_size = 3, padding = "same", activation = tf.nn.relu, strides = 1)
            max_pool2 = tf.layers.max_pooling2d(inputs = cnn4, pool_size = 2, strides = 2)
            
    
            flatten_1 = tf.layers.flatten(inputs = max_pool2)
            
            
            d1 = tf.layers.dense(inputs = flatten_1, units = 50, activation = tf.nn.relu)        
            d2 = tf.layers.dense(inputs = d1, units = 10, activation = tf.nn.relu)
    
            self.logits = tf.layers.dense(inputs = d2, units = self.number_of_actions, activation = None)        
            self.one_hot_actions = tf.one_hot(self.action_ph, self.logits.get_shape().as_list()[-1])
            self.pi = tf.random.categorical(self.logits, 1)
            self.det_pi = tf.argmax(self.logits, axis=-1)
            
        
        self.sess.run(tf.global_variables_initializer())
        
        policy_weights_ph = tf.trainable_variables('policy_net')
        
        with open('policy_weights.pkl', 'rb') as outp:
            policy_weights = pickle.load(outp)
        
        for i in range(len(policy_weights_ph)):
            self.sess.run(tf.assign(policy_weights_ph[i], policy_weights[i]))
            
            
    def act(self, obs, action_space):
        self.obs_buffer.append(obs)
        actions_filtered = get_filtered_actions(obs,(None,None))
        inp_mat = np.zeros((1, 11, 11, 33))
        temp_inp1_mat = featurize(obs)
        temp_inp2_mat = featurize_retrospective(self.obs_buffer)
        inp_mat[0, :, :, :] = np.concatenate([temp_inp1_mat, temp_inp2_mat], axis=-1)
        
        act, log_prob = self.sess.run([self.det_pi, self.logits], feed_dict = {self.obs_ph: inp_mat})
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
            
            return [act, int(message[0, 0]), int(message[0, 1])]
        else:
            
            return [np.asscalar(act), int(message[0, 0]), int(message[0, 1])]