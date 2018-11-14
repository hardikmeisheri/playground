'''This is the basic docker agent runner'''
import abc
import logging
import json
from .. import constants
from flask import Flask, jsonify, request
import numpy as np

LOGGER = logging.getLogger(__name__)


class DockerAgentRunner(metaclass=abc.ABCMeta):
    """Abstract base class to implement Docker-based agent"""

    def __init__(self):
        pass

    @abc.abstractmethod
    def act(self, observation, action_space):
        """Given an observation, returns the action the agent should"""
        
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
        
        feat = featurize(observation)
        probs = self.actor_model.predict(feat[np.newaxis])
        
        action = np.argmax(probs)
        return action.item()
        
        
        raise NotImplementedError()

    def run(self, host="0.0.0.0", port=10080):
        """Runs the agent by creating a webserver that handles action requests."""
        app = Flask(self.__class__.__name__)

        @app.route("/action", methods=["POST"])
        def action(): #pylint: disable=W0612
            '''handles an action over http'''
            data = request.get_json()
            observation = data.get("obs")
            observation = json.loads(observation)
            action_space = data.get("action_space")
            action_space = json.loads(action_space)
            action = self.act(observation, action_space)
            return jsonify({"action": action})

        @app.route("/init_agent", methods=["POST"])
        def init_agent(): #pylint: disable=W0612
            '''initiates agent over http'''
            data = request.get_json()
            id = data.get("id")
            id = json.loads(id)
            game_type = data.get("game_type")
            game_type = constants.GameType(json.loads(game_type))
            self.init_agent(id, game_type)
            return jsonify(success=True)

        @app.route("/shutdown", methods=["POST"])
        def shutdown(): #pylint: disable=W0612
            '''Requests destruction of any created objects'''
            self.shutdown()
            return jsonify(success=True)

        @app.route("/episode_end", methods=["POST"])
        def episode_end(): #pylint: disable=W0612
            '''Info about end of a game'''
            data = request.get_json()
            reward = data.get("reward")
            reward = json.loads(reward)
            self.episode_end(reward)
            return jsonify(success=True)

        @app.route("/ping", methods=["GET"])
        def ping(): #pylint: disable=W0612
            '''Basic agent health check'''
            return jsonify(success=True)

        LOGGER.info("Starting agent server on port %d", port)
        app.run(host=host, port=port)
