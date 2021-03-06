"""Implementation of a simple deterministic agent using Docker."""

from pommerman import agents
from pommerman.runner import DockerAgentRunner
import tensorflow as tf

class MyAgent_with_comm(DockerAgentRunner):

    def __init__(self):
        sess = tf.Session()
        self._agent = agents.RL_Agent(sess)

    def init_agent(self, id, game_type):
        return self._agent.init_agent(id, game_type)

    def act(self, observation, action_space):
        return self._agent.act(observation, action_space)

    def episode_end(self, reward):
        return self._agent.episode_end(reward)

    def shutdown(self):
        return self._agent.shutdown()


def main():
    '''Inits and runs a Docker Agent'''
    agent = MyAgent_with_comm()
    agent.run()


if __name__ == "__main__":
    main()
