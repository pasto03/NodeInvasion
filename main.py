import os
import shutil

import math
import random
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

from IPython.display import HTML

from tqdm import tqdm
import gym
from gym.spaces import MultiDiscrete

import networkx as nx

from node_graph import WarGraph, WarGraphNode


class NodeInvasionEnv(gym.Env):
    def __init__(self, max_timestep=200, render_mode='np_array', graph=None, mainNode='S', n_nodes=30, n_edges=5,
                 delta_energy=1):
        self.render_mode = render_mode  # 'np_array' -> obs; 'human' -> video
        # 'human' render mode takes about 200x longer time to run an episode
        self.max_timestep = max_timestep if self.render_mode == 'np_array' else min(max_timestep, 120)
        self.delta_energy = delta_energy
        self.timestep = 0

        self.graph = graph
        self.mainNode = mainNode
        if not self.graph:
            self.graph = WarGraph(mainNode, n_edges=7)
            nodes = self.graph.generate_graph(n_nodes=n_nodes, n_edges=n_edges)
            # self.graphNodes = self.graph.nodes

        self.graph.reset_nodes()
        self.graph.initialize_mainNode(mainNode)  # mainNode can be changed later
        self.graphNodes = self.graph.nodes

        if self.render_mode == 'human':
            self.G, self.node_color, self.node_color_dict = self._initialize_nx_graph(self.graphNodes)
            self.root_path = "nodeGraph"
            self.img_paths = []
            self.image_idx = 0

            if os.path.exists(self.root_path):
                shutil.rmtree(self.root_path)
            os.mkdir(self.root_path)
            os.mkdir(self.root_path + '/images')

        self.graphNodeIdx = {idx: i for idx, i in enumerate(list(self.graphNodes.keys()))}  # index for each nodes
        self.inv_graphNodeIdx = {v: k for k, v in self.graphNodeIdx.items()}

        self.obs = np.array([(node.energy, int(node.occupied)) for node in self.graphNodes.values()])
        self.n_nodes = self.obs.shape[0]

        self.observation_space = MultiDiscrete([101, 2])
        self.action_space = MultiDiscrete([self.n_nodes] * 2)

    def reset(self):
        self.timestep = 0
        self.graph.reset_nodes()
        self.graph.initialize_mainNode(self.mainNode)
        if self.render_mode == 'human':
            if os.path.exists(self.root_path):
                shutil.rmtree(self.root_path)
            os.mkdir(self.root_path)
            os.mkdir(self.root_path + '/images')
        return np.array([(node.energy, int(node.occupied)) for node in self.graphNodes.values()])

    # initialize nx graph
    @staticmethod
    def _initialize_nx_graph(nodes: dict[str, WarGraphNode]):
        G = nx.Graph()
        node_color_dict = dict()
        for node in nodes.values():
            color = 'lightgray' if node.energy == 0 else 'lightskyblue'
            node_color_dict[node.name] = color
            G.add_node(node.name, energy=node.energy, full=node.full, occupied=node.occupied)
        node_color = list(node_color_dict.values())

        for node in nodes.values():
            u = node.name
            for v in node.edges:
                G.add_edge(u, v)
        return G, node_color, node_color_dict

    # a function to get index of key in dict
    @staticmethod
    def keyindex(d: dict, key):
        i = 0
        for k, v in d.items():
            if k == key:
                return i
            i += 1
        return -1

    # update energy per step
    def __update_energy(self):
        for gNode in self.graphNodes.values():
            if gNode.occupied and 100 - gNode.energy >= self.delta_energy:
                gNode.energy += self.delta_energy

    # update available nodes in graph(indicates which nodes in the graph can be selected to diffuse)
    # call this function before end of a step(after success diffusion and energy update)
    def __update_available_nodes(self):
        # first requirement: node is occupied
        occupied_nodes = [node for _, node in self.graphNodes.items() if node.occupied]
        # second requirement: node.energy > 50
        for node in occupied_nodes:
            # we remove those nodes in available nodes where energy <= 50
            node_in_available = node.name in self.graph.available_nodes.keys()
            if node.energy <= 50 and node_in_available:
                _ = self.graph.available_nodes.pop(node.name)
            if node.energy > 50 and not node_in_available:
                self.graph.available_nodes[node.name] = node

    def __save_nx_graph(self):
        """this function should be called when graph is modified -- success diffusion or energy updated"""
        labels = nx.get_node_attributes(self.G, 'energy')

        plt.figure(figsize=(17, 17))
        subax1 = plt.subplot(224)

        # set seed to fix position of all nodes
        optimal_distance = (1 / math.sqrt(self.G.number_of_nodes())) * 2  # double of default optimal distance
        nx.draw_networkx(self.G, node_color=self.node_color, pos=nx.spring_layout(self.G, seed=1, k=optimal_distance),
                         labels=labels, font_size=10, font_weight='bold', node_size=350)

        img_fullpath = self.root_path + '/images' + f'/{self.image_idx}.png'
        self.img_paths.append(img_fullpath)
        plt.savefig(img_fullpath, bbox_inches='tight')
        plt.close()

        self.image_idx += 1

    # update nx graph upon success diffusion operation
    def __nx_graph_diffusion(self, selected_nodeName, target_nodeName):
        # change color of diffused node upon effective diffusion
        self.G.nodes()[selected_nodeName]['energy'] -= 50
        self.G.nodes()[target_nodeName]['energy'] = 50
        self.G.nodes()[target_nodeName]['occupied'] = True
        self.node_color[self.keyindex(self.node_color_dict, target_nodeName)] = 'lightskyblue'

    def step(self, action):
        # 1. diffuse energy
        selected_nodeName, target_nodeName = self.graphNodeIdx[action[0]], self.graphNodeIdx[action[1]]
        try:
            # reward for success diffusion: total_energy / n_nodes
            self.graph.diffusion(selected_nodeName, target_nodeName)
            reward = int(sum(self.obs[:, 0]) / self.n_nodes)

            if self.render_mode == 'human':
                # 1.1 update nx graph nodes for diffuse operation
                self.__nx_graph_diffusion(selected_nodeName, target_nodeName)

                # 1.2 take a snapshot of success diffusion
                self.__save_nx_graph()

        except AssertionError as ae:
            # penalty for invalid difussion: -1
            reward = -1

        # 3. update energy(should be done after snapshot of success diffusion)
        # 3.1 update energy for graph
        self.__update_energy()

        # 3.2 update energy attribute for nx graph
        if self.render_mode == 'human':
            for name, node in self.G.nodes(data=True):
                # condition: node is occupied and 100 - node energy >= delta energy(delta energy will change someday)
                if node['occupied'] and 100 - node['energy'] >= self.delta_energy:
                    node['energy'] += self.delta_energy

            # 3.3 take snapshot after energy updated
            self.__save_nx_graph()

        # 4. update available nodes in graph(indicates which nodes in the graph can be selected to diffuse)
        self.__update_available_nodes()

        # 5. update step outputs(obs, reward, done, _)
        # 5.1 observation, timestep
        # obs: [energy, int(occupied)] for each nodes
        self.obs = np.array([(node.energy, int(node.occupied)) for node in self.graphNodes.values()])
        self.timestep += 1

        # 5.2 done
        truncation = self.timestep >= self.max_timestep
        termination = (sum(self.obs[:, 0]) / self.n_nodes) == 100  # terminated when all nodes have full energy

        done = truncation or termination

        # 5.3 reward
        # update reward if done
        if done:
            reward = int(sum(self.obs[:, 0]) / self.n_nodes)

        return self.obs, reward, done, dict()

    @staticmethod
    def _visualize_rewards(rewards):
        plt.plot(rewards)
        plt.xlabel("Timestep")
        plt.ylabel("Reward")
        plt.show()

    def render(self, interval=400, ipython=True):
        interval = max(interval, 400)
        if self.render_mode == 'np_array':
            return plt.imread(self.img_paths[-1])

        elif self.render_mode == 'human':
            images = [plt.imread(img_path) for img_path in self.img_paths]
            # visualize graph as video
            fig, ax = plt.subplots()
            plt.axis('off')
            artists = []

            for img in images:
                container = ax.imshow(img)
                artists.append([container])

            ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=interval)
            plt.close()
            if not ipython:
                ani.save("01.mp4", writer='ffmpeg', fps=5)
                return
            return HTML(ani.to_html5_video())


def _test_env():
    env = NodeInvasionEnv(max_timestep=100, n_nodes=30, n_edges=3, render_mode='human')
    obs = env.reset()

    # nodes_record = [graph.nodes]
    rewards = []
    bar = tqdm(total=env.max_timestep)

    done = False
    while not done:
        graph = env.graph
        selected_nodeName = random.choice(list(graph.available_nodes.keys()))
        selectedNode = graph.nodes[selected_nodeName]
        target_nodeName = random.choice(selectedNode.edges)
        targetNode = graph.nodes[target_nodeName]

        action = (env.inv_graphNodeIdx[selected_nodeName], env.inv_graphNodeIdx[target_nodeName])
        next_obs, reward, done, _ = env.step(action)

        obs = next_obs
        rewards.append(reward)
        # nodes_record.append(graph.nodes)

        bar.update(1)
    bar.close()

    # env.timestep
    return env


if __name__ == '__main__':
    env = _test_env()
    # render output images as video and save as mp4
    ani = env.render(ipython=False)
    # ani.save("01.mp4", writer='ffmpeg', fps=5)

