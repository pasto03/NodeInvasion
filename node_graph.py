import random


class WarGraphNode:
    def __init__(self, name, n_edges=3, initial_energy=0):
        self.name = name
        self.edges = []
        self.initial_energy = initial_energy
        self.energy = self.initial_energy
        self.occupied = self.energy > 0
        self.n_edges = n_edges
        self.full = False

    def _reset_all(self):
        self.energy = self.initial_energy
        self.occupied = self.energy > 0
        self.edges = []
        self.full = False

    def reset(self):
        self.energy = self.initial_energy
        self.occupied = self.energy > 0

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"WarGraphNode(name={self.name}, energy={self.energy}, occupied={self.occupied}, edges={self.edges}, " \
               f"n_edges={self.n_edges}, full={self.full})"


# WarGraph class
class WarGraph:
    def __init__(self, nodeName, n_edges=3, random_mainNode=True):
        self.random_mainNode = random_mainNode
        headEnergy = 100 if not self.random_mainNode else 0
        self.n_edges = n_edges
        # self.mainNode -> self.headNode as this node may not be mainNode
        self.headNode = WarGraphNode(name=nodeName, n_edges=self.n_edges, initial_energy=headEnergy)
        self.adj_list = {self.headNode.name: []}
        self.nodes = {self.headNode.name: self.headNode}
        self.available_nodes = {self.headNode.name: self.headNode}

        self.initialized = False

    # add node
    def addNode(self, target, nodeName: str):
        assert target in self.adj_list, "Target node not exist"
        targetNode = self.nodes[target]

        node = self.nodes[nodeName] if nodeName in self.adj_list.keys() else WarGraphNode(nodeName)

        # update edges for target node and new node
        targetNode.edges.append(node.name)
        node.edges.append(targetNode.name)

        # update adjacency list
        self.adj_list[node.name] = node.edges
        self.adj_list[targetNode.name] = targetNode.edges

        # update nodes
        if node.name not in self.nodes.keys():
            self.nodes[node.name] = node

    # after reset_nodes(), mainNode need to be reintialized again
    def reset_nodes(self):
        for node in self.nodes.values():
            node.reset()
        self.initialized = False

    def initialize_mainNode(self, nodeName=None):
        if self.initialized:
            return
            # param nodeName has higher priority than self.random_mainNode
        if not nodeName:
            assert self.random_mainNode, "Please specify nodeName for mainNode or set self.random_mainNode as True"
            nodeName = random.choice(list(self.nodes.keys()))
        else:
            assert nodeName in self.nodes.keys(), "Specified nodeName not exist"
        mainNode = self.nodes[nodeName]
        mainNode.energy = 100
        mainNode.occupied = True
        self.initialized = True
        return mainNode

    def generate_graph(self, n_nodes, n_edges=5, index_constraint=None):
        available_nodes = {self.headNode.name: self.headNode}
        for n in range(1, n_nodes):
            # create a newNode
            newNode = WarGraphNode(n, n_edges=n_edges)

            # randomly select a node where its capacity is not full(not node.full)
            index_constraint = index_constraint if index_constraint else - int(n_nodes / 2)
            selected_nodeName = random.choice(
                list(available_nodes.keys())[index_constraint:])  # only select from last 10 available nodes
            selectedNode = available_nodes[selected_nodeName]

            # update their edges
            newNode.edges.append(selected_nodeName)
            selectedNode.edges.append(n)

            # add new node to graph.nodes
            self.nodes[n] = newNode

            # update available nodes
            available_nodes[n] = newNode

            # update adjacency list
            self.adj_list[newNode.name] = newNode.edges
            self.adj_list[selectedNode.name] = selectedNode.edges

            # if node.full, remove its key from available nodes
            newNode.full = len(newNode.edges) == newNode.n_edges
            if newNode.full:
                _ = available_nodes.pop(n)
            selectedNode.full = len(selectedNode.edges) == selectedNode.n_edges
            if selectedNode.full:
                _ = available_nodes.pop(selectedNode.name)

        return self.nodes

    def diffusion(self, selected_nodeName: str, target_nodeName: str):
        assert self.initialized, "Please call self.initialize_mainNode() before diffusion operation"
        assert not selected_nodeName == target_nodeName, "Selecting same selected node and target node is invalid"
        assert selected_nodeName in self.adj_list, "Selected node not exist"
        assert target_nodeName in self.adj_list, "Target node not exist"
        assert target_nodeName in self.nodes[selected_nodeName].edges, "Target node is not an edge of selected node"
        diffusable_nodes = [k for k, v in self.nodes.items() if v.energy > 50]
        # print("Diffusable nodes:", diffusable_nodes)

        selected_node = self.nodes[selected_nodeName]
        assert selected_node.energy > 50, "Not enough energy"

        # print("All good")
        # print("Selected node edges:", selected_node.edges)

        # select one of the edge and diffuse selected node's energy
        target_node = self.nodes[target_nodeName]
        assert not target_node.occupied, "Target node has been occupied"

        target_node.energy = 50
        target_node.occupied = True
        selected_node.energy -= 50

    def __repr__(self):
        return f"WarGraph(headNode={self.headNode})"


if __name__ == '__main__':
    graph = WarGraph('S', n_edges=7)
    graph.generate_graph(n_nodes=30, n_edges=3, index_constraint=-10)
    print(graph.nodes)
