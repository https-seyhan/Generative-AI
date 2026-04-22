
class Node:
    def __init__(self, name, func):
        self.name = name
        self.func = func
        self.next_nodes = []

    def connect(self, node):
        self.next_nodes.append(node)

    def run(self, data):
        result = self.func(data)
        for node in self.next_nodes:
            node.run(result)

# Example usage
def run_graph(start_node, data):
    start_node.run(data)
