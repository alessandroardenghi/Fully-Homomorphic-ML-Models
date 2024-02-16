
class MyTree:
    def __init__(self, feature=None, threshold=None, leaf_value=None):
        self.feature = feature
        self.threshold = threshold
        self.leaf_value = leaf_value
        self.left = None
        self.right = None

def build_tree(max_depth):
    if max_depth <= 0:
        return None

    node = MyTree()

    node.left = build_tree(max_depth - 1)
    node.right = build_tree(max_depth - 1)

    return node


# Print tree structure
def print_tree(node, level=0):
    if node is None:
        return

    indent = '  ' * level
    if node.feature is not None:
        print(indent + f"Feature: {node.feature}")
        print(indent + f"Threshold: {node.threshold}")
    else:
        print(indent + f"Leaf Value: {node.leaf_value}")

    print_tree(node.left, level + 1)
    print_tree(node.right, level + 1)
