
class CartanMatrix(object):
    """
    A CartanMatrix is a set of nodes and a dictionary of 2-tuples of nodes.
    """
    def __init__(self, nodes=None, matrix=None):
        # 'set' object
        self.nodes  = nodes
        # 'dictionary' object whose keys are 2-tuples of elements of 'nodes'
        self.matrix = matrix

    def copy(self):
        """
        Return a copy of 'self'.
        """
        nd = set([n for n in self.nodes])
        mt = {}
        for m in self.nodes:
            for n in self.nodes:
                mt[(m, n)] = self.matrix[(m, n)]
        A = CartanMatrix(nd, mt)
        return A

    def display(self):
        """
        Print on the terminal.
        """
        print("+" + "-" * len(self.nodes) * 3 + "--+")
        for m in self.nodes:
            print("|", end='')
            for n in self.nodes:
                print("%3d" % self.matrix[(m, n)], end='')
            print("  |")
        print("+" + "-" * len(self.nodes) * 3 + "--+")
