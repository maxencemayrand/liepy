import random

class Graph(object):
    """
    Data structure for finite directed graphs with weights on nodes and edges.
    The set of nodes and their weights is implemented by a dictionary. The set
    of edges is implemented by a dictionary: each key is a 2-tuple of integers,
    where a tuple (i, j) means there is an edge from node i to node j (in the
    odering given by the list), and the value of the key is the weight of that
    edge.
    """
    def __init__(self, nodes={}, edges={}, name=None):
        self.nodes = nodes  # Dictionary of nodes (values are weights).
        self.edges = edges  # Dictionary of edges (2-tuples with weights).
        self.name  = name   # string: name of the graph.
        self.size  = len(self.nodes) # Size of the graph.

    def set_nodes(self, nodes):
        """
        Set the dictionary of nodes.
        """
        self.nodes = nodes
        self.update_size()

    def set_edges(self, edges):
        """
        Set the dictionary of edges.
        """
        self.edges = edges

    def update_size(self):
        """
        Update the size of the graph.
        """
        self.size = len(self.nodes)

    def info(self):
        """
        Print the data of the graph on the terminal.
        """
        print("Name: %s" % self.name) # Name of graph.
        print()
        print("%d nodes" % self.size) # Number of nodes.
        print("%d edges" % len(self.edges)) # Number of edges.
        print()
        print("Nodes and weights") # List of the nodes and their weight.
        for n in self.nodes:
            print("%r : %r" % (n, self.nodes[n]))
        print()
        print("Edges and weights") # List of the edges and their weight.
        for e in self.edges:
            print("%r --> %r  :  %r" % (e[0], e[1], self.edges[e]))

    def relabel(self, k):
        """
        Relabel the nodes as the integers k, k+1, ...
        """
        # The first step is to construct a bijective map (a dictionary) from
        # the set of nodes to {k, k+1, ...}. We call this map 'ordering'.
        ordering = {}
        i = k
        for n in self.nodes:
            ordering[n] = i
            i += 1
        # Then we use this map to relabel the edges and nodes
        self.edges = {(ordering[e[0]], ordering[e[1]]) : self.edges[e] for e
            in self.edges}
        self.nodes = {ordering[n] : self.nodes[n] for n in self.nodes}


    def add(self, dyn):
        """
        Add a graph.
        i.e. self becomes the disjoint union of self and dyn.
        We relabel all the nodes as integers 0, 1, 2, ...
        """
        # We don't want to change the original 'dyn'
        copy = dyn.copy()

        # Relabel both set of nodes so that we don't have duplicates.
        self.relabel(0)
        copy.relabel(self.size)

        # Update the nodes
        self.nodes.update(copy.nodes)

        # Update the edges.
        self.edges.update(copy.edges)

        # If they both have a name, let's add them.
        if self.name != None and copy.name != None:
            self.name = "%s + %s" % (self.name, copy.name)
        elif self.size == 0 and copy.name != None:
            self.name = copy.name

        # Update the size
        self.update_size()


    def copy(self):
        """
        Return a copy of self.
        """
        nodes = {n : self.nodes[n] for n in self.nodes}
        edges = {e : self.edges[e] for e in self.edges}
        graph = Graph(nodes, edges)
        graph.name = self.name
        return graph


    def components(self):
        """
        Get the connected components of the graph (forgetting the direction of
        the edges.) Returns a list of the components where each element is a
        Graph.
        """
        # If self happens to be the empty Graph, return that alone.
        if len(self.nodes) == 0:
            return [self.copy()]

        # Otherwise, let's build a copy of self so that we can keep deleting
        # the connected components until it is empty.
        graph = self.copy()

        # To return. This will store the connected components as Graph objects.
        comp = []

        # Each iteration will find a connected component, delete it from 'graph'
        # and append it to 'comp'.
        while(graph.size > 0):
            # Start by taking an arbitrary node of graph. We will find its
            # connected component.
            for n in graph.nodes:
                break

            # Set 'A' will store all the nodes in the component
            A = set([n])

            # Set 'B' is an intermediate variable.
            B = set([n])

            # Set 'C' will be the new nodes that we find by connecting edges.
            C = set()

            # All the edges in the component.
            ed = {}

            while(len(B) > 0):
                # Take an element of 'B' and look for all the nodes connected to
                # it.
                for b in B:
                    for e in graph.edges:
                        # When a node is found, store it in 'C'.
                        if e[0] == b:
                            C.add(e[1])
                            ed[e] = graph.edges[e]
                        if e[1] == b:
                            C.add(e[0])
                            ed[e] = graph.edges[e]
                B = C - A # Keep only the new nodes for the next iteration.
                A |= C # Add all the nodes we found to A.
            # The loop ends when an iteration didn't find any new node.

            # At this point all the nodes in the connected component are stored
            # in 'A', and all the edges in 'ed'.

            # Make a dictionary of all the nodes in the component with their
            # weight.
            nd = {n : graph.nodes[n] for n in A}

            # Make a Graph out of that and add it to the list 'comp'.
            comp.append(Graph(nd, ed))

            # Delete the nodes and edges of that component.
            for n in A:
                graph.nodes.pop(n)
            for e in ed:
                graph.edges.pop(e)
            graph.update_size()

        return comp

    def relabel_map(self, p):
        """
        Returns a new graph but with the label of the nodes changed according
        to 'p', where 'p' is a map from the set of keys of self.nodes to a new
        set of labels.
        """
        graph = self.copy()
        graph.nodes = {p[n] : graph.nodes[n] for n in graph.nodes}
        graph.edges = {(p[e[0]], p[e[1]]) : graph.edges[e] for e in graph.edges}
        return graph


    def equals(self, g):
        """
        Check if 'self' is equal to the graph g forgetting the direction of
        the edges.
        """
        if self.nodes != g.nodes:
            return False
        for e in self.edges:
            er = (e[0], e[1])
            el = (e[1], e[0])
            if er in g.edges:
                if g.edges[er] != self.edges[e]:
                    return False
            elif el in g.edges:
                if g.edges[el] != self.edges[e]:
                    return False
            else:
                return False

        return True

    def edges_connected(self, n):
        """
        Number of edges connected to node 'n'
        """
        k = 0
        for e in self.edges:
            if e[0] == n or e[1] == n:
                k += 1
        return k

    def max_n_edges(self):
        """
        Maximum number of edges connected to a single node.
        """
        m = 0
        for n in self.nodes:
            k = self.edges_connected(n)
            print(k)
            if k > m:
                m = k
        return k

    def sum_nodes(self):
        """
        Total of the node weights
        """
        t = 0
        for n in self.nodes:
            t += self.nodes[n]
        return t

    def hasse_reduce(self):
        """
        Computes the Hasse diagram associated to a partial order. In other
        words, it assumes that the directed graph represents a partial order
        and removes the edges that can be deduced from transitivity.
        """
        ref_edges = {}
        for a in self.edges:
            do_we_keep_that_edge = True
            for i in self.nodes:
                if i == a[0] or i == a[1]:
                    continue
                if (a[0], i) in self.edges and (i, a[1]) in self.edges:
                    do_we_keep_that_edge = False
                    break
            if do_we_keep_that_edge:
                ref_edges[a] = self.edges[a]
        self.edges = ref_edges

    def latex_hasse(self, v=5):
        """
        Outputs a LaTeX version of the Hasse diagram with nodes as the labels.
        """
        arr = []
        for n in self.nodes:
            u = random.uniform(0, v)
            arr.append((n, u - v / 2, self.nodes[n][1], name_simp(self.nodes[n][0]), self.nodes[n][2]))

        arr.sort(key=lambda x : x[2])

        k = 0
        ls = []
        for i in range(len(arr)):
            if i > 0 and arr[i][2] > arr[i-1][2]:
                k += 1
            ls.append((arr[i][0], arr[i][1], k, arr[i][4], arr[i][3]))
        ls = ls[::-1]

        print("$$")
        print("\\begin{tikzpicture}\\tiny")
        for i in ls:
            if i[4] == '0':
                print("\\node (%d) at (%.2f, %d) {$%d$};" %
                        (i[0], i[1], i[2], i[3]))
            elif i[3] > 1:
                print("\\node (%d) at (%.2f, %d) {$%d%s$};" % i)
            else:
                print("\\node (%d) at (%.2f, %d) {$%s$};" %
                        (i[0], i[1], i[2], i[4]))
        for e in self.edges:
            print("\\draw (%d) -- (%d);" % (e[0], e[1]))
        print("\\end{tikzpicture}")
        print("$$")


def name_simp(s):
    """
    Takes a string "A1 + A1 + A2 + B3" and return "A_1^2A_2B_3"
    """
    if s == "0":
        return s
    arr = s.split(" + ")
    toret = ""
    i = 0
    while(i < len(arr)):
        k = 0
        while(i + k < len(arr) and arr[i] == arr[i + k]):
            k += 1
        toret += arr[i][0]
        if k == 1:
            toret += "_%d" % int(arr[i][1])
        else:
            toret += "_%d^%d" % (int(arr[i][1]), k)
        i += k
    return toret

