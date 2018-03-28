import random
import itertools as it


class Graph(object):
    """
    Data structure for finite directed graphs with weights on nodes and edges.
    The set of nodes and their weights is implemented by a dictionary. The set
    of edges is implemented by a dictionary: each key is a 2-tuple of integers,
    where a tuple (i, j) means there is an edge from node i to node j (in the
    odering given by the list), and the value of the key is the weight of that
    edge.
    """
    def __init__(self, nodes={}, edges={}):
        self.name  = None   # string: name of the graph.
        self.nodes = nodes  # Dictionary of nodes (values are weights).
        self.edges = edges  # Dictionary of edges (2-tuples with weights).
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
        print "=" * 50
        print
        print "Name: %s" % self.name # Name of graph.
        print
        print "%d nodes." % self.size # Number of nodes.
        print "%d edges." % len(self.edges) # Number of edges.
        print
        print "Nodes and weights:" # List of the nodes and their weight.
        for n in self.nodes:
            print "%r : %r" % (n, self.nodes[n])
        print
        print "Edges and weights:" # List of the edges and their weight.
        for e in self.edges:
            print "%r --> %r  :  %r" % (e[0], e[1], self.edges[e])
        print
        print "=" * 50

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
            print k
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

    def latex_hasse(self, v):
        """
        Outputs a LaTeX version of the Hasse diagram.
        """
        arr = []
        for n in self.nodes:
            u = random.uniform(0, v)
            arr.append((n, u, self.nodes[n][1], name_simp(self.nodes[n][0])))

        arr.sort(key=lambda x : x[2])

        k = 0
        ls = []
        for i in range(len(arr)):
            if i > 0 and arr[i][2] > arr[i-1][2]:
                k += 1
            ls.append((arr[i][0], arr[i][1], k, arr[i][3]))
        ls = ls[::-1]

        print "$$"
        print "\\begin{tikzpicture}"
        print "\\tikzset{dot/.style={draw, circle, fill, inner sep=1pt},}"
        for i in ls:
            print "\\node[dot] (%d) at \t(%.2f, %d) {};" % (i[0], i[1], i[2])
            print "\\node[left] at \t\t(%.2f, %d) {$%s$};" % (i[1], i[2], i[3])
        for e in self.edges:
            print "\\draw (%d) -- (%d);" % (e[0], e[1])
        print "\\end{tikzpicture}"
        print "$$"

    def latex_hasse2(self, v):
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

        print "$$"
        print "\\begin{tikzpicture}\\tiny"
        for i in ls:
            if i[4] == '0':
                print "\\node (%d) at (%.2f, %d) {$%d$};" % (i[0], i[1], i[2], i[3])
            elif i[3] > 1:
                print "\\node (%d) at (%.2f, %d) {$%d%s$};" % i
            else:
                print "\\node (%d) at (%.2f, %d) {$%s$};" % (i[0], i[1], i[2], i[4])
        for e in self.edges:
            print "\\draw (%d) -- (%d);" % (e[0], e[1])
        print "\\end{tikzpicture}"
        print "$$"


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



# Sizes of Weyl groups.
def fac(n):
    if n == 0:
        return 1
    return n * fac(n - 1)
def W_A(n):
    return fac(n + 1)
def W_B(n):
    return fac(n) * (2 ** n)
def W_C(n):
    return fac(n) * (2 ** n)
def W_D(n):
    return fac(n) * (2 ** (n - 1))
def W_E(n):
    if n == 6:
        return (2**7) * (3**4) * 5
    if n == 7:
        return (2**10) * (3**4) * 5 * 7
    if n == 8:
        return (2**14) * (3**5) * (5**2) * 7
def W_F(n):
    if n == 4:
        return (2**7) * (3**2)
def W_G(n):
    if n == 2:
        return (2**2) * 3



class DynkinDiagram(Graph):
    """
    Dynkin diagram object.
    """
    def __init__(self, nodes={}, edges={}):
        super(DynkinDiagram, self).__init__(nodes, edges)

    def identify(self):
        """
        Returns the name of the Dynkin diagram, e.g A3 + F4
        """
        # If it already has a name, then let's just return it.
        if self.name != None:
            return self.name

        # If it is empty, then it's the '0' diagram.
        if self.size == 0:
            return "0"

        # First break the diagram into its connected components.
        comp = self.components()

        # Will store the names of each component.
        names = []

        # For each component compute its name and add it to the list.
        for c in comp:
            names.append(c.simple_diagram_name())

        # Sort in alphabetic order
        names.sort()

        # Put '+'s in between the names
        name = ""
        for n in names[:-1]:
            name += n + " + "
        name += names[-1]

        self.name = name
        return name

    def weyl_size(self):
        if self.name == None:
            self.identify()

        if self.size == 0:
            return 1

        ls = self.name.split(" + ")
        w = 1
        for s in ls:
            letter = s[0]
            rank = int(s[1])
            if letter == 'A':
                w *= W_A(rank)
            if letter == 'B':
                w *= W_B(rank)
            if letter == 'C':
                w *= W_C(rank)
            if letter == 'D':
                w *= W_D(rank)
            if letter == 'E':
                w *= W_E(rank)
            if letter == 'F':
                w *= W_F(rank)
            if letter == 'G':
                w *= W_G(rank)
        return w

    def identify2(self):
        # If it already has a name, then let's just return it.
        if self.name != None:
            return self.name

        # If it is empty, then it's the '0' diagram.
        if self.size == 0:
            return "0"

        # First break the diagram into its connected components.
        comp = self.components()

        # Will store the names of each component.
        names = []

        # For each component compute its name and add it to the list.
        for c in comp:
            names.append(c.simple_diagram_name2())

        # Sort in alphabetic order
        names.sort()

        # Put '+'s in between the names
        name = ""
        for n in names[:-1]:
            name += n + " + "
        name += names[-1]

        self.name = name
        return name

    def normalize(self):
        """
        Normalize the weights of the nodes so that the minimum is 1.
        """
        m = min([self.nodes[n] for n in self.nodes])
        for n in self.nodes:
            self.nodes[n] /= m

    def copy(self):
        """
        Return a copy of self
        """
        nodes = {n : self.nodes[n] for n in self.nodes}
        edges = {e : self.edges[e] for e in self.edges}
        dyn = DynkinDiagram(nodes, edges)
        dyn.name = self.name
        return dyn


    def simple_diagram_name(self):
        """
        Assumes that self is a simple Dynkin diagram and finds which one.
        Returns the name of the diagram, e.g. returns "D4".
        """
        graph = self.copy() # Make a copy, because we will:
        graph.relabel(0)    # relabel the nodes in the same way as the simple DD
        graph.normalize()   # and renormalize it.

        # List of all simple Dynkin diagrams of the same rank.
        simple = simple_diagrams(graph.size)

        # Make all permutations of the nodes and test if it matches one of the
        # simple Dynkin diagram.
        for perm in it.permutations(range(graph.size)):
            p = {i : perm[i] for i in range(graph.size)}
            test = graph.relabel_map(p)
            for s in simple:
                if test.equals(s):
                    return s.name

        raise ValueError("Not a simple Dynkin diagram")

    def simple_diagram_name2(self):
        graph = self.copy() # Make a copy, because we will:
        graph.relabel(0)    # relabel the nodes in the same way as the simple DD
        graph.normalize()   # and renormalize it.

        rk = graph.size
        m = max(graph.edges.values())
        s = graph.sum_nodes()
        t = graph.max_n_edges()

        if rk == 1:
            return "A1"
        elif rk == 2:
            if m == 1:
                return "A2"
            if m == 2:
                return "B2"
            if m == 3:
                return "G2"
        elif rk == 3:
            if m == 1:
                return "A3"
            if m == 2: # B vs C
                if s == 1 + 2 * (rk - 1):
                    return "B%d" % rk
                if s == rk + 1:
                    return "C%d" % rk
        elif rk == 4:
            if m == 1: # A vs D
                if t == 2:
                    return "A%d" % rk
                if t == 3:
                    return "D%d" % rk
            if m == 2: # B vs C vs F
                if s == 1 + 2 * (rk - 1):
                    return "B%d" % rk
                if s == rk + 1:
                    return "C%d" % rk
                if s == 6:
                    return "F4"
        elif rk == 5:
            if m == 1: # A vs D
                if t == 2:
                    return "A%d" % rk
                if t == 3:
                    return "D%d" % rk
            if m == 2: # B vs C
                if s == 1 + 2 * (rk - 1):
                    return "B%d" % rk
                if s == rk + 1:
                    return "C%d" % rk
        elif 6 <= rk <= 8:
            if m == 1: # A vs D vs E
                if t == 1:
                    return "A%d" % rk
                if t == 2: # D vs E
                    # Get the node with three edges
                    for n in graph.nodes:
                        if graph.edges_connected(n) == 3:
                            break
                    # Get the neighborhing nodes.
                    neinodes = []
                    for e in graph.edges:
                        if e[0] == n:
                            neinodes.append(e[1])
                        elif e[1] == n:
                            neinodes.append(e[0])
                    r = 0
                    for q in neinodes:
                        r += graph.edges_connected(q)
                    if r == 4:
                        return "D%d" % rk
                    if r == 5:
                        return "E%d" % rk
            if m == 2: # B vs C
                if s == 1 + 2 * (rk - 1):
                    return "B%d" % rk
                if s == rk + 1:
                    return "C%d" % rk
        elif rk > 8:
            if m == 1: # A vs D
                if t == 2:
                    return "A%d" % rk
                if t == 3:
                    return "D%d" % rk
            if m == 2: # B vs C
                if s == 1 + 2 * (rk - 1):
                    return "B%d" % rk
                if s == rk + 1:
                    return "C%d" % rk

        raise ValueError("Not a simple Dynkin diagram")

    def latex(self, l):
        """
        Print on the terminal a LaTeX version of the DynkinDiagram.
        """
        print "$$"
        print "\\begin{tikzpicture}"
        print "\\tikzset{cir/.style={draw, circle, inner sep=2pt},}"
        print "\\tikzset{dot/.style={draw, circle, fill, inner sep=2pt},}"

        m = min(self.nodes[n] for n in self.nodes)
        for n in self.nodes:
            if self.nodes[n] > m:
                print "\\node[dot] (%r) at (%f, %f) {};" % (n,
                    random.uniform(0, l), random.uniform(0, l))
            else:
                print "\\node[cir] (%r) at (%f, %f) {};" % (n,
                    random.uniform(0, l), random.uniform(0, l))
        for e in self.edges:
            if self.edges[e] == 1:
                print "\\draw (%r) -- (%r);" % (e[0], e[1])
            if self.edges[e] == 2:
                print "\\draw[double, double distance=2pt] (%r) -- (%r);" % (
                    e[0], e[1])
            if self.edges[e] == 3:
                print "\\draw[double, double distance=2pt] (%r) -- (%r);" % (
                    e[0], e[1])
                print "\\draw (%r) -- (%r);" % (e[0], e[1])
        print "\\end{tikzpicture}"
        print "$$"

    def components(self):
        """
        Return a list of the connected components as DynkinDiagram objects.
        """
        comp = super(DynkinDiagram, self).components()
        return [DynkinDiagram(c.nodes, c.edges) for c in comp]

    def n_edges(self, m, n):
        """
        Returns the number of edges between node 'm' and node 'n'.
        """
        if (m, n) in self.edges:
            return self.edges[(m, n)]
        if (n, m) in self.edges:
            return self.edges[(n, m)]
        return 0

    def cartan(self):
        """
        Return the Cartan matrix as a dictionary which associates an integer
        # to each pair of nodes.
        """

        A = {} # To return

        # Loop over all pair of nodes
        for m in self.nodes:
            for n in self.nodes:

                # The entries of the diagonal are always equal to 2
                if m == n:
                    A[(m, n)] = 2

                else:
                    # Number of edges connecting those nodes.
                    e = self.n_edges(m, n)

                    # Compute the values of the Cartan matrix for the pairs
                    # (m, n) and (n, m).
                    if e == 0:
                        A[(m, n)] = 0
                        A[(n, m)] = 0
                    elif self.nodes[m] < self.nodes[n]:
                        A[(m, n)] = -e
                        A[(n, m)] = -1
                    else:
                        A[(m, n)] = -1
                        A[(n, m)] = -e

        return CartanMatrix(self.nodes, A)


class A(DynkinDiagram):
    """
    Simple Dynkin diagram of type A
    """
    def __init__(self, n):
        nodes = {i : 1 for i in range(n)}
        edges = {(i, i+1) : 1 for i in range(n-1)}
        super(A, self).__init__(nodes, edges)
        self.name = "A%d" % n
class B(DynkinDiagram):
    """
    Simple Dynkin diagram of type B
    """
    def __init__(self, n):
        nodes = {i : 2 for i in range(1, n)}
        nodes[0] = 1
        edges = {(i, i+1) : 1 for i in range(1, n-1)}
        edges[(0, 1)] = 2
        super(B, self).__init__(nodes, edges)
        self.name = "B%d" % n
class C(DynkinDiagram):
    """
    Simple Dynkin diagram of type C
    """
    def __init__(self, n):
        nodes = {i : 1 for i in range(n-1)}
        nodes[n-1] = 2
        edges = {(i, i+1) : 1 for i in range(n-2)}
        edges[(n-2, n-1)] = 2
        super(C, self).__init__(nodes, edges)
        self.name = "C%d" % n
class D(DynkinDiagram):
    """
    Simple Dynkin diagram of type D
    """
    def __init__(self, n):
        if n >= 4:
            nodes = {i : 1 for i in range(n)}
            edges = {(i, i+1) : 1 for i in range(n-3)}
            edges[(n-3, n-2)] = 1
            edges[(n-3, n-1)] = 1
            super(D, self).__init__(nodes, edges)
            self.name = "D%d" % n
        else:
            raise ValueError(
                "Incorrect value for the Dynkin diagram of type D."
            )
class E(DynkinDiagram):
    """
    Simple Dynkin diagram of type E
    """
    def __init__(self, n):
        if n == 6:
            nodes = {i : 1 for i in range(6)}
            edges = {
                (0, 1) : 1,
                (1, 2) : 1,
                (2, 3) : 1,
                (3, 4) : 1,
                (2, 5) : 1
            }
            super(E, self).__init__(nodes, edges)
            self.name = "E%d" % n
        elif n == 7:
            nodes = {i : 1 for i in range(7)}
            edges = {
                (0, 1) : 1,
                (1, 2) : 1,
                (2, 3) : 1,
                (3, 4) : 1,
                (4, 5) : 1,
                (2, 6) : 1
            }
            super(E, self).__init__(nodes, edges)
            self.name = "E%d" % n
        elif n == 8:
            nodes = {i : 1 for i in range(8)}
            edges = {
                (0, 2) : 1,
                (2, 3) : 1,
                (3, 4) : 1,
                (4, 5) : 1,
                (5, 6) : 1,
                (6, 7) : 1,
                (1, 3) : 1
            }
            super(E, self).__init__(nodes, edges)
            self.name = "E%d" % n
        else:
            raise ValueError(
                "Incorrect value for the Dynkin diagram of type E."
            )
class F(DynkinDiagram):
    """
    Simple Dynkin diagram of type F
    """
    def __init__(self, n):
        if n == 4:
            nodes = {0 : 1, 1 : 1, 2 : 2, 3 : 2}
            edges = {
                (0, 1) : 1,
                (1, 2) : 2,
                (2, 3) : 1
            }
            super(F, self).__init__(nodes, edges)
            self.name = "F%d" % n
        else:
            raise ValueError(
                "Incorrect value for the Dynkin diagram of type F."
            )
class G(DynkinDiagram):
    """
    Simple Dynkin diagram of type G
    """
    def __init__(self, n):
        if n == 2:
            nodes = {0 : 1, 1: 3}
            edges = {
                (0, 1) : 3
            }
            super(G, self).__init__(nodes, edges)
            self.name = "G%d" % n
        else:
            raise ValueError(
                "Incorrect value for the Dynkin diagram of type G."
            )


def simple_diagrams(r):
    """
    Return a list of all simple Dynkin diagrams of rank 'r' as Graph objects.
    """
    if r == 1:
        dyn1 = A(1)
        ls = [dyn1]
    if r == 2:
        dyn1 = A(2)
        dyn2 = B(2)
        dyn3 = G(2)
        ls = [dyn1, dyn2, dyn3]
    if r == 3:
        dyn1 = A(3)
        dyn2 = B(3)
        dyn3 = C(3)
        ls = [dyn1, dyn2, dyn3]
    if r >= 4:
        dyn1 = A(r)
        dyn2 = B(r)
        dyn3 = C(r)
        dyn4 = D(r)
        if r == 4:
            dyn5 = F(4)
            ls = [dyn1, dyn2, dyn3, dyn4, dyn5]
        elif r == 5:
            ls = [dyn1, dyn2, dyn3, dyn4]
        elif r >= 6 and r <= 8:
            dyn5 = E(r)
            ls = [dyn1, dyn2, dyn3, dyn4, dyn5]
        else:
            ls = [dyn1, dyn2, dyn3, dyn4]

    return ls

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
        print "+" + "-" * len(self.nodes) * 4 + "---+"
        for m in self.nodes:
            print "|",
            for n in self.nodes:
                print "%3d" % self.matrix[(m, n)],
            print "  |"
        print "+" + "-" * len(self.nodes) * 4 + "---+"
