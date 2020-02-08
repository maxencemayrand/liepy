from .graph import Graph
from .cartanmatrix import CartanMatrix
import random
import itertools
import re

class DynkinDiagram(Graph):
    """
    Dynkin diagram object.
    """
    def __init__(self, type=None, nodes={}, edges={}):
        """
        Can be initialized by the type of the Dynkin diagram, 
        which is a string such as 'A3 + 4F5'
        """
        if type:
            nodes, edges = get_nodes_and_edges(type)
            
        super(DynkinDiagram, self).__init__(nodes, edges, type)

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
        dyn = DynkinDiagram(nodes=nodes, edges=edges)
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
        for perm in itertools.permutations(range(graph.size)):
            p = {i : perm[i] for i in range(graph.size)}
            test = graph.relabel_map(p)
            for s in simple:
                if test.equals(s):
                    return s.name

        raise ValueError("Not a simple Dynkin diagram")

    def latex(self, l):
        """
        Print on the terminal a LaTeX version of the DynkinDiagram.
        """
        print("$$")
        print("\\begin{tikzpicture}")
        print("\\tikzset{cir/.style={draw, circle, inner sep=2pt},}")
        print("\\tikzset{dot/.style={draw, circle, fill, inner sep=2pt},}")

        m = min(self.nodes[n] for n in self.nodes)
        for n in self.nodes:
            if self.nodes[n] > m:
                print("\\node[dot] (%r) at (%f, %f) {};" % (n,
                    random.uniform(0, l), random.uniform(0, l)))
            else:
                print("\\node[cir] (%r) at (%f, %f) {};" % (n,
                    random.uniform(0, l), random.uniform(0, l)))
        for e in self.edges:
            if self.edges[e] == 1:
                print("\\draw (%r) -- (%r);" % (e[0], e[1]))
            if self.edges[e] == 2:
                print("\\draw[double, double distance=2pt] (%r) -- (%r);" % (
                    e[0], e[1]))
            if self.edges[e] == 3:
                print("\\draw[double, double distance=2pt] (%r) -- (%r);" % (
                    e[0], e[1]))
                print("\\draw (%r) -- (%r);" % (e[0], e[1]))
        print("\\end{tikzpicture}")
        print("$$")

    def components(self):
        """
        Return a list of the connected components as DynkinDiagram objects.
        """
        comp = super(DynkinDiagram, self).components()
        return [DynkinDiagram(nodes=c.nodes, edges=c.edges) for c in comp]

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

def get_nodes_and_edges_single(type, rank):
    """Get nodes and edges of the Dynkin diagram of a Lie algebra of this type and rank"""
    if type == 'A':
        nodes = {i : 1 for i in range(rank)}
        edges = {(i, i+1) : 1 for i in range(rank - 1)}
        return nodes, edges
    if type == 'B':
        nodes = {i : 2 for i in range(1, rank)}
        nodes[0] = 1
        edges = {(i, i+1) : 1 for i in range(1, rank - 1)}
        edges[(0, 1)] = 2
        return nodes, edges
    if type == 'C':
        nodes = {i : 1 for i in range(rank - 1)}
        nodes[rank - 1] = 2
        edges = {(i, i+1) : 1 for i in range(rank - 2)}
        edges[(rank - 2, rank - 1)] = 2
        return nodes, edges
    if type == 'D':
        if rank >= 4:
            nodes = {i : 1 for i in range(rank)}
            edges = {(i, i+1) : 1 for i in range(rank - 3)}
            edges[(rank - 3, rank - 2)] = 1
            edges[(rank - 3, rank - 1)] = 1
        else:
            raise ValueError("Incorrect value for the Dynkin diagram of type D.")
        return nodes, edges
    if type == 'E':
        if rank == 6:
            nodes = {i : 1 for i in range(6)}
            edges = {
                (0, 1) : 1,
                (1, 2) : 1,
                (2, 3) : 1,
                (3, 4) : 1,
                (2, 5) : 1
            }
        elif rank == 7:
            nodes = {i : 1 for i in range(7)}
            edges = {
                (0, 1) : 1,
                (1, 2) : 1,
                (2, 3) : 1,
                (3, 4) : 1,
                (4, 5) : 1,
                (2, 6) : 1
            }
        elif rank == 8:
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
        else:
            raise ValueError("Incorrect value for the Dynkin diagram of type E.")
        return nodes, edges
    if type == 'F':
        if rank == 4:
            nodes = {0 : 1, 1 : 1, 2 : 2, 3 : 2}
            edges = {
                (0, 1) : 1,
                (1, 2) : 2,
                (2, 3) : 1
            }
        else:
            raise ValueError("Incorrect value for the Dynkin diagram of type F.")
        return nodes, edges
    if type == 'G':
        if rank == 2:
            nodes = {0 : 1, 1: 3}
            edges = {
                (0, 1) : 3
            }
        else:
            raise ValueError("Incorrect value for the Dynkin diagram of type G.")
        return nodes, edges

def parsetype(type_str):
    """type_string is, e.g. 'A3 + 2F4', and it returns [('A', 3), ('F', 4), ('F', 4)]"""
    types = [t.strip() for t in type_str.split('+')]
    comps = []
    for s in types:
        if s[0].isalpha():
            s = '1' + s
        s = re.split(r'(\d+)', s) 
        m = int(s[1])
        t = s[2].capitalize()
        n = int(s[3])
        for _ in range(m):
            comps.append((t, n))
    return comps

def get_nodes_and_edges(type_str):
    types = parsetype(type_str)
    g = Graph()
    for type, rank in types:
        g.add(Graph(*get_nodes_and_edges_single(type, rank)))
    return g.nodes, g.edges

def simple_diagrams(r):
    """
    Return a list of all simple Dynkin diagrams of rank 'r' as Graph objects.
    """
    if r == 1:
        dyn1 = DynkinDiagram('A1')
        ls = [dyn1]
    if r == 2:
        dyn1 = DynkinDiagram('A2')
        dyn2 = DynkinDiagram('B2')
        dyn3 = DynkinDiagram('G2')
        ls = [dyn1, dyn2, dyn3]
    if r == 3:
        dyn1 = DynkinDiagram('A3')
        dyn2 = DynkinDiagram('B3')
        dyn3 = DynkinDiagram('C3')
        ls = [dyn1, dyn2, dyn3]
    if r >= 4:
        dyn1 = DynkinDiagram(f'A{r}')
        dyn2 = DynkinDiagram(f'B{r}')
        dyn3 = DynkinDiagram(f'C{r}')
        dyn4 = DynkinDiagram(f'D{r}')
        if r == 4:
            dyn5 = DynkinDiagram('F4')
            ls = [dyn1, dyn2, dyn3, dyn4, dyn5]
        elif r == 5:
            ls = [dyn1, dyn2, dyn3, dyn4]
        elif r >= 6 and r <= 8:
            dyn5 = DynkinDiagram(f'E{r}')
            ls = [dyn1, dyn2, dyn3, dyn4, dyn5]
        else:
            ls = [dyn1, dyn2, dyn3, dyn4]

    return ls


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
