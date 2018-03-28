from graph import *
from math import *
import itertools as it


class RootSystem(object):
    """
    Abstracly, a RootSystem is a finite subset of R^n.
    Concretely, it is a finite set (the 'roots') together with a number
    associated to each pair of roots, i.e. their inner-product.
    We implement the set of roots as a 'set' object of tuples of integers,
    and their inner-product as a dictionary whose keys are pair of roots and
    whose values are integers.
    """
    def __init__(self, rts=None, prd=None):

        # The two main variables:

        # A 'set' object containing the roots as 'tuples of integers'.
        self.rts = rts

        # A dictionary giving an integer for each pair of roots,
        # i.e. their inner-product.
        self.prd = prd


        # Auxiliary variables used for computations:

        # A set of positive roots.
        self.p_rts = None
        # The corresponding set of simple roots.
        self.s_rts = None
        # The rank of the root system.
        self.rank = None
        # The Weyl group. It is implemented as a list of dictionaries. Each
        # element 'w' of 'self.weyl' is a dictionary whose keys are all the
        # roots (i.e. 'self.rts') and whose values are also roots, thus giving
        # a map from the set of roots to itself.
        self.weyl = None

    def get_p_rts(self):
        """
        Get a subset of postive roots.
        """
        print "Get positive roots."
        # We use Knapp's method (p.155). That is, first fix any ordering of the
        # set of all roots, say r1, r2, r3, ..., rk. Then, a root 'a' is
        # positive if there exists i such that <a, rj> = 0 for 1 <= j < i and
        # <a, ri> >0.

        # Order the roots:
        ordered_rts = {}
        i = 0
        for r in self.rts:
            ordered_rts[i] = r
            i += 1

        # Compute the positive roots
        self.p_rts = set()
        for r in self.rts:
            for i in range(len(self.rts)):
                p = self.prd[(r, ordered_rts[i])]
                if p != 0:
                    break
            if p > 0:
                self.p_rts.add(r)
        print "Done with positive roots."

    def is_simple(self, r):
        """
        Check if a positive root is simple according to the given notion of
        positivity.
        """
        for a in self.p_rts:
            if add_tuple(r, mul_tuple(-1, a)) in self.p_rts:
                return False
        return True

    def get_s_rts(self):
        """
        Get a set of simple roots.
        """
        print "Get simple roots."
        # Get a subset of positive roots if not already defined.
        if self.p_rts == None:
            self.get_p_rts()

        self.s_rts = set()
        # Go over each positive root and check if it is simple using 'is_simple'
        for r in self.p_rts:
            if self.is_simple(r):
                self.s_rts.add(r)

        # Update the rank
        self.rank = len(self.s_rts)
        print "Done with simple roots."

    def root_reflection(self, a):
        """
        Return the root reflection of the root 'r' as a dictionary (as in the
        Weyl group). This method is used to construct the Weyl group, since it
        is generated by the root reflections of the simple roots.
        """
        dic = {}
        for b in self.rts:
            # The formula is b |--> b + coef * a, where
            # coef = - 2 <b, a> / |a|^2
            coef = - 2 * self.prd[(b, a)] / self.prd[(a, a)]
            dic[b] = add_tuple(b, mul_tuple(coef, a))

        return dic

    def compose(self, p, q):
        """
        Compose two elements of the Weyl group.
        """
        pq = {} # Will be the product of p and q
        for r in self.rts:
            pq[r] = p[q[r]]
        return pq

    def get_weyl(self):
        """
        Get the Weyl group in 'self.weyl'
        """
        print "Get Weyl group."
        # The methods needs a choice of simple roots, so get one if not
        # already done.
        if self.s_rts == None:
            self.get_s_rts()

        # The simple reflections.
        simp_refl = [self.root_reflection(r) for r in self.s_rts]

        # First place all the simple reflections.
        self.weyl = [v for v in simp_refl]

        # At each iteration we multiply every element of the list 'self.weyl'
        # and add them. The loop stops when such an iteration doesn't produce
        # any new element.
        change = True
        while change:
            change = False
            for v in simp_refl:
                temp = []
                for w in self.weyl:
                    u = self.compose(w, v)
                    if u not in self.weyl:
                        temp.append(u)
                        change = True
                self.weyl += temp

        print "Done with Weyl group."
        return self.weyl

    def init_from_dynkin(self, dyn):
        """
        Initiate from a 'DynkinDiagram' object 'dyn'.
        It computes all the variables of __init__.
        """
        print "Initiate from Dynkin diagram."
        # First compute the Cartan matrix
        A = dyn.cartan()

        size = len(A.nodes)
        # Fix an ordering of the nodes of A.
        # In other words, we create a dictionary 'n' with keys 0, 1, ..., size
        # such that nd[i] is the i'th node of 'A.nodes'.
        nd = {}
        i = 0
        for n in A.nodes:
            nd[i] = n
            i += 1

        self.s_rts = set([basic_tuple(size, i) for i in range(size)])

        # The rank is the number of simple roots:
        self.rank = len(self.s_rts)

        Phi = self.s_rts.copy() # Will store the positive roots.

        # The following loop will find the positive roots by induction on the
        # level.

        # Will become false when at a given level we didn't find any root
        # of the next level. Then, the loop will stop.
        change = True
        # Stores the roots of the level considered in a given iteration.
        roots = Phi.copy()
        while(change == True):
            change = False
            newrts = set() # Will store the new roots found in this iteration.

            # For each root in 'roots', check which simple root can be added
            # to it so that we get a new root of the next level.
            for b in roots:
                for i in range(size):
                    # 'a' is the ith simple root according to the ordering of
                    # the nodes given by 'nd'.
                    a = basic_tuple(size, i)

                    # Compute the integers p and q of the a-string through b.
                    p = 1
                    while(add_tuple(b, mul_tuple(-p, a)) in Phi):
                        p += 1
                    p -= 1
                    q = p
                    for j in range(size):
                        q -= b[j] * A.matrix[(nd[i], nd[j])]

                    # If q > 0 then a + b is a root (and is of the next level).
                    if q > 0:
                        newrts.add(add_tuple(a, b))
                        change = True

            # At this point 'newrts' contains all the roots of the next level.
            # Hence we set 'roots' to 'newrts' for the next iteration.
            roots = newrts

            # Add the new roots to the set of all positive roots.
            Phi |= newrts

        # At this point we have all the positive roots stored in 'Phi'
        # so we set them to the variable 'self.p_rts'.
        self.p_rts = Phi.copy()

        # Now we get all roots:
        self.rts = set()
        for a in Phi:
            self.rts.add(a)
            self.rts.add(mul_tuple(-1, a))

        # Finally compute the products:
        self.prd = {}
        for a in self.rts:
            for b in self.rts:
                p = 0
                for i in range(size):
                    for j in range(size):
                        p += (a[i] * b[j] * dyn.nodes[nd[i]] *
                                A.matrix[(nd[i], nd[j])])
                self.prd[(a, b)] = p
                # (We should have divided p by 2, but we didn't so
                # that everything stays in integers. The product we have
                # is just a scaler multiple of the one determined by the
                # Dynkin diagram, so it doesn't matter.)
        print "Done with initiate from Dynkin diagram."

    def dynkin(self):
        """
        Return the corresponding 'DynkinDiagram' object.
        """
        if self.s_rts == None:
            self.get_s_rts()

        # Choose an arbitrary ordering of the simple roots:
        sr = {}
        i = 0
        for r in self.s_rts:
            sr[i] = r
            i += 1

        # The nodes and their weights:
        nd = {i : self.prd[(sr[i], sr[i])] for i in range(self.rank)}

        ed = {}
        for i in range(self.rank):
            for j in range(i+1, self.rank):
                ned = 4
                ned *= self.prd[(sr[i], sr[j])] ** 2
                ned /= self.prd[(sr[i], sr[i])]
                ned /= self.prd[(sr[j], sr[j])]
                if ned != 0:
                    ed[(i, j)] = ned

        dyn = DynkinDiagram(nd, ed)
        return dyn

    def dim(self):
        """
        Dimension of the corresponding semisimple Lie algebra
        """
        if self.s_rts == None:
            self.get_s_rts()
        return len(self.s_rts) + len(self.rts)

    def is_subsystem(self, Psi):
        """
        Check if the frozenset of positive roots 'Psi' defines a closed
        subsystem
        """
        for a in Psi:
            for b in Psi:
                c = add_tuple(a, b)
                if c in self.p_rts and c not in Psi:
                    return False
                c = add_tuple(a, mul_tuple(-1, b))
                if c in self.p_rts and c not in Psi:
                    return False
        return True

    def subsystems(self):
        """
        Return the set of all symmetric closed subsystems of 'self' as a set of
        'frozenset's of roots in 'self.rts'.
        """
        print "Compute all subsystems."
        # We can shorten the computation if we have a set of positive roots.
        if self.p_rts == None:
            self.get_p_rts()

        # start with the empty frozenset
        subsys = set([frozenset()])

        # Loop over all possible subset of the set of positive roots and check
        # if it defines a closed subsystem.
        for k in range(1, len(self.p_rts) + 1):
            for Psi in [frozenset(c) for c in it.combinations(self.p_rts, k)]:
                if self.is_subsystem(Psi):
                    # Add the negative roots.
                    Psi_all = set(Psi)
                    for r in Psi:
                        Psi_all.add(mul_tuple(-1, r))
                    subsys.add(frozenset(Psi_all))

        # At this point, each element of the set 'subsys' consists of a
        # frozenset of rts which is a symmetric closed system.
        print "Done with subsystems."
        return subsys

    def root_subsystems(self):
        """
        Convert the frozensets returned by 'subsystems' into 'RootSystem'
        objects.
        """
        # First get the frozensets of positive roots:
        subsys = self.subsystems()

        # Will contain the RootSystem's
        rootsubsys = set()

        # Loop over all elements 'subsys' and create its 'RootSystem' object.
        for s in subsys:
            rts = set(s)    # The set of roots
            prd = {}        # Will be the dictionary of inner-products.
            for a in rts:
                for b in rts:
                    prd[(a, b)] = self.prd[(a, b)]
            # Create 'RootSystem' object corresponding to this subsystem
            rootsys = RootSystem(rts, prd)
            rootsubsys.add(rootsys)

        # Return the set of RootSystem's.
        return rootsubsys

    def iso_subsystems(self):
        """
        Returns only one representative of each conjugacy class of symmetric
        # closed subsystems.
        """
        print "Compute isomorphism classes of root systems."
        if self.weyl == None:
            self.get_weyl()

        subsys = self.subsystems().copy()
        iso_classes = set()
        while(len(subsys) > 0):
            s = subsys.pop()
            orbit = set()
            for w in self.weyl:
                orbit.add(frozenset([w[r] for r in s]))
            print "Orbit length : %d" % len(orbit)
            iso_classes.add((s, len(orbit)))
            subsys -= orbit

        print "Done with isomorphism classes of root systems."
        return iso_classes

    def root_iso_subsystems(self):
        """
        Same as 'iso_subsystems' but return them as 'RootSystem' objects
        """
        iso_classes = self.iso_subsystems()

        # Will contain the RootSystem's
        rootsubsys = set()

        # Loop over all elements 'subsys' and create its 'RootSystem' object.
        for s in iso_classes:
            rts = set(s[0])     # The set of roots
            prd = {}            # Will be the dictionary of inner-products.
            for a in rts:
                for b in rts:
                    prd[(a, b)] = self.prd[(a, b)]
            # Create 'RootSystem' object corresponding to this subsystem
            rootsys = RootSystem(rts, prd)
            rootsubsys.add((rootsys, s[1]))

        # Return the set of RootSystem's.
        return rootsubsys

    def leq(self, a, b):
        """
        Check if the isomorphism class 'a' is <= the isomorphism class 'b' in
        the partial order induced by inclusion.
        """
        if self.weyl == None:
            self.get_weyl()
        for w in self.weyl:
            c = set([w[r] for r in a])
            if c <= b:
                return True
        return False

    def poset_iso_subsystems(self):
        """
        Generate a graph representing the partial order.
        """
        print "Compute partial order."
        iso_classes = self.root_iso_subsystems()
        # choose an ordering:
        iso = {}
        i = 0
        for s in iso_classes:
            iso[i] = s
            i += 1

        nodes = {}
        for i in range(len(iso_classes)):
            # dyn = iso[i].dynkin()       CHANGED
            dyn = iso[i][0].dynkin()    # CHANGED
            # nodes[i] = (dyn.identify(), iso[i].HKdim())               CHANGED
            nodes[i] = (dyn.identify(), iso[i][0].HKdim(),
                        len(self.weyl) * iso[i][1] / dyn.weyl_size()) # CHANGED

        edges = {}
        for i in range(len(iso_classes)):
            for j in range(len(iso_classes)):
                if i == j:
                    continue
                # if self.leq(iso[i].rts, iso[j].rts):        CHANGED
                if self.leq(iso[i][0].rts, iso[j][0].rts):  # CHANGED
                    edges[(i, j)] = None

        graph = Graph(nodes, edges)
        graph.hasse_reduce()
        print "Done with partial order."
        return graph

    def HKdim(self):
        return len(self.rts) - len(self.s_rts)





# A couple of methods usful to manipulate tuples:

def basic_tuple(n, i):
    """
    Returns the n-tuple (0, ..., 0, 1, 0, ..., 0) where 1 is in position i.
    """
    ls = []
    for j in range(n):
        if j == i:
            ls.append(1)
        else:
            ls.append(0)
    return tuple(ls)

def add_tuple(s, t):
    """
    Returns the component-wise addition of tuples s and t
    """
    if len(s) != len(t):
        raise ValueError("Tuples of different lengths")

    return tuple(s[i] + t[i] for i in range(len(s)))

def mul_tuple(c, t):
    """
    Multiply every entry of the tuple 't' by 'c'
    """
    return tuple(c * t[i] for i in range(len(t)))

def dim_A(n):
    return n * (n + 2)
def dim_B(n):
    return n * (2 * n + 1)
def dim_C(n):
    return n * (2 * n + 1)
def dim_D(n):
    return n * (2 * n - 1)
def dim_E(n):
    if n == 6:
        return 78
    if n == 7:
        return 133
    if n == 8:
        return 248
def dim_F(n):
    if n == 4:
        return 52
def dim_G(n):
    if n == 2:
        return 14
