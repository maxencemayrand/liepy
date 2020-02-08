import liepy

# Create Dynkin diagram object for the Lie algebra sp(8, C)
dynkin = liepy.DynkinDiagram('C4')

# Create the corresponding root system
rootsystem = liepy.RootSystem(dynkin)

# Compute the poset of root subsystems.
poset = rootsystem.poset_subsystems()

# Output the LaTeX code
poset.latex_hasse()