import networkx as nx
from karateclub import DeepWalk

g = nx.newman_watts_strogatz_graph(100, 20, 0.05)

model = DeepWalk()
model.fit(g)
embedding = model.get_embedding()
