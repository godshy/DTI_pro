#!/usr/bin/env python
# coding: utf-8

# In[16]:



import networkx as nx
import pandas as pd
import pygraphviz as pgv
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
# Image.MAX_IMAGE_PIXELS = 3075432330


# In[5]:


graph_source = pd.read_csv('to_graph.csv')
df = pd.read_csv('final_result_mibig.csv')


# In[13]:


graph_source


# In[24]:


node_organism = list(graph_source['organism_name'])
node_organism_colored = []

node_compound = list(graph_source['compound'])
node_compound_colored = []

edges = []
for k in range(len(node_organism)):
    edges.append((node_organism[k], node_compound[k]))


# In[27]:
# organism ---> compound
pos = {}
#pos.update((node, (0, index)) for index, node in enumerate(node_organism))
#pos.update((node, (1, index)) for index, node in enumerate(node_compound))
G = nx.DiGraph()
G.add_nodes_from(node_organism, bipartite=0, color='red')
G.add_nodes_from(node_compound, bipartite=1, color='green')
G.add_edges_from(edges)

# In[35]:
G_new = pgv.AGraph(directed=True)
G_new.add_nodes_from(node_organism, bipartite=0, color='red')
G_new.add_nodes_from(node_compound, bipartite=1, color='green')
G_new.add_edges_from(edges)
G_new.layout(prog="dot")
# G_new.draw('c2o_v_13.svg', prog='dot')

# In[28]:

# nx.nx_agraph.view_pygraphviz(G, prog='fdp')
pos = {}
pos.update((node, (0, index)) for index, node in enumerate(node_organism))
pos.update((node, (1, index)) for index, node in enumerate(node_compound))
# print(pos)
# plt.figure(figsize=(35, 550))
# nx.draw(G, pos=pos)
# plt.savefig('c2o_v_10.png', format='png', with_labels=True)

degree_organism = G_new.degree(node_organism)
degree_compound = G_new.degree(node_compound)
# nx.nx_agraph.view_pygraphviz(G, format='svg', prog='dot')
print(type(node_organism), len(node_organism))
print(type(node_compound), len(node_compound))
print('Number of total nodes:', G_new.number_of_nodes())
node_organism_noduplicate = list(dict.fromkeys(node_organism))
node_compound_noduplicate = list(dict.fromkeys(node_compound))
degree_compound = np.array(degree_compound)
degree_organism = np.array(degree_organism)
print('node of compound:', len(node_compound_noduplicate))
print('node of organism:', len(node_organism_noduplicate))
print('mean degrees of compounds: ', np.mean(degree_compound))
print('mean degrees of organism:', np.mean(degree_organism))
print('END')

