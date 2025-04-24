import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import asyncio
import argparse

def path_len(G, path):
  ''' Returns the length of the path given, according to the network'''
  return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))

def custom_heuristic(G, current, neighbor):
  return 1.0

def default_heuristic(G, current, neighbor):
  edge_weight = G[current][neighbor]['weight']
  return (1.0 / edge_weight) ** beta


def calculate_heuristic(G, current, neighbor):
  ''' Calculates the heuristic of the path to each neighbor '''
  handlers = {
    'custom': custom_heuristic
  }

  handler = handlers.get(heuristic_method, default_heuristic)
  return handler(G, current, neighbor)
  

def choose_next_node(G, pheromone, current, visited):
  ''' Function to probabilitsically choose the next node for the agent to visit

  Args:
    G -- The networkx graph representation of the network
    pheromone -- The network pheromone table
    current -- The current node the ant agent is at
    visited -- The set containing the path the ant agent has traversed

  Returns:
    neighbor -- The integer id of the next node for the agent to visit
    None -- None is returned if the current node has no neighbors which the ant has not yet visited (cycle)
  '''
  # Consider neighbors of current node, that the ant has not visited yet
  neighbors = [n for n in G.neighbors(current) if n not in visited]
  if not neighbors: # If neighbors is none, then the ant has run into a cycle (bad!)
    return None
  
  # Calculate weights for each neighbor of the current node
  weights = []
  for neighbor in neighbors:
    edge = tuple(sorted((current, neighbor)))
    pher = pheromone[edge] ** alpha
    heuristic = calculate_heuristic(G, current, neighbor)
    weights.append(pher * heuristic)
  
  # Create a probability distribution from the weights, pick the next node using that dist
  total = sum(weights)
  probs = [w / total for w in weights]
  
  return random.choices(neighbors, weights=probs)[0]

async def ant_agent(G, pheromone, src, dst, ant_id):
  '''Ant agent asynchronous function
  
  Keyword args:
  G -- The networkx graph representing the network
  pheromone -- The pheromone table of the network
  src -- The number of the source router
  dst -- The number of the destination router
  
  Returns:
  path -- The discovered path from src to dst
  '''
  current = src
  path = [current]
  visited = set([current])
  while current != dst:
    next_node = choose_next_node(G, pheromone, current, visited)
    if next_node is None: # Ant has run into a cycle
      return None
    edge_weight = G[current][next_node]['weight']
    for step in range(10):
      t = step / 10
      x = pos[current][0] * (1 - t) + pos[next_node][0] * t
      y = pos[current][1] * (1 - t) + pos[next_node][1] * t
      ant_states[ant_id] = (x, y)
      await asyncio.sleep((0.05 * edge_weight / 2) / animation_speed)
    path.append(next_node)
    visited.add(next_node)
    current = next_node
  
  return path

def update_pheromones(pheromone, paths):
  '''Update pheromones on links given traversed path of an ant agent'''
  # Apply evaporation
  for edge in pheromone:
    pheromone[edge] *= (1 - evap_rate)
  # Lay down new pheromones on the path traversed by the ant
  for path in paths:
    if path:
      length = path_len(G, path)
      for i in range(len(path) - 1):
        edge = tuple(sorted((path[i], path[i+1])))
        pheromone[edge] += pheromone_deposit / length

def draw_frame(iteration):
  plt.clf()
  pher_vals = np.array([pheromone[tuple(sorted(e))] for e in G.edges])
  max_pher = pher_vals.max() if pher_vals.max() > 0 else 1.0

  norm_pher = pher_vals / max_pher

  # Clamp width values for vis purposes
  min_width = 0.3
  max_width = 5.0
  widths = np.clip(norm_pher * max_width, min_width, max_width)

  min_alpha = 0.1
  colors = cm.viridis(norm_pher)
  colors[:, -1] = np.clip(norm_pher, min_alpha, 1.0)

  nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=600)
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, width=widths, edge_color=colors)

  nx.draw_networkx_edge_labels(G, pos, edge_labels={
        (u, v): f"{pheromone[tuple(sorted((u, v)))]:.2f}" for u, v in G.edges
  })

  for x, y in ant_states:
    plt.plot(x, y, 'ro', markersize=8)

  plt.title(f'Iteration: {iteration}')
  plt.axis('off')
  plt.pause(0.01)

async def draw_loop():
  while True:
    draw_frame(draw_loop.iteration)
    await asyncio.sleep(0.03 / animation_speed)
draw_loop.iteration = 0


async def run_simulation(src_node, dst_node, link_sever_prob, link_sever_time):
  ''' Run the ACO simulation '''
  severed_edges = {}

  plt.ion()
  for it in range(num_iterations):
    draw_loop.iteration = it + 1
    global ant_states
    ant_states = [(pos[0][0], pos[0][1]) for _ in range(num_ants)]

    # Handle link severing
    if link_sever_prob:
      if random.random() < link_sever_prob:
        edge = random.choice(list(G.edges))
        u, v = edge
        edge_sorted = tuple(sorted((u, v)))
        severed_edges[edge_sorted] = {
          'time_remaining': link_sever_time,
          'weight': G[u][v]['weight']
        }
        G.remove_edge(u, v)
        print(f'it:{it} -- ({u}, {v}) severed for {link_sever_time}')

    edges_to_restore = []
    for edge, data in severed_edges.items():
      data['time_remaining'] -= 1
      if data['time_remaining'] <= 0:
        edges_to_restore.append(edge)

    # Visualize the graph and ants
    draw_task = asyncio.create_task(draw_loop())

    ant_tasks = [ant_agent(G, pheromone, src_node, dst_node, ant_id) for ant_id in range(num_ants)]
    paths = await asyncio.gather(*ant_tasks)

    draw_task.cancel()
    update_pheromones(pheromone, paths)

    for edge in edges_to_restore:
      u, v = edge
      weight = severed_edges[edge]['weight']
      G.add_edge(u, v, weight=weight)
      del severed_edges[edge]
      print(f'it:{it} -- ({u}, {v}) restored')
  
  plt.ioff()
  valid_paths = [p for p in paths if p]
  if valid_paths:
    best_path = min([p for p in paths if p], key=lambda x: path_len(G, x))
    print('Best path found:', best_path)
    print('Path length:', path_len(G, best_path))
  else:
    print('No paths found in final iteration')
  plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser(prog='ANTI-RIP', 
                                   description='ANTI-RIP: Ant-based Network Traffic Intelligent Routing & Improvement Protocol',
                                   epilog='Created by William Winslade on 24 April 2025')
  parser.add_argument('-G', '--graph-file', help='file path to user specified graph description', default='', type=str)
  parser.add_argument('-SN', '--src-node', help='ID of source node', default=0, type=int)
  parser.add_argument('-DN', '--dst-node', help='ID of destination node', default=-1, type=int)
  parser.add_argument('-A', '--alpha', help='alpha parameter', default=1.0, type=float)
  parser.add_argument('-B', '--beta', help='beta parameter', default=2.0, type=float)
  parser.add_argument('-N', '--num-ants', help='number of ant agents', default=10, type=int)
  parser.add_argument('-I', '--iterations', help='number of iterations', default=25, type=int)
  parser.add_argument('-E', '--evap-rate', help='evaporation rate', default=0.5, type=float)
  parser.add_argument('-D', '--pheromone-deposit', help='pheromone deposit quantity', default=100.0, type=float)
  parser.add_argument('-S', '--animation-speed', help='animation speed, lower is slower', default=1.0, type=float)
  parser.add_argument('-H', '--heuristic-method', help='heuristic calculation method', default='', type=str)
  parser.add_argument('-L', '--link-sever-prob', help='probability of a link being severed (0, 1)', default=0.0, type=float)
  parser.add_argument('-T', '--link-sever-time', help='duration in iterations to keep link severed', default=3, type=int)
  args = parser.parse_args()

  global alpha, beta, evap_rate, pheromone_deposit, num_ants, num_iterations, animation_speed, heuristic_method
  graph_file = args.graph_file
  alpha = args.alpha
  beta = args.beta
  evap_rate = args.evap_rate
  pheromone_deposit = args.pheromone_deposit
  num_ants = args.num_ants
  num_iterations = args.iterations
  animation_speed = args.animation_speed
  heuristic_method = args.heuristic_method

  ## Graph and pheromone table creation
  global G, pos, pheromone, ant_states

  # Check if the user specified a graph file
  if graph_file:
    edges = []

    with open(graph_file, 'r') as f:
      lines = f.readlines()
      for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
          raise RuntimeError('Bad graph specification file. Each line should be: u v w')
        u, v, w = map(int, parts)
        edges.append((u, v, w))
    
    G = nx.Graph()
    G.add_weighted_edges_from(edges)

  else:
    G = nx.Graph()
    edges = [
      (0, 1, 1), (1, 2, 2), (0, 2, 2),
      (2, 3, 1), (1, 3, 3)
    ]

    G.add_weighted_edges_from(edges)

  # Create pheromone table
  pheromone = {tuple(sorted((u, v))): 1.0 for u, v in G.edges}

  # For animation layout: positions
  pos = nx.spring_layout(G, seed=69)

  # For ant animation
  ant_states = []

  # Link severing params
  link_sever_prob = args.link_sever_prob
  link_sever_time = args.link_sever_time

  # Source and destination parsing
  src_node = args.src_node
  assert src_node >= 0 and src_node in G.nodes, 'Source node must be a node within the network spec'

  dst_node = args.dst_node
  if dst_node == -1:
    dst_node = max(G.nodes)
  else:
    assert dst_node != src_node , 'Destination node cannot equal the source node'
    assert dst_node in G.nodes, 'Destination node must be a node within the network spec'


  

  # Run the sim
  asyncio.run(run_simulation(src_node, dst_node, link_sever_prob, link_sever_time))



