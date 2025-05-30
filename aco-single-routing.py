import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import asyncio
import argparse
import math
from matplotlib.animation import FuncAnimation, PillowWriter
from collections import defaultdict

def path_len(G, path):
  ''' Returns the length of the path given, according to the network'''
  return sum(G[path[i]][path[i+1]]['weight'] for i in range(len(path) - 1))
  # tot_pher = 0.0
  # for i in range(len(path) - 1):
  #   edge = tuple(sorted((path[i], path[i+1])))
  #   tot_pher += pheromone[edge]
  # return 1 - (tot_pher / len(path))

def link_availability_heuristic(G, current, neighbor):
  # failure_counts: {edge: failure_count}
  global failure_counts
  edge = tuple(sorted((current, neighbor)))
  fail_score = failure_counts.get(edge, 0)
  edge_weight = G[current][neighbor]['weight']

  return (1.0 / edge_weight) * math.exp(-fail_score)

def connectivity_heuristic(G, current, neighbor):
  edge_weight = G[current][neighbor]['weight']
  degree = G.degree(neighbor)

  return (degree / edge_weight)

def default_heuristic(G, current, neighbor):
  edge_weight = G[current][neighbor]['weight']
  return (1.0 / edge_weight)

def hybrid_heuristic(G, current, neighbor):
  global failure_counts
  edge = tuple(sorted((current, neighbor)))
  fail_score = failure_counts.get(edge, 0)
  edge_weight = G[current][neighbor]['weight']
  hybrid_score = (
    2.0 * (1 - edge_weight) +
    1 * G.degree(neighbor) +
    0.75 * (- fail_score)
  )

  return hybrid_score

def calculate_heuristic(G, current, neighbor):
  ''' Calculates the heuristic of the path to each neighbor '''
  handlers = {
    'link-availability': link_availability_heuristic,
    'connectivity': connectivity_heuristic,
    'hybrid': hybrid_heuristic,
  }
  if heuristic_method and heuristic_method not in handlers:
    print('Unrecognized heuristic method specified -- Using default (inverse weight)')

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
    weights.append(pher * (heuristic ** beta))
  
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
  # Setup figure
  ax_graph.clear()
  ax_table.clear()

  # Create the title depending on args
  global src_node, dst_nodes
  if len(dst_nodes) == len(G.nodes) - 1:
    fig.suptitle(f'Routing {src_node} -> all | Iteration {iteration}', fontsize=14)
  else:
    fig.suptitle(f'Routing {src_node} -> {dst_nodes} | Iteration {iteration}', fontsize=14)

  # Get pheromone values, normalize and clamp for colorized representation on fig
  pher_vals = np.array([pheromone[tuple(sorted(e))] for e in G.edges])
  max_pher = pher_vals.max() if pher_vals.max() > 0 else 1.0
  norm_pher = pher_vals / max_pher

  min_width = 0.3
  max_width = 5.0
  widths = np.clip(norm_pher * max_width, min_width, max_width)

  min_alpha = 1.0
  colors = cm.inferno(norm_pher)
  colors[:, -1] = np.clip(norm_pher, min_alpha, 1.0)

  # Draw nodes, their labels, and edges
  nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, ax=ax_graph)
  nx.draw_networkx_labels(G, pos, ax=ax_graph)
  nx.draw_networkx_edges(G, pos, width=widths, edge_color=colors, ax=ax_graph)

  # Draw pheromone values on edges
  # nx.draw_networkx_edge_labels(G, pos, edge_labels={
  #     (u, v): f"{pheromone[tuple(sorted((u, v)))]:.2f}" for u, v in G.edges
  # }, ax=ax_graph)

  # Draw the ants
  colors = cm.hsv(np.linspace(0, 1, len(ant_states)))
  for (x, y), c in zip(ant_states, colors):
    ax_graph.plot(x, y, 'o', markersize=8, color=c)

  # Turn off axes on graph
  ax_graph.axis('off')
  

  # Display N best routes on the figure per dst node
  ax_table.axis('off')
  global routing_table
  table_lines = [f'Routing Table for Node {src_node}\n--------------------\n']
  for dst in sorted(routing_table.keys()):
    if dst == src_node:
      continue
    for i, (path, cost) in enumerate(routing_table[dst]):
      table_lines.append(f'To {dst}: [{i+1}]: ({cost:.2f}) {' -> '.join(map(str, path))} ')
  
  table_text = '\n'.join(table_lines)
  ax_table.text(0.01, 1.0, table_text, fontsize=6, family='monospace', va='top')

  # Save frames for later stitching if wanted
  global frame_count, save_frames
  if save_frames:
    plt.savefig(f'frames/frame_{frame_count:06d}.png', dpi=150)
    frame_count += 1
 
  plt.pause(0.01)
  

async def draw_loop():
  while True:
    draw_frame(draw_loop.iteration)
    await asyncio.sleep(0.03 / animation_speed)
draw_loop.iteration = 0


async def run_simulation(src_node, dst_nodes, link_sever_prob, link_sever_time):
  ''' Run the ACO simulation '''
  severed_edges = {}
  global routing_table
  routing_table = defaultdict(list)

  plt.ion()
  it = 0
  while it < num_iterations or run_continuously:
    draw_loop.iteration = it + 1
    global ant_states
    ant_states = []
    for _ in range(num_ants * len(dst_nodes)):
      ant_states.append((pos[src_node][0], pos[src_node][1]))

    # Handle link severing
    if link_sever_prob:
      if random.random() < link_sever_prob:
        # Get an edge from the graph
        edge = random.choice(list(G.edges))
        u, v = edge
        edge_sorted = tuple(sorted((u, v)))

        # Check if this link going down will affect the routing table
        routes_to_remove = []
        for dst in list(routing_table.keys()):
          routes = routing_table[dst]
          filtered_routes = [
            (path, cost) for (path, cost) in routes
            if not any(tuple(sorted((path[i], path[i+1]))) == edge_sorted for i in range(len(path) - 1)) 
          ]
        
          if len(filtered_routes) < len(routes):
            print(f'Removed {len(routes) - len(filtered_routes)} route(s) to {dst} due to link severed {edge_sorted}')

          if filtered_routes:
            routing_table[dst] = filtered_routes
          else:
            del routing_table[dst]

        # Add the edge to the severed_edges dict, remove it from the graph
        severed_edges[edge_sorted] = {
          'time_remaining': link_sever_time,
          'weight': G[u][v]['weight']
        }

        # Update the edge failure count
        failure_counts[edge_sorted] = failure_counts.get(edge_sorted, 0) + 1

        # Remove the edge from the graph
        G.remove_edge(u, v)
        print(f'it:{it} -- ({u}, {v}) severed for {link_sever_time}')


    edges_to_restore = []
    for edge, data in severed_edges.items():
      data['time_remaining'] -= 1
      if data['time_remaining'] <= 0:
        edges_to_restore.append(edge)

    # Visualize the graph and ants
    draw_task = asyncio.create_task(draw_loop())

    ant_tasks = []
    ant_id = 0
    for dst_node in dst_nodes:
      for _ in range(num_ants):
        ant_tasks.append(ant_agent(G, pheromone, src_node, dst_node, ant_id))
        ant_id += 1
    
    
    paths = await asyncio.gather(*ant_tasks)
    for path in paths:
      if path:
        dst = path[-1]
        length = path_len(G, path)
        routes = routing_table.get(dst, [])
        if not any(existing_path == path for existing_path, _ in routes):
          routes.append((path, length))
          routes.sort(key=lambda x: x[1])
          routing_table[dst] = routes[:routing_table_size]
      
    draw_frame(it + 1)
    draw_task.cancel()
    update_pheromones(pheromone, paths)

    for edge in edges_to_restore:
      u, v = edge
      weight = severed_edges[edge]['weight']
      G.add_edge(u, v, weight=weight)
      del severed_edges[edge]
      print(f'it:{it} -- ({u}, {v}) restored')
    
    # Increase iteration counter
    it += 1
  
  plt.ioff()
  plt.show()

def argument_parser():
  parser = argparse.ArgumentParser(prog='ANTI-RIP', 
                                   description='ANTI-RIP: Ant-based Network Traffic Intelligent Routing & Improvement Protocol',
                                   epilog='Created by William Winslade on 24 April 2025')
  parser.add_argument('-G', '--graph-file', help='file path to user specified graph description', default='', type=str)
  parser.add_argument('-SN', '--src-node', help='ID of source node', default=None, type=int)
  parser.add_argument('-DN', '--dst-nodes', help='ID of destination node(s). Pass "all" to route to all non-source nodes', default=None, type=str)
  parser.add_argument('-A', '--alpha', help='alpha parameter', default=1.0, type=float)
  parser.add_argument('-B', '--beta', help='beta parameter', default=2.0, type=float)
  parser.add_argument('-N', '--num-ants', help='number of ant agents', default=10, type=int)
  parser.add_argument('-I', '--iterations', help='number of iterations', default=25, type=int)
  parser.add_argument('-E', '--evap-rate', help='evaporation rate', default=0.5, type=float)
  parser.add_argument('-D', '--pheromone-deposit', help='pheromone deposit quantity', default=100, type=float)
  parser.add_argument('-S', '--animation-speed', help='animation speed, lower is slower', default=1.0, type=float)
  parser.add_argument('-H', '--heuristic-method', help='heuristic calculation method', default='', type=str)
  parser.add_argument('-L', '--link-sever-prob', help='probability of a link being severed (0, 1)', default=0.0, type=float)
  parser.add_argument('-T', '--link-sever-time', help='duration in iterations to keep link severed', default=3, type=int)
  parser.add_argument('-RT', '--routing-table-size', help='Max number of paths to save to routing table per destination', default=1, type=int)
  parser.add_argument('-R', '--save-frames', help='Pass this flag to save rendered animation frames', action='store_true')
  parser.add_argument('--inf', help='Run simulation forever', action='store_true')
  args = parser.parse_args()

  return args

if __name__ == '__main__':
  args = argument_parser()

  global alpha, beta, evap_rate, pheromone_deposit, num_ants, num_iterations, animation_speed, heuristic_method, save_frames, routing_table_size, run_continuously
  graph_file = args.graph_file
  alpha = args.alpha
  beta = args.beta
  evap_rate = args.evap_rate
  pheromone_deposit = args.pheromone_deposit
  num_ants = args.num_ants
  num_iterations = args.iterations
  run_continuously = args.inf
  animation_speed = args.animation_speed
  heuristic_method = args.heuristic_method
  save_frames = args.save_frames
  routing_table_size = args.routing_table_size

  ## Graph and pheromone table creation
  global G, pos, pheromone, ant_states, failure_counts

  # Deal with graphfile argument, initialize the graph
  if graph_file:
    # If the user passed the below string, call the equivalent networkx graph generator
    if graph_file == 'nx.random_geometric_graph':
      G = nx.random_geometric_graph(15, 0.7)
      pos = nx.get_node_attributes(G, 'pos')
      for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        distance = math.hypot(x1 - x2, y1 - y2)
        G[u][v]['weight'] = distance
    else:
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
  
  # Deal with source node and dest node args
  src_arg = args.src_node
  if src_arg:
    if src_arg not in G.nodes:
      raise RuntimeError('Invalid --src-node argument. Source node must be a node in the network')
    src_node = src_arg
  else:
    src_node = min(G.nodes)

  dst_arg = args.dst_nodes
  if dst_arg is None:
    dst_nodes = [max(G.nodes)]
  elif dst_arg.lower() == 'all':
    dst_nodes = [n for n in G.nodes if n != src_node]
  else:
    try:
      dst = int(dst_arg)
      if dst not in G.nodes:
        raise ValueError('Destination node must be a node in the network')
      if dst == src_node:
        raise ValueError('Destination node cannot be the source node')
      dst_nodes = [dst]
    except ValueError as e:
      raise RuntimeError(f'Invalid --dst-nodes argument: {e}')

  # Create pheromone table
  pheromone = {tuple(sorted((u, v))): 1.0 for u, v in G.edges}

  # For animation layout: positions
  pos = nx.spring_layout(G, seed=69)
  for u, v in G.edges():
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    dist = math.hypot(x1 - x2, y1 - y2)
    G[u][v]['weight'] = dist

  # Link severing params
  link_sever_prob = args.link_sever_prob
  link_sever_time = args.link_sever_time

  # Prep pyplot subplot for visualization
  fig, (ax_graph, ax_params, ax_table) = plt.subplots(1, 3, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 1, 2]})
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.ion()

  ax_params.axis('off')
  if heuristic_method == '':
    heuristic_text = 'Inverse Weight'
  else:
    heuristic_text = heuristic_method

  param_text = f'''
  Alpha: {alpha:.2f}
  Beta: {beta:.2f}
  Evaporation Rate: {evap_rate:.2f}
  Pher. Deposit Q: {pheromone_deposit}
  Num Ants / Destination: {num_ants}
  Heuristic: {heuristic_text}
  Link Failure Prob.: {link_sever_prob}
  Link Failure Duration: {link_sever_time}
  Max Routes / Node: {routing_table_size}
  '''
  ax_params.text(0.01, 1.0, param_text, fontsize=9, va='top', family='monospace')

  # Display color bars for better interpretations
  pher_sm = cm.ScalarMappable(cmap=cm.inferno, norm=Normalize(vmin=0, vmax=1))
  pher_cbar = plt.colorbar(pher_sm, ax=ax_params, orientation='horizontal')
  pher_cbar.set_ticks([0, 1])
  pher_cbar.set_ticklabels(['Less Pheromone', 'More Pheromone'])

  ant_sm = cm.ScalarMappable(cmap=cm.hsv, norm=Normalize(vmin=0, vmax=1))
  ant_cbar = plt.colorbar(ant_sm, ax=ax_params, orientation='horizontal')
  ant_cbar.set_ticks([0, 1])
  ant_cbar.set_ticklabels(['Low Ant ID', 'High Ant ID'])

  # For ant animation
  ant_states = []

  # For availability heuristic
  failure_counts = {}

  

  global frame_count
  frame_count = 0

  # Run the sim
  asyncio.run(run_simulation(src_node, dst_nodes, link_sever_prob, link_sever_time))



