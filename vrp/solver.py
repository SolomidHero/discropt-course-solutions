#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from collections import namedtuple
from mip import *
from tqdm import tqdm
import numpy as np
from VrpSolver2 import VrpSolver


Customer = namedtuple("Customer", ['index', 'demand', 'x', 'y'])

def length(customer1, customer2):
  return math.sqrt((customer1.x - customer2.x)**2 + (customer1.y - customer2.y)**2)

def distance_matrix(customers):
  inds, demands, xs, ys = zip(*customers)
  xs, ys = np.array(xs), np.array(ys)
  return ((xs - xs[:, np.newaxis]) ** 2 + (ys - ys[:, np.newaxis]) ** 2) ** 0.5

def distance_matrix_by_coords(xs, ys):
  return ((xs - xs[:, np.newaxis]) ** 2 + (ys - ys[:, np.newaxis]) ** 2) ** 0.5

def distance_row(x, y, xs, ys):
  return ((xs - x) ** 2 + (ys - y) ** 2) ** 0.5

def pair_dists(xs, ys):
  return ((xs - np.roll(xs, -1)) ** 2 + (ys - np.roll(ys, -1)) ** 2) ** 0.5

def calc_objective(customers, tours):
  inds, demands, xs, ys = np.array(customers, dtype=int).T
  obj = 0
  for t in tours:
    obj += pair_dists(xs[t[:-1]], ys[t[:-1]]).sum()

  return obj
    

def get_tour(start_index, array):
  cur = start_index
  tour = np.array([0, cur])

  while cur != 0:
    cur = array[cur]
    tour = np.append(tour, cur)

  return tour

get_val = np.vectorize(lambda x, attr: getattr(x, attr))

def random_solve(solver, customers, vehicle_count, vehicle_capacity, params=None, times=1):
  best_obj = np.inf
  best_solution = None
  best_perm = np.arange(len(customers) - 1)
  for _ in range(times):
    perm = np.random.permutation(len(customers) - 1)
    customers_permuted = [customers[0]] + [customers[p + 1] for p in perm]

    solution = solver(customers, vehicle_count, vehicle_capacity, **params)
    if solution[0] < best_obj:
      best_solution = solution

  return best_solution

def solve_it(input_data):
  # parse the input
  lines = input_data.split('\n')

  parts = lines[0].split()
  customer_count = int(parts[0])
  vehicle_count = int(parts[1])
  vehicle_capacity = int(parts[2])
  
  customers = []
  for i in range(1, customer_count+1):
    line = lines[i]
    parts = line.split()
    customers.append(Customer(i-1, int(parts[0]), float(parts[1]), float(parts[2])))

  # mip solution
  tl = 800
  if len(customers) < 5:
    obj, opt, tours = vrp_mip_relax(customers, vehicle_count, vehicle_capacity, time_limit=tl)

    # prepare the solution in the specified output format
    outputData = '%.2f' % obj + ' ' + str(opt) + '\n'
    for v in range(0, vehicle_count):
      if len(tours) > v:
        outputData += ' '.join([str(customers[ci].index) for ci in tours[v]]) + '\n'
      else:
        outputData += f"{customers[0].index} {customers[0].index}\n"
  else:
    # solution with optimizations: 2-opt, interchange, shift
    solver = VrpSolver(customers, vehicle_count, vehicle_capacity)
    solver.solve(tl=tl)
    outputData = str(solver)

  return outputData


def vrp_mip_relax(customers, vehicle_count, vehicle_capacity, relax_num=5, verbose=True, time_limit=1800):
  _, opt, tours = vrp_mip(customers, vehicle_count, vehicle_capacity, verbose=verbose, time_limit=time_limit)

  _, _, xs, ys = np.array(customers).T
  rlx_tours = []
  for t in tours:
    t = np.array(t[:-1], dtype=int)
    optimal_tour = np.arange(len(t))
    if len(t) > 3:
      for i in range(relax_num):
        optimal_tour = relax_tsp_solution(optimal_tour, xs[t], ys[t])
    rlx_tours.append(np.append(t[optimal_tour], 0))

  obj = 0
  for t in rlx_tours:
    obj += pair_dists(xs[t[:-1]], ys[t[:-1]]).sum()

  return obj, opt, rlx_tours

def vrp_double_mip(customers, vehicle_count, vehicle_capacity, verbose=False, time_limit=1800):
  _, _, init_tours = vrp_mip(
    customers, vehicle_count, vehicle_capacity, verbose=True, time_limit=30
  )

  cust_count = len(customers)
  customers_np = np.array(customers)
  inds, demands, xs, ys = np.array(customers, dtype=int).T

  m = Model("vrp2")
  m.verbose = verbose
  m.opt_tol = 1e-3

  edges = np.array([
    m.add_var(var_type=BINARY) for i in tqdm(range(cust_count ** 2))
  ]).reshape(cust_count, cust_count)
  ids = [m.add_var(var_type=INTEGER) for _ in tqdm(range(cust_count))]

  # init with previous mip solution as first feasible
  # improve this solution a little
  opt_init_tours = []
  for t in init_tours:
    t = np.array(t[:-1], dtype=int)
    if len(t) > 3:
      optimal_tour = relax_tsp_solution(np.arange(len(t)), xs[t], ys[t])
      opt_init_tours.append(np.append(t[optimal_tour], 0))
  init_tours = opt_init_tours
  init_ids = calc_caps(customers, init_tours)

  init_edges = [
    (edges[i, j], 1) for ts in init_tours for i, j in zip(ts[:-1], ts[1:])
  ]
  init_ids = [(ids[i], init_ids[i]) for i in range(len(init_ids))]
  m.start = init_edges + init_ids + [(edges[0, 0], 0)]

  m = mip_update_model(m, edges, ids, customers, vehicle_count, vehicle_capacity)

  status = m.optimize(max_seconds=time_limit)
  opt = int(status == OptimizationStatus.OPTIMAL)

  obj = m.objective_value

  edges_vals = get_val(edges, 'x').astype(int)
  starts = np.arange(cust_count)[edges_vals[0] > 0]
  arrows = edges_vals.argmax(axis=1)

  tours = []
  for i in starts:
    tours.append(get_tour(i, arrows))

  return obj, opt, tours

def calc_caps(customers, tours):
  inds, demands, xs, ys = np.array(customers, dtype=int).T
  caps = np.zeros(len(customers), dtype=int)
  for tour in tours:
    for i in range(1, len(tour) - 1):
      caps[inds[tour[i]]] = caps[inds[tour[i - 1]]] + demands[tour[i]]

  return caps

def relax(index, xs, ys, solution):
  value = solution[index]
  solution = np.delete(solution, index)

  cur_xs, cur_ys = xs[solution], ys[solution]
  dists = distance_row(xs[value], ys[value], cur_xs, cur_ys)
  dists_sum = np.roll(dists, -1) + dists - pair_dists(cur_xs, cur_ys)
  target = dists_sum.argmin()

  solution = np.insert(solution, target + 1, value)
  return solution

def relax_tsp_solution(solution, xs, ys):
  for i in range(1, len(solution)):
    solution = relax(i, xs, ys, solution)
  
  return solution

def tsp_dp(customers, permutation_num=1, relaxations=1):
  min_obj = np.inf
  min_solution = np.arange(len(customers))

  for i in range(permutation_num):
    perm = np.append(np.random.permutation(len(customers) - 1) + 1, 0)[::-1]

    # solution is permutation
    solution = np.array([0, 1, 2])
    inds, demands, xs, ys = np.array(customers)[perm].T

    for cur in range(len(solution), len(customers)):
      cur_xs, cur_ys = xs[:cur][solution], ys[:cur][solution]
      dists = distance_row(xs[cur], ys[cur], cur_xs, cur_ys)
      dists_sum = np.roll(dists, -1) + dists - pair_dists(cur_xs, cur_ys)
      target = dists_sum.argmin()
      solution = np.insert(solution, target + 1, cur)

    for _ in range(relaxations):
      solution = relax_tsp_solution(solution, xs, ys)

    obj = pair_dists(xs[solution], ys[solution]).sum()

    if obj < min_obj:
      min_obj = obj
      min_solution = perm[solution]

  return min_obj, 0, min_solution

def mip_update_model(m, edges, ids, customers, vehicle_count, vehicle_capacity):
  dists = distance_matrix(customers)
  cust_count = len(customers)

  # objective
  m.objective = minimize(xsum((dists * edges).flatten()))

  # warehouse: exits number no more than vehicles
  m += xsum(edges[0]) <= vehicle_count

  # warehouse: entries num == exits num
  m += xsum(edges[0]) == xsum(edges.T[0])

  # customers: vehicle gets in and out of customer once
  for i in range(1, cust_count):
    m += xsum(edges[i][np.arange(cust_count) != i]) == 1
  for i in range(1, cust_count):
    m += xsum(edges[np.arange(cust_count) != i][:, i]) == 1

  # vehicle: each vehicle capacity >= demands on it
  m += ids[0] == 0
  for i in tqdm(range(cust_count)):
    m += ids[i] <= vehicle_capacity
    for j in range(1, cust_count):
      if i != j:
        m += ids[i] + (vehicle_capacity + customers[j].demand) * \
            edges[i, j] - ids[j] <= vehicle_capacity

  return m

def vrp_mip(customers, vehicle_count, vehicle_capacity, verbose=False, time_limit=1800):
  cust_count = len(customers)
  customers_np = np.array(customers)

  m = Model("vrp")
  m.verbose = verbose
  m.opt_tol = 1e-3

  edges = np.array([
    m.add_var(var_type=BINARY) for i in tqdm(range(cust_count ** 2))
  ]).reshape(cust_count, cust_count)
  ids = [m.add_var(var_type=INTEGER) for _ in tqdm(range(cust_count))]
  
  # init with greedy solution as first feasible
  _, _, init_tours, init_ids = random_solve(
    vrp_trivial,
    customers, vehicle_count, vehicle_capacity, params={"with_caps": True}
  )

  # improve greedy solution a little
  print(calc_objective(customers, init_tours))
  init_tours = dp_procedure(customers, init_tours)
  print(calc_objective(customers, init_tours))
  init_tours = two_opt_procedure(customers, init_tours)
  print(calc_objective(customers, init_tours))
  init_ids = calc_caps(customers, init_tours)

  init_edges = [
    (edges[i, j], 1) if i != j else (edges[i, j], 0)
      for ts in init_tours for i, j in zip(ts[:-1], ts[1:])
  ]
  init_ids = [(ids[i], init_ids[i]) for i in range(len(init_ids))]
  m.start = init_edges + init_ids + [(edges[0, 0], 0)]

  m = mip_update_model(m, edges, ids, customers, vehicle_count, vehicle_capacity)

  status = m.optimize(max_seconds=time_limit)
  opt = int(status == OptimizationStatus.OPTIMAL)

  obj = m.objective_value

  edges_vals = get_val(edges, 'x').astype(int)
  starts = np.arange(cust_count)[edges_vals[0] > 0]
  arrows = edges_vals.argmax(axis=1)

  tours = []
  for i in starts:
    tours.append(get_tour(i, arrows))
  
  return obj, opt, tours

def vrp_trivial(customers, vehicle_count, vehicle_capacity, with_caps=False):
  # build a trivial solution
  # assign customers to vehicles starting by the largest customer demands
  vehicle_tours = []

  customer_count = len(customers)
  depot = customers[0]
  remaining_customers = set(customers)
  remaining_customers.remove(depot)

  for v in range(0, vehicle_count):
    # print "Start Vehicle: ",v
    vehicle_tours.append([])
    capacity_remaining = vehicle_capacity
    while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
      used = set()
      order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
      for customer in order:
        if capacity_remaining >= customer.demand:
          capacity_remaining -= customer.demand
          vehicle_tours[v].append(customer)
          # print '   add', ci, capacity_remaining
          used.add(customer)
      remaining_customers -= used

  # checks that the number of customers served is correct
  assert sum([len(v) for v in vehicle_tours]) == len(customers) - 1

  # calculate the cost of the solution; for each vehicle the length of the route
  obj = 0
  for v in range(0, vehicle_count):
    vehicle_tour = vehicle_tours[v]
    if len(vehicle_tour) > 0:
      obj += length(depot, vehicle_tour[0])
      for i in range(0, len(vehicle_tour)-1):
        obj += length(vehicle_tour[i], vehicle_tour[i+1])
      obj += length(vehicle_tour[-1], depot)

  
  vehicle_tours = [[depot, *tour, depot] for tour in vehicle_tours]
  if with_caps:
    caps = np.zeros(customer_count, dtype=int)
    for tour in vehicle_tours:
      for i in range(1, len(tour) - 1):
        caps[tour[i].index] = caps[tour[i - 1].index] + tour[i].demand

  vehicle_tours = [list(map(lambda c: c.index, tour)) for tour in vehicle_tours]

  if with_caps:
    return obj, 0, vehicle_tours, caps
  return obj, 0, vehicle_tours

def dp_procedure(customers, tours):
  customers = np.array(customers, dtype=int)
  opt_init_tours = []
  for t in tours:
    t = np.array(t[:-1], dtype=int)
    if len(t) > 3:
      _, _, optimal_tour = tsp_dp(customers[t], permutation_num=10, relaxations=10)
      opt_init_tours.append(np.append(t[optimal_tour], 0))
  return opt_init_tours

def two_opt_procedure(customers, tours, times=5):
  inds, demands, xs, ys = np.array(customers, dtype=int).T
  for _ in range(times):
    opt_tours = []
    for tour in tours:
      tour_len = len(tour)
      cur_xs, cur_ys = xs[tour], ys[tour]
      dists = distance_matrix_by_coords(cur_xs, cur_ys)
      diff_tours = np.array([])
      min_dist = np.inf
      min_i_j = (0, tour_len - 1)

      improved = False
      for i in range(tour_len - 1):
        for j in range(i + 3, tour_len):
          if dists[i, j - 1] + dists[j, i + 1] < dists[i, i + 1] + dists[j, j - 1]:
            tour = np.concatenate((tour[:i + 1], tour[j - 1:i:-1], tour[j:]))
            improved = True
            break
        if improved:
          break

      opt_tours.append(tour)
    tours = opt_tours
  return tours


import sys

if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    file_location = sys.argv[1].strip()
    with open(file_location, 'r') as input_data_file:
      input_data = input_data_file.read()
    print(solve_it(input_data))
  else:

    print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/vrp_5_4_1)')

