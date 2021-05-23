#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
from mip import *
import numpy as np
from collections import namedtuple
from tqdm import tqdm

Point = namedtuple("Point", ['x', 'y'])

def length(point1, point2):
  return ((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2) ** 0.5

def distance_matrix(points):
  xs, ys = zip(*points)
  xs, ys = np.array(xs), np.array(ys)
  return ((xs - xs[:, np.newaxis]) ** 2 + (ys - ys[:, np.newaxis]) ** 2) ** 0.5

def distance_row(x, y, xs, ys):
  return ((xs - x) ** 2 + (ys - y) ** 2) ** 0.5

def pair_dists(xs, ys):
  return ((xs - np.roll(xs, -1)) ** 2 + (ys - np.roll(ys, -1)) ** 2) ** 0.5

def solve_it(input_data):
  # parse the input
  lines = input_data.split('\n')

  nodeCount = int(lines[0])

  points = []
  for i in range(1, nodeCount+1):
    line = lines[i]
    parts = line.split()
    points.append(Point(float(parts[0]), float(parts[1])))

  # build a trivial solution
  # visit the nodes in the order they appear in the file
  # obj, opt, cycle = tsp_mip(points, verbose=False, time_limit=600)
  obj, opt, cycle = tsp_dp(points, permutation_num=1)

  # prepare the solution in the specified output format
  output_data = '%.2f' % obj + ' ' + str(opt) + '\n'
  output_data += ' '.join(map(str, cycle))

  return output_data

def tsp_mip1(points, verbose=False, time_limit=1800):
  get_val = np.vectorize(lambda x: x.x)
  dists = distance_matrix(points)
  m = Model("tsp")
  m.verbose = verbose

  edges = np.array([m.add_var(ub=len(points) - 1, var_type=INTEGER) for _ in points])
  ids = np.array([m.add_var(ub=len(points), var_type=INTEGER) for _ in points])

  m.objective = minimize(xsum(dists[i][edges[i].x] for i in range(len(edges))))

  # constraint : enter and leave
  for i in range(len(points)):
    m += edges[i] != i

  # constraint : subtour elimination
  for i in range(1, len(points)):
    m += ids[i] - 1 >= ids[edges[i]]

  # optimizing
  status = m.optimize(max_seconds=time_limit)
  opt = int(status == OptimizationStatus.OPTIMAL)

  # checking if a solution was found
  solution = np.array([0])
  if m.num_solutions:
    while True:
      solution = np.append(solution, get_val(edges[solution[-1]]))
      if solution[-1] == 0:
        break

  return m.objective_value, opt, solution[:-1]

def relax(index, xs, ys, solution):
  value = solution[index]
  solution = np.delete(solution, index)

  cur_xs, cur_ys = xs[solution], ys[solution]
  dists = distance_row(xs[value], ys[value], cur_xs, cur_ys)
  dists_sum = np.roll(dists, -1) + dists - pair_dists(cur_xs, cur_ys)
  target = dists_sum.argmin()

  solution = np.insert(solution, target + 1, value)
  return solution

def tsp_dp(points, permutation_num=1, relaxations=1):
  min_obj = np.inf
  min_solution = np.arange(len(points))

  for i in tqdm(range(permutation_num)):
    perm = np.random.permutation(len(points))

    # solution is permutation
    solution = np.array([0, 1, 2])
    xs, ys = np.array(points)[perm].T

    for cur in range(len(solution), len(points)):
      cur_xs, cur_ys = xs[:cur][solution], ys[:cur][solution]
      dists = distance_row(xs[cur], ys[cur], cur_xs, cur_ys)
      dists_sum = np.roll(dists, -1) + dists - pair_dists(cur_xs, cur_ys)
      target = dists_sum.argmin()
      solution = np.insert(solution, target + 1, cur)

    for _ in range(relaxations):
      for i in range(len(points)):
        solution = relax(i, xs, ys, solution)

    obj = pair_dists(xs[solution], ys[solution]).sum()

    if obj < min_obj:
      min_obj = obj
      min_solution = perm[solution]

  return min_obj, 0, min_solution

def tsp_mip(points, verbose=False, time_limit=1800):
  dists = distance_matrix(points)
  m = Model("tsp")
  m.verbose = verbose

  edges = np.array([[m.add_var(var_type=BINARY) for _ in points] for _ in points])
  ids = [m.add_var() for _ in points]

  m.objective = minimize(xsum((dists * edges).flatten()))

  # constraint : leave each city only once
  for i in range(len(points)):
    m += xsum(edges[i][np.arange(len(points)) != i]) == 1

  # constraint : enter each city only once
  for i in range(len(points)):
    m += xsum(edges[np.arange(len(points)) != i][:, i]) == 1

  # constraint : subtour elimination
  for i in range(1, len(points)):
    for j in range(1, len(points)):
      if i != j:
        m += ids[i] - (len(points) + 1) * edges[i][j] >= ids[j] - len(points)

  # optimizing
  status = m.optimize(max_seconds=time_limit)
  opt = int(status == OptimizationStatus.OPTIMAL)

  # checking if a solution was found
  solution = np.array([0])
  get_val = np.vectorize(lambda x: x.x)
  if m.num_solutions:
    while True:
      solution = np.append(solution, get_val(edges[solution[-1]]).argmax())
      if solution[-1] == 0:
        break

  return m.objective_value, opt, solution[:-1]

import sys

if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    file_location = sys.argv[1].strip()
    with open(file_location, 'r') as input_data_file:
      input_data = input_data_file.read()
    print(solve_it(input_data))
  else:
    print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/tsp_51_1)')

