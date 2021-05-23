#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
# from gurobipy import *
from mip import *
import numpy as np

Set = namedtuple("Set", ['index', 'cost', 'items'])

def solve_it(input_data):
  # Modify this code to run your optimization algorithm

  # parse the input
  lines = input_data.split('\n')

  parts = lines[0].split()
  item_count = int(parts[0])
  set_count = int(parts[1])

  sets = []
  for i in range(1, set_count+1):
    parts = lines[i].split()
    sets.append(Set(i-1, float(parts[0]), set(map(int, parts[1:]))))

  # solution using mip solver
  obj, opt, solution = mip1(
    item_count, sets, True
  )

  # prepare the solution in the specified output format
  output_data = str(obj) + ' ' + str(opt) + '\n'
  output_data += ' '.join(map(str, solution))

  return output_data


def mip1(item_count, sets, verbose=False, num_threads=None, time_limit=3600):
  m = Model("set_cover")
  m.verbose = verbose

  costs = np.array([s.cost for s in sets])
  selections = np.array([m.add_var(var_type=BINARY) for _ in sets])
  m.objective = minimize(xsum(costs * selections))
  for item in range(item_count):
    m += xsum(np.array([item in s.items for s in sets]) * selections) >= 1

  status = m.optimize(max_seconds=time_limit)

  opt = int(status == OptimizationStatus.OPTIMAL)
  selected = list(map(lambda x: int(x.x), selections))
  total_cost = int(sum([sets[i].cost * selected[i] for i in range(len(sets))]))

  return total_cost, opt, selected

def mip(item_count, sets, verbose=False, num_threads=None, time_limit=3600):
  m = Model("set_covering")
  m.setParam('OutputFlag', verbose)
  if num_threads:
    m.setParam("Threads", num_threads)

  if time_limit:
    m.setParam("TimeLimit", time_limit)

  selections = m.addVars(len(sets), vtype=GRB.BINARY, name="set_selection")

  m.setObjective(LinExpr([s.cost for s in sets], [selections[i]
                                                for i in range(len(sets))]), GRB.MINIMIZE)

  m.addConstrs((LinExpr([1 if j in s.items else 0 for s in sets], [selections[i] for i in range(len(sets))]) >= 1
              for j in range(item_count)),
              name="ieq1")

  m.update()
  m.optimize()

  soln = [int(var.x) for var in m.getVars()]
  total_cost = int(sum([sets[i].cost * soln[i] for i in range(len(sets))]))

  if m.status == 2:
    opt = 1
  else:
    opt = 0

  return total_cost, opt, soln

import sys

if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    file_location = sys.argv[1].strip()
    with open(file_location, 'r') as input_data_file:
      input_data = input_data_file.read()
    print(solve_it(input_data))
  else:
    print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/sc_6_1)')

