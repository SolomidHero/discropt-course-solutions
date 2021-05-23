#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import math
import numpy as np
from mip import *
from tqdm import tqdm

Point = namedtuple("Point", ['x', 'y'])
Facility = namedtuple("Facility", ['index', 'setup_cost', 'capacity', 'location'])
Customer = namedtuple("Customer", ['index', 'demand', 'location'])

def length(point1, point2):
  return math.sqrt((point1.fac_used - point2.fac_used)**2 + (point1.y - point2.y)**2)

def distance_matrix(facs, custs):
  fx, fy = np.array([f.location for f in facs]).T
  cx, cy = np.array([c.location for c in custs]).T

  return ((fx - cx[:, np.newaxis]) ** 2 + (fy - cy[:, np.newaxis]) ** 2) ** 0.5

def solve_it(input_data):
  # Modify this code to run your optimization algorithm

  # parse the input
  lines = input_data.split('\n')

  parts = lines[0].split()
  facility_count = int(parts[0])
  customer_count = int(parts[1])
  
  facilities = []
  for i in range(1, facility_count+1):
    parts = lines[i].split()
    facilities.append(Facility(i-1, float(parts[0]), int(parts[1]), Point(float(parts[2]), float(parts[3])) ))

  customers = []
  for i in range(facility_count+1, facility_count+1+customer_count):
    parts = lines[i].split()
    customers.append(Customer(i-1-facility_count, int(parts[0]), Point(float(parts[1]), float(parts[2]))))

  # solution with mip
  obj, opt, solution = facility_mip(facilities, customers)

  # prepare the solution in the specified output format
  output_data = '%.2f' % obj + ' ' + str(opt) + '\n'
  output_data += ' '.join(map(str, solution))

  return output_data


def facility_mip(facilities, customers, verbose=False, time_limit=900):
  # print("prepare...")
  facility_count = len(facilities)
  costs = np.array(list(map(lambda x: x.setup_cost, facilities)))
  customer_count = len(customers)
  get_val = np.vectorize(lambda x, attr: getattr(x, attr))

  # m = Model("facility", solver_name=CBC)
  m = Model("facility")
  m.verbose = verbose
  m.max_mip_gap = 0.05
  m.opt_tol = 1e-3

  dists = distance_matrix(facilities, customers)
  # print("creating problem...")s
  demands = np.array([c.demand for c in customers])

  fac_used = np.array([m.add_var(var_type=BINARY) for _ in facilities])
  # print("creating choices...")
  cust_choices = np.array([
    m.add_var(var_type=BINARY) for _ in tqdm(range(facility_count * customer_count))
  ]).reshape(customer_count, facility_count)

  # print("creating objective...")
  m.objective = minimize(
    xsum(costs * fac_used) +
    xsum((cust_choices * dists).flatten())
  )

  # print("creating constraints...")
  # one customer per facility
  for i in tqdm(range(customer_count)):
    m += xsum(cust_choices[i]) == 1

  # facility is capable of customers
  for i in tqdm(range(facility_count)):
    m += xsum(demands * cust_choices.T[i]) <= facilities[i].capacity

  # facilty is used if customer uses it
  for i in tqdm(range(facility_count)):
    m += xsum(cust_choices.T[i]) <= fac_used[i] * customer_count

  # print("solving...")
  # optimizing
  status = m.optimize(max_seconds=time_limit)
  opt = int(status == OptimizationStatus.OPTIMAL)

  obj = m.objective_values[0]
  solution = get_val(cust_choices, 'x').argmax(axis=1)

  return obj, opt, solution


import sys

if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    file_location = sys.argv[1].strip()
    with open(file_location, 'r') as input_data_file:
      input_data = input_data_file.read()
    print(solve_it(input_data))
  else:
    print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/fl_16_2)')

