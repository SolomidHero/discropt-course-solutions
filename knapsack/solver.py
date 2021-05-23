#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
from tqdm import tqdm
import numpy as np
Item = namedtuple("Item", ['index', 'value', 'weight'])

def solve_it(input_data):
  # parse the input
  lines = input_data.split('\n')

  firstLine = lines[0].split()
  item_count = int(firstLine[0])
  capacity = int(firstLine[1])

  items = []

  for i in range(1, item_count + 1):
    line = lines[i]
    parts = line.split()
    items.append(Item(i-1, int(parts[0]), int(parts[1])))

  # dynamic programming
  value, is_opt, taken = dp(capacity, items) # optimal, O(nclogn) speed, O(c) memory
  # value, is_opt, taken = dp2(capacity, items) # optimal, O(nc) speed, O(nc) memory
  # value, is_opt, taken = dp3(capacity, items) # optimal, wrong back track

  # prepare the solution in the specified output format
  output_data = f"{value} {is_opt}\n"
  output_data += ' '.join(map(str, taken))
  return output_data

def dp2(capacity: int, items: Item):
  K = len(items)
  taken = np.zeros(K, dtype=int)
  values = np.zeros((K + 1, capacity + 1), dtype=int)
  for i, value, weight in tqdm(items):
    values[i + 1, :weight] = values[i, :weight]
    values[i + 1, weight:] = np.maximum(
      values[i, :-weight] + value,
      values[i, weight:]
    )

  cur_weight = capacity
  for i in reversed(range(K)):
    if values[i, cur_weight] != values[i + 1][cur_weight]:
      taken[i] = 1
      cur_weight -= items[i].weight

  return values[-1, -1], 1, taken


def dp3(capacity: int, items: Item):
  taken = np.zeros(len(items), dtype=int)
  values = np.zeros(capacity + 1, dtype=int)
  backtrack = np.zeros(capacity + 1, dtype=int)

  for i, value, weight in tqdm(items):
    mask = values[weight:] < values[:-weight] + value
    values[weight:][mask] = (values[:-weight] + value)[mask]
    backtrack[weight:][mask] = i + 1

  cur_weight = capacity
  while (cur_weight > 0 and backtrack[cur_weight] != 0):
    taken[backtrack[cur_weight] - 1] = 1
    cur_weight -= items[backtrack[cur_weight] - 1].weight

  return values[-1], 1, taken

def dp(capacity: int, items: Item):
  def dp_split(capacity: int, items: Item):
    assert len(items) > 0
    if len(items) == 1:
      if items[0].weight > capacity:
        return np.array([0], dtype=int)
      return np.array([1], dtype=int)

    half = len(items) // 2
    l_row = dp_values(capacity, items[:half])
    r_row = dp_values(capacity, items[half:])
    max_values = l_row + r_row[::-1]
    cap_partition = max_values.argmax()
  
    return np.append(
      dp_split(cap_partition, items[:half]),
      dp_split(capacity - cap_partition, items[half:]),

    )

  def dp_values(capacity: int, items: Item):
    values = np.zeros(capacity + 1, dtype=int)

    for i, value, weight in tqdm(items):
      mask = values[weight:] < values[:-weight] + value
      values[weight:][mask] = (values[:-weight] + value)[mask]

    return values

  return dp_values(capacity, items)[-1], 1, dp_split(capacity, items)

if __name__ == '__main__':
  import sys
  if len(sys.argv) > 1:
    file_location = sys.argv[1].strip()
    with open(file_location, 'r') as input_data_file:
      input_data = input_data_file.read()
    print(solve_it(input_data))
  else:
    print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

