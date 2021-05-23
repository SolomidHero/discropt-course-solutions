import math
# from itertools import combinations_with_replacement as combinations_wr
from itertools import combinations_with_replacement as combinations_wr, combinations
from time import time
import numpy as np
from tqdm import tqdm

class VrpSolver(object):
  def __init__(self, customers, vehicle_count, vehicle_capacity):
    self.customers = customers
    self.indices, self.demands, self.xs, self.ys = np.array(self.customers).T
    self.c_count = len(customers)
    self.v_count = vehicle_count
    self.v_cap = vehicle_capacity
    self.tours = self.trivial_solution()
    self.obj = self.get_objective()
    print(self.obj)

    assert self.customers[0].demand == 0

  def get_results():
    return self.get_objective(), 0, self.tours

  def __str__(self):
    obj = self.get_objective()
    opt = 0
    if not self.is_valid_solution():
      raise ValueError("Solution not valid")
    output_str = "{:.2f} {}\n".format(obj, opt)
    for tour in self.tours:
      output_str += (' '.join(map(str, [c for c in tour])) + '\n')
    return output_str

  @staticmethod
  def dist(c1, c2):
    return math.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2)

  @staticmethod
  def distance_row(x, y, xs, ys):
    return ((xs - x) ** 2 + (ys - y) ** 2) ** 0.5

  @staticmethod
  def pair_dists(xs, ys):
    return ((xs - np.roll(xs, -1)) ** 2 + (ys - np.roll(ys, -1)) ** 2) ** 0.5

  def have_duplicate_missing(self):
    customers = set(range(1, len(self.customers)))
    for tour in self.tours:
      for c in tour[1:-1]:
        if c not in customers:
          print("Duplicate")
          return True
    if customers:
      print("Missing")
      return True
    return False

  def is_valid_tour(self, tour):
    is_correct = tour[0] == 0 and tour[-1] == 0 and len(np.unique(tour)) == len(tour) - 1
    return is_correct and self.demands[tour[1:-1]].sum() <= self.v_cap

  def is_valid_solution(self):
    return all([self.is_valid_tour(tour) for tour in self.tours])

  def single_tour_dist(self, tour):
    if not self.is_valid_tour(tour):
      return np.inf
    return self.pair_dists(self.xs[tour[:-1]], self.ys[tour[:-1]]).sum()

  def get_tour_dists(self):
    return [self.single_tour_dist(tour) for tour in self.tours]

  def get_objective(self):
    dists = self.get_tour_dists()
    if np.inf in dists:
      raise ValueError("Invalid tour detected.")
    else:
      return sum(dists)

  def trivial_solution2(self):
    tours = []
    remaining_customers = set(self.customers[1:])
    for v in range(self.v_count):
      remaining_cap = self.v_cap
      tours.append([])
      tours[-1].append(0)
      while remaining_customers and remaining_cap > min([c.demand for c in remaining_customers]):
        for customer in sorted(remaining_customers, reverse=True, key=lambda c: c.demand):
          if customer.demand <= remaining_cap:
            tours[-1].append(customer.index)
            remaining_cap -= customer.demand
            remaining_customers.remove(customer)
      tours[-1].append(0)
    if remaining_customers:
      raise ValueError("Greedy solution does not exist.")
    else:
      self.tours = tours
      self.obj = self.get_objective()
      return self.tours

  def trivial_solution(self):
    # build a trivial solution
    # assign customers to vehicles starting by the largest customer demands
    vehicle_tours = []
    customer_count = len(self.customers)
    depot = self.customers[0]
    remaining_customers = set(self.customers)
    remaining_customers.remove(depot)
    for v in range(0, self.v_count):
      vehicle_tours.append([])
      capacity_remaining = self.v_cap
      while sum([capacity_remaining >= customer.demand for customer in remaining_customers]) > 0:
        used = set()
        order = sorted(remaining_customers, key=lambda customer: -customer.demand*customer_count + customer.index)
        for customer in order:
          if capacity_remaining >= customer.demand:
            capacity_remaining -= customer.demand
            vehicle_tours[v].append(customer)
            used.add(customer)
        remaining_customers -= used

    # checks that the number of customers served is correct
    assert sum([len(v) for v in vehicle_tours]) == len(self.customers) - 1

    vehicle_tours = [[depot, *tour, depot] for tour in vehicle_tours]
    vehicle_tours = [np.array(list(map(lambda c: c.index, tour))) for tour in vehicle_tours]
    return vehicle_tours

  def relax_tsp_index(self, index, xs, ys, solution):
    value = solution[index]
    solution = np.delete(solution, index)

    cur_xs, cur_ys = xs[solution], ys[solution]
    dists = self.distance_row(xs[value], ys[value], cur_xs, cur_ys)
    dists_sum = np.roll(dists, -1) + dists - self.pair_dists(cur_xs, cur_ys)
    target = dists_sum.argmin()

    solution = np.insert(solution, target + 1, value)
    return solution

  def relax_tsp_solution(self, solution, xs, ys, relax_num=5):
    cur_solution = solution
    for _ in range(relax_num):
      new_solution = cur_solution
      for i in range(1, len(solution)):
        new_solution = self.relax_tsp_index(i, xs, ys, new_solution)
      if (new_solution == cur_solution).all():
        break

    return cur_solution

  def tour_relax(self, tour, permutation_num=20, relax_num=10):
    if len(tour) <= 4:
      return tour

    tour = tour[:-1]
    min_tour = tour[self.relax_tsp_solution(np.arange(len(tour)), self.xs[tour], self.ys[tour])]
    min_obj = self.single_tour_dist(min_tour)

    for i in range(permutation_num):
      perm = np.append(np.random.permutation(len(tour) - 1) + 1, 0)[::-1]

      # tour is permutation
      cur_tour = np.array([0, 1, 2])
      xs, ys = self.xs[tour[perm]], self.ys[tour[perm]]

      for cur in range(len(cur_tour), len(min_tour)):
        cur_xs, cur_ys = xs[:cur][cur_tour], ys[:cur][cur_tour]
        dists = self.distance_row(xs[cur], ys[cur], cur_xs, cur_ys)
        dists_sum = np.roll(dists, -1) + dists - self.pair_dists(cur_xs, cur_ys)
        target = dists_sum.argmin()
        cur_tour = np.insert(cur_tour, target + 1, cur)

      cur_tour = self.relax_tsp_solution(cur_tour, xs[cur_tour], ys[cur_tour])

      obj = self.pair_dists(xs[cur_tour], ys[cur_tour]).sum()

      if obj < min_obj:
        min_obj = obj
        min_tour = tour[perm[cur_tour]]

    return np.append(min_tour, 0)

  def relocate(self, i_from, start_from, end_from, i_to, start_to, debug=False):
    """
    :param i_from: index of tour relocate from
    :param start_from: start index of segment
    :param end_from: end index of segment (inclusive)
    :param i_to: index of tour relocate to
    :param start_to: location
    :param debug: print details if True
    :return: True if improved
    relocate a segment of tour into another tour
    2 possible ways:
    relocate directly and reverse after relocate
    """
    if debug:
      print("relocate")
    from_t = self.tours[i_from]
    to_t = self.tours[i_to]
    improved = False

    segment = from_t[start_from:end_from + 1]
    prev_dist = self.single_tour_dist(from_t) + self.single_tour_dist(to_t)

    from_t = np.append(from_t[:start_from], from_t[end_from + 1:])
    to_t = np.concatenate((to_t[:start_to], segment, to_t[start_to:]))
    to_rev = np.concatenate((to_t[:start_to], segment[::-1], to_t[start_to:]))

    dist_from = self.single_tour_dist(from_t)
    dist_to = self.single_tour_dist(to_t)
    dist_to_rev = self.single_tour_dist(to_rev)
    if dist_to_rev < dist_to:
      to_t = to_rev
      dist_to = dist_to_rev

    if dist_to + dist_from < prev_dist:
      self.tours[i_from] = self.tour_relax(from_t)
      self.tours[i_to] = self.tour_relax(to_t)
      self.obj = self.get_objective()
      return True
    return False

  def exchange(self, i1, start_1, end_1, i2, start_2, end_2, debug=False):
    """
    :param i1:
    :param start_1:
    :param end_1:
    :param i2:
    :param start_2:
    :param end_2:
    :param debug:
    :return:
    exchange 2 segments from 2 tours
    4 possible ways:
    exchange directly, reverse either segment, reverse both segments
    """
    if debug:
      print("exchange")
    t1 = self.tours[i1]
    t2 = self.tours[i2]
    improved = False

    seg_1 = t1[start_1:end_1 + 1]
    seg_2 = t2[start_2:end_2 + 1]

    # tour lengths
    prev_dist = self.single_tour_dist(t1) + self.single_tour_dist(t2)

    # tour1 <- seg2, not reversed and reversed
    t1, t1_rev = (
      np.concatenate((t1[: start_1], seg_2, t1[end_1 + 1:])),
      np.concatenate((t1[: start_1], seg_2[::-1], t1[end_1 + 1:]))
    )
    # tour2 <- seg1, not reversed and reversed
    t2, t2_rev = (
      np.concatenate((t2[: start_2], seg_1, t2[end_2 + 1:])),
      np.concatenate((t2[: start_2], seg_1[::-1], t2[end_2 + 1:]))
    )

    # choose best tour exchanges
    dist_t1 = self.single_tour_dist(t1)
    dist_t1_rev = self.single_tour_dist(t1_rev)
    if dist_t1_rev < dist_t1:
      t1 = t1_rev
      dist_t1 = dist_t1_rev
    dist_t2 = self.single_tour_dist(t2)
    dist_t2_rev = self.single_tour_dist(t2_rev)
    if dist_t2_rev < dist_t2:
      t2 = t2_rev
      dist_t2 = dist_t2_rev

    if dist_t1 + dist_t2 < prev_dist:
      self.tours[i1] = self.tour_relax(t1)
      self.tours[i2] = self.tour_relax(t2)
      self.obj = self.get_objective()
      return True

    return False

  def two_opt(self, i, start, end, debug=False):
    """
    :param i:
    :param start:
    :param end:
    :param debug:
    :return:
    reverse a segment of a tour
    only 1 way to do this
    """
    if debug:
      print("two_opt")
    tour = self.tours[i]
    segment = tour[start:end + 1]
    dist = self.single_tour_dist(tour)

    tour = np.concatenate((tour[:start], segment[::-1], tour[end + 1:]))
    new_dist = self.single_tour_dist(tour)

    if new_dist < dist:
      self.tours[i] = tour
      self.obj = self.get_objective()
      return True
    return False

  def cross(self, i1, i2, j1, j2, debug=False):
    """
    :param i1:
    :param i2:
    :param j1:
    :param j2:
    :param debug:
    :return:
    split two tours into head and tail respectively, and re-shuffle them
    2 possible ways
    """
    if debug:
      print("cross")
    t1 = self.tours[i1]
    t2 = self.tours[i2]

    # old tour lengths
    prev_dist = self.single_tour_dist(t1) + self.single_tour_dist(t2)

    head1 = t1[:j1]
    tail1 = t1[j1:]
    head2 = t2[:j2]
    tail2 = t2[j2:]

    # head + tail
    t1 = np.append(head1, tail2)
    t2 = np.append(head2, tail1)

    # head + head(reversed) / tail(reversed) + tail
    t1_rev = np.append(head1, head2[::-1])
    t2_rev = np.append(tail1[::-1], tail2)

    # new tour lengths
    dist_t1 = self.single_tour_dist(t1)
    dist_t1_rev = self.single_tour_dist(t1_rev)
    dist_t2 = self.single_tour_dist(t2)
    dist_t2_rev = self.single_tour_dist(t2_rev)

    if dist_t2_rev + dist_t1_rev < dist_t2 + dist_t1:
      t1, t2 = t1_rev, t2_rev
      dist_t1, dist_t2 = dist_t1_rev, dist_t2_rev

    if dist_t1 + dist_t2 < prev_dist:
      self.tours[i1] = self.tour_relax(t1)
      self.tours[i2] = self.tour_relax(t2)
      self.obj = self.get_objective()
      return True

    return False

  def solve(self,
        relocate=True,
        exchange=True,
        two_opt=True,
        cross=True,
        tl=None,
        verbose=False,
        debug=False):
    is_improved = True
    t_start = time()
    iterations = 0

    while is_improved:
      if tl and time() - t_start >= tl:
        break
      relocate_improved = False
      exchange_improved = False
      two_opt_improved = False
      cross_improved = False
      self.obj = self.get_objective()
      prev_obj = self.obj
      if verbose or debug:
        print(self.obj)

      # try relocate
      if relocate:
        for i_from, tour_from in enumerate(self.tours):
          if relocate_improved: break
          for start_from, end_from in combinations_wr(range(1, len(tour_from) - 1), 2):
            if relocate_improved: break
            for i_to, tour_to in enumerate(self.tours):
              if relocate_improved: break
              if i_from == i_to: continue
              for start_to in range(1, len(tour_to)):
                if self.relocate(i_from, start_from, end_from, i_to, start_to, debug):
                  relocate_improved = True
                  break

      # try exchange
      if exchange:
        for i1, tour_1 in enumerate(self.tours):
          if exchange_improved: break
          for start_1, end_1 in combinations_wr(range(1, len(tour_1) - 1), 2):
            if exchange_improved: break
            for i2, tour_2 in enumerate(self.tours):
              if exchange_improved: break
              if i1 <= i2: continue
              for start_2, end_2 in combinations_wr(range(1, len(tour_2) - 1), 2):
                if self.exchange(i1, start_1, end_1, i2, start_2, end_2, debug):
                  exchange_improved = True
                  break

      # try exchange
      if two_opt:
        for i, tour in enumerate(self.tours):
          for start, end in combinations(range(1, len(tour) - 1), 2):
            if self.two_opt(i, start, end, debug):
              two_opt_improved = True
              break

      # try cross
      if cross:
        for i1, tour_1 in enumerate(self.tours):
          if cross_improved: break
          for j1 in range(2, len(tour_1) - 2):
            if cross_improved: break
            for i2, tour_2 in enumerate(self.tours):
              if i1 <= i2: continue
              if cross_improved: break
              for j2 in range(2, len(tour_2) - 2):
                if self.cross(i1, i2, j1, j2, debug):
                  cross_improved = True
                  break
      if verbose or debug:
        print(relocate_improved, exchange_improved, two_opt_improved, cross_improved)
      print(f"{iterations}\t {self.obj}\t {time() - t_start:.4}\t{(np.array([len(t) for t in self.tours]) > 2).sum()}/{self.v_count}")
      iterations += 1
      is_improved = relocate_improved or exchange_improved or two_opt_improved or cross_improved
    return self.tours
