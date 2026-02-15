from Problem import Problem
import networkx as nx

def solution(p: Problem):
    # Get the graph and parameters
    graph = p.graph
    alpha = p.alpha
    beta = p.beta

    # remaining gold for each city
    remaining_gold = {node: graph.nodes[node]['gold'] for node in graph.nodes()}
    repeat_visits = {node: 0 for node in graph.nodes()}

    path = [(0, 0.0)]
    current_city = 0
    carried_gold = 0.0

    # ---- Safe path-building helpers ----
    def append_visit(city, gold_taken):
        """Append (city, gold_taken) but never create consecutive identical nodes.
        If the last entry is already 'city', merge gold into it.
        """
        gold_taken = float(gold_taken)
        if path and path[-1][0] == city:
            path[-1] = (city, float(path[-1][1]) + gold_taken)
        else:
            path.append((city, gold_taken))

    # compute cost for a single edge move
    def edge_cost(u, v, weight):
        dist = graph[u][v]['dist']
        return dist + (alpha * dist * weight) ** beta

    sssp_cache = {}  # source -> (paths_dict, dists_dict)

    def _ensure_source(source):
        if source not in sssp_cache:
            # single call gives both distances and paths
            dists, paths = nx.single_source_dijkstra(graph, source=source, weight='dist')
            sssp_cache[source] = (paths, dists)

    def get_shortest_path(u, v):
        _ensure_source(u)
        return sssp_cache[u][0].get(v, None)

    def get_shortest_distance(u, v):
        _ensure_source(u)
        return sssp_cache[u][1].get(v, float('inf'))
    def path_cost(nodes, w):
        """Compute cost of traversing a path with given weight."""
        if not nodes or len(nodes) < 2:
            return 0.0
        return sum(edge_cost(a, b, w) for a, b in zip(nodes, nodes[1:]))

    def move_along(pth):
        """Move along a path, appending intermediate nodes with 0 gold."""
        nonlocal current_city
        if not pth or len(pth) < 2:
            return
        for v in pth[1:]:
            append_visit(v, 0.0)
            current_city = v

    def go_to_base():
        """Return to base, updating path and state."""
        nonlocal current_city, carried_gold
        if current_city != 0:
            pth = get_shortest_path(current_city, 0)
            if pth is not None:
                move_along(pth)
                current_city = 0
            # If no path exists, we're stuck - but this shouldn't happen in a valid graph
        carried_gold = 0.0

    # This function calculates the optimal load cap for partial pickups
    def calculate_optimal_load_cap(distance_from_base):
        if beta <= 1.0 or distance_from_base <= 0:
            return float('inf')
        try:
            denominator = (beta - 1) * alpha * beta * (distance_from_base ** (beta - 1))
            if denominator <= 0:
                return float('inf')
            w_star = (2.0 / denominator) ** (1.0 / beta)
            return max(1.0, w_star)
        except (ZeroDivisionError, ValueError):
            return float('inf')

    # Beta-adaptive parameters
    non_base_nodes = [i for i in graph.nodes() if i != 0]
    max_gold = max((remaining_gold[i] for i in non_base_nodes), default=1000.0)

    if beta >= 1.5:
        max_weight_per_trip = max(200.0, max_gold * 1.2)
        heavy_gold_penalty_multiplier = 0.02
    elif beta >= 1.2:
        max_weight_per_trip = max(300.0, max_gold * 1.5)
        heavy_gold_penalty_multiplier = 0.015
    else:
        max_weight_per_trip = max(500.0, max_gold * 2.0)
        heavy_gold_penalty_multiplier = 0.01

    # Main collection loop
    while any(remaining_gold[i] > 0 for i in graph.nodes() if i != 0):
        # Ensure we're at base before starting a new trip
        go_to_base()
        cities_collected_in_trip = 0

        while True:
            cities_with_gold = [i for i in graph.nodes() if i != 0 and remaining_gold[i] > 0]
            if not cities_with_gold:
                break

            # Candidate filtering
            K = 6
            city_distances = []
            for city in cities_with_gold:
                dist_from_cur = get_shortest_distance(current_city, city)
                dist_from_base = get_shortest_distance(0, city)
                within_radius = dist_from_base > 0 and dist_from_cur <= 0.8 * dist_from_base
                city_distances.append((city, dist_from_cur, dist_from_base, within_radius))

            city_distances.sort(key=lambda x: x[1])
            radius_candidates = [city for city, _, _, within_radius in city_distances if within_radius]
            nearest_candidates = [city for city, _, _, _ in city_distances[:K]]
            candidate_cities = list(dict.fromkeys(radius_candidates + nearest_candidates))

            if not candidate_cities and city_distances:
                candidate_cities = [city for city, _, _, _ in city_distances[:K]]
            if not candidate_cities:
                candidate_cities = cities_with_gold

            # SAFETY: never select current_city as candidate (prevents A->A in the output)
            candidate_cities = [c for c in candidate_cities if c != current_city]

            # Cost of returning to base now
            path_to_base_now = get_shortest_path(current_city, 0)
            cost_return_now = path_cost(path_to_base_now, carried_gold) if path_to_base_now else 0.0

            best_city = None
            best_path = None
            best_score = float('inf')
            best_marginal_cost = float('inf')
            best_take_amount = 0.0

            for city in candidate_cities:
                path_to_city = get_shortest_path(current_city, city)
                if path_to_city is None:
                    continue

                gold_at_city = remaining_gold[city]
                dist_to_base = get_shortest_distance(city, 0)

                optimal_load_cap = calculate_optimal_load_cap(dist_to_base)
                visits = repeat_visits.get(city, 0)
                # Ramp pickup if we keep revisiting the same city (reduces huge micro-trip paths)
                ramped_cap = optimal_load_cap * (1.25 ** visits)
                take_amount = min(gold_at_city, max(1.0, ramped_cap))
                new_weight = carried_gold + take_amount

                if new_weight > max_weight_per_trip:
                    # Try to shrink pickup to fit weight budget instead of skipping
                    take_amount = max(0.0, max_weight_per_trip - carried_gold)
                    if take_amount <= 0.0:
                        continue
                    new_weight = carried_gold + take_amount

                # Option A: return now, separate trip later
                path_base_to_city = get_shortest_path(0, city)
                path_to_base_from_city = get_shortest_path(city, 0)
                if not path_base_to_city or not path_to_base_from_city:
                    continue

                option_a = (
                    cost_return_now
                    + path_cost(path_base_to_city, 0.0)
                    + path_cost(path_to_base_from_city, take_amount)
                )

                # Option B: do it now
                option_b = (
                    path_cost(path_to_city, carried_gold)
                    + path_cost(path_to_base_from_city, new_weight)
                )

                delta = option_b - option_a
                penalty = heavy_gold_penalty_multiplier * gold_at_city * dist_to_base
                score = delta + penalty

                if delta < 0 and score < best_score:
                    best_score = score
                    best_marginal_cost = delta
                    best_city = city
                    best_path = path_to_city
                    best_take_amount = take_amount

            if best_city is None:
                # Force progress if at base and still nothing beats baseline
                if current_city == 0 and cities_with_gold:
                    best_city = None
                    min_dist = float('inf')
                    for city in cities_with_gold:
                        dist = get_shortest_distance(0, city)
                        if dist < min_dist:
                            pth = get_shortest_path(0, city)
                            if pth is not None:
                                min_dist = dist
                                best_city = city
                                best_path = pth

                    if best_city is not None:
                        dist_to_base = get_shortest_distance(best_city, 0)
                        optimal_load_cap = calculate_optimal_load_cap(dist_to_base)
                        best_take_amount = min(remaining_gold[best_city], max(1.0, optimal_load_cap))
                        best_marginal_cost = 0.0

                if best_city is None:
                    break

            # Stop condition
            if best_marginal_cost >= 0:
                if not (current_city == 0 and cities_collected_in_trip == 0):
                    break

            # Move to the best city
            if current_city != best_city:
                move_along(best_path)

            # Collect gold (partial pickup)
            gold_to_take = min(remaining_gold[best_city], best_take_amount if best_take_amount > 0 else 1.0)
            append_visit(best_city, gold_to_take)

            repeat_visits[best_city] = repeat_visits.get(best_city, 0) + 1

            remaining_gold[best_city] -= gold_to_take
            carried_gold += gold_to_take
            current_city = best_city
            cities_collected_in_trip += 1

        go_to_base()

    # Final: ensure we end at base (without duplicate 0->0)
    go_to_base()
    if path[-1][0] != 0:
        append_visit(0, 0.0)

    # Final safety: merge any consecutive duplicates (paranoia)
    clean = [path[0]]
    for city, gold in path[1:]:
        if city == clean[-1][0]:
            clean[-1] = (city, clean[-1][1] + gold)
        else:
            clean.append((city, gold))

    if clean[0] != (0, 0.0):
        clean.insert(0, (0, 0.0))

    return clean