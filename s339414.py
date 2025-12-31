from Problem import Problem
import networkx as nx

def solution(p: Problem):
    # Get the graph and parameters
    graph = p.graph
    alpha = p.alpha
    beta = p.beta
    
    # remaining gold for each city
    remaining_gold = {}
    for node in graph.nodes():
        remaining_gold[node] = graph.nodes[node]['gold']
    
    path = [(0, 0)]
    current_city = 0
    carried_gold = 0.0
    
    # compute cost for a single edge move
    def edge_cost(u, v, weight):
        dist = graph[u][v]['dist']
        return dist + (alpha * dist * weight) ** beta
    
    # precompute all-pairs shortest paths and distances
    shortest_paths_cache = {}
    shortest_distances_cache = {}
    
    for source in graph.nodes():
        paths_from_source = nx.single_source_dijkstra_path(graph, source=source, weight='dist')
        distances_from_source = nx.single_source_dijkstra_path_length(graph, source=source, weight='dist')
        
        for t, pth in paths_from_source.items():
            shortest_paths_cache[(source, t)] = pth
        for t, dd in distances_from_source.items():
            shortest_distances_cache[(source, t)] = dd
    
    def get_shortest_path(u, v):
        return shortest_paths_cache.get((u, v))
    
    def get_shortest_distance(u, v):
        return shortest_distances_cache.get((u, v))
    
    def path_cost(nodes, w):
        """Compute cost of traversing a path with given weight."""
        if not nodes or len(nodes) < 2:
            return 0.0
        return sum(edge_cost(a, b, w) for a, b in zip(nodes, nodes[1:]))
    
    def move_along(pth):
        """Helper to move along a path, updating path and current_city."""
        nonlocal current_city
        if pth:
            for v in pth[1:]:
                path.append((v, 0))
                current_city = v
    
    def go_to_base():
        """Helper to return to base, updating path and state."""
        nonlocal current_city, carried_gold
        if current_city == 0:
            if path[-1] != (0, 0):
                path.append((0, 0))
            carried_gold = 0.0
            return
        pth = get_shortest_path(current_city, 0)
        move_along(pth)
        if path[-1] != (0, 0):
            path.append((0, 0))
        current_city = 0
        carried_gold = 0.0
    
    # This function calculates the optimal load cap for partial pickups
    # when beta > 1, splitting heavy loads into lighter trips can be cheaper
    def calculate_optimal_load_cap(distance_from_base):
        if beta <= 1.0 or distance_from_base <= 0:
            # Linear or sublinear: no benefit from splitting, take all
            return float('inf')
        
        try:
            denominator = (beta - 1) * alpha * beta * (distance_from_base ** (beta - 1))
            if denominator <= 0:
                return float('inf')
            
            w_star = (2.0 / denominator) ** (1.0 / beta)
            return max(1.0, w_star)  # At least take 1 unit
        except (ZeroDivisionError, ValueError):
            return float('inf')
    
    # Beta-adaptive parameters
    # Higher beta = more penalty for carrying weight, so be more conservative
    # Find max gold to ensure we can always collect at least one city
    max_gold = max(remaining_gold[i] for i in graph.nodes() if i != 0) if any(remaining_gold[i] > 0 for i in graph.nodes() if i != 0) else 1000.0
    
    if beta >= 1.5:
        # High beta: very conservative, avoid heavy loads
        # But ensure we can always collect at least the largest city
        max_weight_per_trip = max(200.0, max_gold * 1.2)  # At least 20% more than max gold
        heavy_gold_penalty_multiplier = 0.02  # Higher penalty for heavy gold
    elif beta >= 1.2:
        # Medium-high beta: moderate conservatism
        max_weight_per_trip = max(300.0, max_gold * 1.5)
        heavy_gold_penalty_multiplier = 0.015
    else:
        # Lower beta: can be more aggressive
        max_weight_per_trip = max(500.0, max_gold * 2.0)
        heavy_gold_penalty_multiplier = 0.01
    
    # Main collection loop
    while any(remaining_gold[i] > 0 for i in graph.nodes() if i != 0):
        # Ensure we're at base before starting a new trip
        go_to_base()
        
        # Build a trip
        cities_collected_in_trip = 0
        
        while True:
            cities_with_gold = [i for i in graph.nodes() if i != 0 and remaining_gold[i] > 0]
            
            if not cities_with_gold:
                break
            
            # Candidate filtering: only consider nearby cities to speed up and make better choices
            K = 6  # Number of nearest cities to consider (I choose 6 because its my favourite number)
            
            # Calculate distances from current city to all cities with gold
            city_distances = []
            for city in cities_with_gold:
                dist_from_cur = get_shortest_distance(current_city, city)
                dist_from_base = get_shortest_distance(0, city)
                
                # Filter: cities within radius OR will be in K nearest
                # Radius: dist(cur,j) <= 0.8 * dist(0,j)
                # This means city is closer to current position than 80% of its distance from base
                within_radius = dist_from_base > 0 and dist_from_cur <= 0.8 * dist_from_base
                
                city_distances.append((city, dist_from_cur, dist_from_base, within_radius))
            
            # Sort by distance from current city
            city_distances.sort(key=lambda x: x[1])
            
            # Take cities that are either:
            # 1. Within radius (dist(cur,j) <= 0.8 * dist(0,j)), OR
            # 2. K nearest cities
            radius_candidates = [city for city, _, _, within_radius in city_distances if within_radius]
            nearest_candidates = [city for city, _, _, _ in city_distances[:K]]
            
            # Combine and deduplicate
            candidate_cities = list(dict.fromkeys(radius_candidates + nearest_candidates))
            
            # if no candidates found, use K nearest regardless
            if not candidate_cities and city_distances:
                candidate_cities = [city for city, _, _, _ in city_distances[:K]]
            
            # if still no candidates, fallback to all cities (shouldn't happen, but just in case)
            if not candidate_cities:
                candidate_cities = cities_with_gold
            
            # Calculate cost of returning to base now
            path_to_base_now = get_shortest_path(current_city, 0)
            cost_return_now = path_cost(path_to_base_now, carried_gold) if path_to_base_now else 0
            
            # Find best city with baseline-aware marginal cost
            best_city = None
            best_path = None
            best_score = float('inf')
            best_marginal_cost = float('inf')
            best_take_amount = 0  # Track how much gold to take (for partial pickups)
            
            for city in candidate_cities:
                path_to_city = get_shortest_path(current_city, city)
                if path_to_city is None:
                    continue
                
                gold_at_city = remaining_gold[city]
                dist_to_base = get_shortest_distance(city, 0)
                
                # calculate optimal load cap for partial pickups
                optimal_load_cap = calculate_optimal_load_cap(dist_to_base)
                min_take = 1.0  # Minimum amount to take (avoid taking 0)
                
                # determine how much to take: partial pickup if beneficial
                take_amount = min(gold_at_city, max(min_take, optimal_load_cap))
                
                # if we can't take at least min_take, skip this city for now
                if take_amount < min_take:
                    continue
                
                new_weight = carried_gold + take_amount
                
                # check weight limit (beta-adaptive)
                if new_weight > max_weight_per_trip:
                    continue  # Skip if would exceed weight limit
                
                # Option A (baseline-style): return now, then later do separate trip for city
                path_base_to_city = get_shortest_path(0, city)
                path_to_base_from_city = get_shortest_path(city, 0)
                if not path_base_to_city or not path_to_base_from_city:
                    continue
                
                cost_base_to_city = path_cost(path_base_to_city, 0.0)
                cost_city_to_base = path_cost(path_to_base_from_city, take_amount)
                option_a = cost_return_now + cost_base_to_city + cost_city_to_base
                
                # Option B (do it now): cost(current→city, current_weight) + cost(city→0, new_weight)
                cost_to_city = path_cost(path_to_city, carried_gold)
                cost_return_from_city = path_cost(path_to_base_from_city, new_weight)
                option_b = cost_to_city + cost_return_from_city
                
                # Delta: negative means doing it now is better than baseline-style
                delta = option_b - option_a
                
                # Small penalty for tie-breaking (delta is the main metric)
                penalty = heavy_gold_penalty_multiplier * gold_at_city * dist_to_base
                score = delta + penalty
                
                # only consider cities where delta < 0 (beats baseline)
                if delta < 0:
                    if score < best_score:
                        best_score = score
                        best_marginal_cost = delta  # Store actual delta for stop condition
                        best_city = city
                        best_path = path_to_city
                        best_take_amount = take_amount  # Store how much to take
            
            if best_city is None:
                # if we're at base and can't find any city with delta < 0, 
                # we must collect something to make progress
                if current_city == 0 and cities_with_gold:
                    # find the closest city with gold (ignore weight limit as fallback)
                    best_city = None
                    min_dist = float('inf')
                    for city in cities_with_gold:
                        dist = get_shortest_distance(0, city)
                        if dist < min_dist:
                            path_to_city = get_shortest_path(0, city)
                            if path_to_city is not None:
                                min_dist = dist
                                best_city = city
                                best_path = path_to_city
                    # if we found a city, calculate baseline-aware delta with partial pickup
                    if best_city is not None:
                        gold_at_city = remaining_gold[best_city]
                        dist_to_base = get_shortest_distance(best_city, 0)
                        
                        # calculate optimal load cap
                        optimal_load_cap = calculate_optimal_load_cap(dist_to_base)
                        min_take = 1.0
                        best_take_amount = min(gold_at_city, max(min_take, optimal_load_cap))
                        
                        # at base with 0 weight: Option A = Option B, so delta = 0
                        best_marginal_cost = 0.0
                
                if best_city is None:
                    break
            
            # simple stop condition: baseline-aware delta
            if best_city is None:
                # no valid city found, break
                break
            
            # if best Delta < 0: take it (beats baseline)
            # Else: stop trip and return to base (baseline is better)
            if best_marginal_cost >= 0:
                # at base with no collections: must make at least one pickup to make progress
                if current_city == 0 and cities_collected_in_trip == 0:
                    # Allow one pickup even if delta >= 0 to ensure progress
                    pass
                else:
                    # stop trip and return to base because baseline is better
                    break
            
            # move to the best city
            if current_city != best_city:
                move_along(best_path)
            
            # collect gold (partial pickup if beneficial)
            # safety: if best_take_amount is 0 or invalid, take at least 1.0 or all remaining
            if best_take_amount <= 0:
                # calculate optimal amount on the fly
                dist_to_base = get_shortest_distance(best_city, 0)
                optimal_load_cap = calculate_optimal_load_cap(dist_to_base)
                best_take_amount = min(remaining_gold[best_city], max(1.0, optimal_load_cap))
            
            gold_to_take = min(remaining_gold[best_city], best_take_amount)
            
            if path and path[-1] == (best_city, 0):
                path[-1] = (best_city, gold_to_take)
            else:
                path.append((best_city, gold_to_take))
            remaining_gold[best_city] -= gold_to_take  # partial pickup: leave rest for later
            carried_gold += gold_to_take
            current_city = best_city
            cities_collected_in_trip += 1
        
        go_to_base()
        #go base after trip
    #go base after last trip
    go_to_base()
    
    return path