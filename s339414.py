#WRITTEN BY ALI BAVI FARD
from Problem import Problem
import networkx as nx

def solution(p: Problem):
    # Get the graph and parameters
    graph = p.graph
    alpha = p.alpha
    beta = p.beta
    
    remaining_gold = {}
    for node in graph.nodes():
        remaining_gold[node] = graph.nodes[node]['gold']
    
    path = [(0, 0)]
    current_city = 0
    carried_gold = 0
    
    # Helper function to compute cost for a single edge move
    def edge_cost(u, v, weight):
        dist = graph[u][v]['dist']
        return dist + (alpha * dist * weight) ** beta
    
    # Precompute all-pairs shortest paths and distances
    # This avoids repeated shortest_path calls in loops
    shortest_paths_cache = {}
    shortest_distances_cache = {}
    
    for source in graph.nodes():
        try:
            # Compute shortest paths from source to all nodes
            paths_from_source = nx.single_source_dijkstra_path(graph, source=source, weight='dist')
            distances_from_source = nx.single_source_dijkstra_path_length(graph, source=source, weight='dist')
            
            for target in graph.nodes():
                if target in paths_from_source:
                    shortest_paths_cache[(source, target)] = paths_from_source[target]
                    shortest_distances_cache[(source, target)] = distances_from_source[target]
        except nx.NetworkXNoPath:
            pass
    
    # Helper function to get cached shortest path
    def get_shortest_path(u, v):
        """Get cached shortest path from u to v"""
        if u == v:
            return [u]
        return shortest_paths_cache.get((u, v), None)
    
    # Helper function to get cached shortest distance
    def get_shortest_distance(u, v):
        """Get cached shortest distance from u to v"""
        if u == v:
            return 0.0
        return shortest_distances_cache.get((u, v), float('inf'))
    
    # Trip-building approach: collects multiple cities per trip before returning to base
    # Continue until all gold is collected
    while any(remaining_gold[i] > 0 for i in graph.nodes() if i != 0):
        # Ensure we're at base before starting a new trip
        if current_city != 0:
            # Return to base first
            path_to_base = get_shortest_path(current_city, 0)
            if path_to_base:
                for i in range(len(path_to_base) - 1):
                    u, v = path_to_base[i], path_to_base[i+1]
                    path.append((v, 0))
                    current_city = v
            if path[-1] != (0, 0):
                path.append((0, 0))
            carried_gold = 0  # Unload at base
        
        cities_collected_in_trip = 0  # Track how many cities we've collected in this trip
        while True:
            # Find cities with remaining gold
            cities_with_gold = [i for i in graph.nodes() if i != 0 and remaining_gold[i] > 0]
            
            if not cities_with_gold:
                break
            
            # Calculates cost of returning to base
            path_to_base_now = get_shortest_path(current_city, 0)
            cost_return_now = 0
            if path_to_base_now:
                for i in range(len(path_to_base_now) - 1):
                    cost_return_now += edge_cost(path_to_base_now[i], path_to_base_now[i+1], carried_gold)
            
            # Find the city with the smallest modified marginal cost
            # Postpone heavy gold unless it's near the end
            best_city = None
            best_path = None
            best_score = float('inf')
            best_marginal_cost = float('inf')
            lambda_penalty = 0.01  # Penalty weight for heavy gold far from base
            
            for city in cities_with_gold:
                # Get shortest path from current city to target city
                path_to_city = get_shortest_path(current_city, city)
                if path_to_city is None:
                    continue
                
                # Calculate cost to reach this city with current weight
                cost_to_city = 0
                for i in range(len(path_to_city) - 1):
                    cost_to_city += edge_cost(path_to_city[i], path_to_city[i+1], carried_gold)
                
                # Get gold at this city
                gold_at_city = remaining_gold[city]
                new_weight = carried_gold + gold_at_city
                
                # Calculate cost to return to base from city with new weight
                path_to_base_from_city = get_shortest_path(city, 0)
                if path_to_base_from_city is None:
                    continue
                
                cost_return_from_city = 0
                dist_to_base = get_shortest_distance(city, 0)  # Use cached distance
                for i in range(len(path_to_base_from_city) - 1):
                    cost_return_from_city += edge_cost(path_to_base_from_city[i], path_to_base_from_city[i+1], new_weight)
                
                # Total cost if we add this city: go to city, then return to base
                cost_add = cost_to_city + cost_return_from_city
                
                # Marginal cost: extra cost of adding this city vs returning now
                marginal_cost = cost_add - cost_return_now
                
                # Penalize heavy gold far from base
                # This encourages collecting small gold first, and postponing heavy gold
                penalty = lambda_penalty * gold_at_city * dist_to_base
                score = marginal_cost + penalty
                
                if score < best_score:
                    best_score = score
                    best_marginal_cost = marginal_cost  # Keep track of actual marginal cost for stop condition
                    best_city = city
                    best_path = path_to_city
            
            if best_city is None:
                break  # No city found, return to base

            # Stop if marginal cost is too high relative to return cost
            # Stop if marginal cost is positive and exceeds a fraction of return cost
            # This adapts to the problem scale (distances, alpha, beta)
            if cost_return_now == 0:
                # At base: allow at least one pickup, then use absolute threshold
                if cities_collected_in_trip > 0 and best_marginal_cost > 10.0:
                    break  # Use absolute threshold when at base after first pickup
            else:
                # Not at base use relative threshold
                if best_marginal_cost > 0 and best_marginal_cost > 0.05 * cost_return_now:
                    break  # Return to base now - adding city is too expensive relative to returning
            
            # Move to the best city
            if current_city != best_city:
                for i in range(len(best_path) - 1):
                    u, v = best_path[i], best_path[i+1]
                    path.append((v, 0))
                    current_city = v
            
            # Collect gold from this city
            gold_to_take = remaining_gold[best_city]
            if path and path[-1] == (best_city, 0):
                path[-1] = (best_city, gold_to_take)
            else:
                path.append((best_city, gold_to_take))
            remaining_gold[best_city] = 0
            carried_gold += gold_to_take
            current_city = best_city
            cities_collected_in_trip += 1  # Track that we collected a city in this trip
        
        # Return to base to unload after trip
        if current_city != 0:
            path_to_base = get_shortest_path(current_city, 0)
            if path_to_base:
                for i in range(len(path_to_base) - 1):
                    u, v = path_to_base[i], path_to_base[i+1]
                    path.append((v, 0))
                    current_city = v
            
            # At base, all gold is automatically unloaded
            if path[-1] != (0, 0):
                path.append((0, 0))
            carried_gold = 0
            current_city = 0
    
    # Ensure we end at base with (0, 0)
    # If we're not already at base, return there
    if current_city != 0:
        path_to_base = get_shortest_path(current_city, 0)
        if path_to_base:
            for i in range(len(path_to_base) - 1):
                u, v = path_to_base[i], path_to_base[i+1]
                path.append((v, 0))
    
    # Ensures the path ends with (0, 0)
    if not path or path[-1] != (0, 0):
        path.append((0, 0))
    
    return path
