# Gold Collection Routing – Heuristic Solver
**Author:** Ali Bavi Fard

---

## Overview

This project tackles a constrained routing and optimization problem defined in the provided `Problem.py`.  
The goal is to collect **all gold from all cities** in a weighted graph and return it to the **base city (node 0)**, while minimizing a **non‑linear travel cost** that depends on both distance and carried weight.

The challenge is not only to find valid paths, but to decide **when it is worth carrying more gold** and **when it is better to return to base**.

This repository contains my final solution implemented in `s339414.py`.

---

## Problem Summary (in my own words)

- The map is a **graph**:
  - Nodes = cities
  - Edges = legal moves with distance `dist`
- City `0` is the **base**
- Every other city has some amount of **gold**
- You start at the base with **0 carried gold**
- You must:
  - collect **all gold**
  - **only unload at the base**
  - end the path at `(0, 0)`

### Output format
The solution must return a list of tuples:

```python
[(city_1, gold_taken_1), (city_2, gold_taken_2), ..., (0, 0)]
```

Each tuple means:
1. Move from the previous city to `city_i`
2. After arriving, take `gold_taken_i` gold from that city

Taking `0` gold is allowed and is used to **pass through cities** without collecting.

---

## Cost Model (Very Important)

Each **edge traversal** `(u → v)` while carrying `w` gold costs:

```
cost = dist(u,v) + (alpha * dist(u,v) * w) ** beta
```

Implications:
- Carrying gold makes movement expensive
- The penalty grows **super‑linearly** when `beta > 1`
- Long moves while heavy are very costly
- Long moves while empty are cheap

Because of this, blindly collecting many cities in one trip is usually a bad idea.

---

## Baseline

The provided `Problem.baseline()` method implements a very safe strategy:

- For each city:
  - go from base → city (empty)
  - take all its gold
  - return to base

This is always valid, but often sub‑optimal.
My solution is designed to **never be worse than the baseline**, and to beat it when the graph structure allows.

---

## My Approach

### Key idea
Instead of visiting one city per trip, I build **trips**:

```
0 → city A → city B → ... → 0
```

but **only** if adding another city is worth it.

The core principle is:
> Only add another city to the current trip if the extra cost is small compared to returning to base now.

### Main components

#### 1. Shortest‑path preprocessing
I precompute:
- shortest paths between all pairs of cities
- shortest distances between all pairs

This avoids repeated Dijkstra calls and makes cost evaluation fast.

#### 2. Trip‑based greedy strategy
While there is still gold left:
1. Start at the base with zero weight
2. Repeatedly:
   - evaluate all remaining cities
   - compute the **marginal cost** of adding each city
   - choose the city with the smallest adjusted marginal cost
3. Stop the trip when adding a city becomes too expensive
4. Return to base and unload

#### 3. Marginal cost reasoning
For a candidate city `c`:

```
marginal_cost =
    (cost to go current → c with current weight)
  + (cost to return c → base with increased weight)
  − (cost to return current → base now)
```

If this value is large and positive, the city is skipped.

#### 4. Heavy‑gold penalty
To avoid picking large gold too early, I add a small bias:

```
penalty = λ * gold_at_city * distance_to_base
```

This naturally prioritizes:
- small gold first
- heavy gold later in the trip or in its own trip

#### 5. Safe stop rule
- At base: allow at least one pickup
- Away from base: stop when marginal cost becomes too large relative to the cost of returning

This ensures robustness and prevents catastrophic over‑loading.

---

## Why matching the baseline is OK

On some instances (especially with large gold and `beta ≈ 1.5`):
- carrying extra gold becomes extremely expensive
- the baseline strategy is actually near‑optimal

In those cases, my solver **correctly chooses not to bundle cities**, resulting in the same cost as the baseline.

This is expected behavior and a sign that the stop condition is working as intended.

---

## Properties of the Solution

✔ Always produces **legal moves**  
✔ Never drops gold outside the base  
✔ Always collects **all gold**  
✔ Always ends at `(0, 0)`  
✔ Never worse than the baseline  
✔ Sometimes better when the graph allows bundling  

