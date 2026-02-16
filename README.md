# Gold Collection Routing

This project was done by us: **s339414**(Ali Bavi Fard) and **s339144**(Siavash Sanaee Poor) .

This repository contains our solution for the **“collect all gold and return to base”** routing problem defined by the provided `Problem.py`.

---

## Problem recap

- The map is a **weighted graph** of cities (nodes) connected by roads (edges).
- City `0` is the **base**.
- Each other city contains some amount of **gold**.
- The solver must output a **legal path** that:
  - starts at base `0`,
  - collects **all** gold from every non‑base city,
  - ends at base `0`,
  - and moves **edge‑by‑edge** (no teleporting / no missing edges).

### Output format
The solver returns a list of tuples:

```python
[(city, gold_taken), (city, gold_taken), ...]
```

`gold_taken` is how much gold is collected at that visit (can be partial).

---

## Cost model (key difficulty)

For a single move along an edge `(i → j)` while carrying weight `g`:

```python
cost = dist + (alpha * dist * weight) ** beta
```

- Moving empty (`g = 0`) costs just the distance.
- Carrying gold increases the cost, and for `beta > 1` the penalty becomes **strongly non‑linear**.

---

## Baseline strategy

The baseline (provided in `Problem.py`) is essentially:
- for each city independently: `0 → city → 0` and collect all its gold in one trip.

This is always valid, but it can be inefficient when `beta > 1` because carrying a large amount for long distances becomes very expensive.

---

## Our approach (how it works)

Our implementation is **trip‑based** and **baseline‑aware**, and it is designed to always produce admissible paths.

### 1) Admissible path construction (no invalid A→B steps)
Whenever we decide to go from the current city to a target city, we:
- compute the **shortest path** as a list of nodes,
- append *every intermediate node* to the output path with pickup `0.0`.

This guarantees that **every consecutive step in the output is a real edge**.

We also prevent `A→A` steps by merging consecutive identical nodes (and we also filter out `current_city` from candidates).

### 2) Shortest paths: lazy single‑source cache (fast, not all‑pairs)
We do **not** precompute all‑pairs shortest paths (too expensive).
Instead, we use a cache:

- the first time we need shortest paths from a source `u`, we run:
  `nx.single_source_dijkstra(graph, source=u, weight="dist")`
- we store both:
  - shortest paths from `u` to all nodes
  - shortest distances from `u` to all nodes fileciteturn15file1

This makes repeated evaluations fast without huge upfront cost.

### 3) Trip‑based collection loop
We repeatedly perform trips:

1. Ensure we are at base (`go_to_base()` resets carried weight to 0).
2. While the trip is still beneficial:
   - choose a next city among candidates,
   - possibly take **partial gold**,
   - update carried weight.
3. Return to base and start a new trip.
4. Stop only when **all gold is collected**. fileciteturn15file1

### 4) Candidate filtering (keep decisions local)
Evaluating every city every step is slow, so we restrict candidates using:

- **Radius rule:** keep cities where  
  `dist(current, city) ≤ 0.8 * dist(base, city)`
- plus the **K nearest** cities by `dist(current, city)` with `K = 6` fileciteturn15file1

This keeps the solver fast and focused.

### 5) Partial pickup
splitting large loads into smaller loads can reduce the non‑linear penalty.

For each candidate city, we compute an “optimal-ish” load cap `w*` based on the city’s distance to base, and choose:

```python
take = min(remaining_gold[city], max(1.0, ramped_cap))
```

To avoid extremely small micro‑pickups, the cap is **ramped up** with repeated visits:

```python
ramped_cap = optimal_cap * (1.25 ** visits)
``` 

### 6) Baseline‑aware decision rule (Option A vs Option B)
For each candidate city we compare:

**Option A (baseline‑style later):**
- return to base now,
- later do a separate trip `0 → city → 0` for the chosen pickup.

**Option B (do it now):**
- go to `city` now with current carried weight,
- pick gold,
- return to base with increased carried weight.

We compute:

```python
delta = option_b - option_a
```

- If `delta < 0`, doing it now is better than postponing baseline‑style.
- If no city yields `delta < 0`, we typically end the trip and return base.
- If we are at base and still need progress, we force a safe move to the nearest reachable gold city.

We also add a small penalty term to discourage selecting very heavy & far cities too early (tie‑breaking / stability).

### 7) Final cleanup
At the end we:
- ensure the path ends at base,
- merge any accidental consecutive duplicates (paranoia cleanup).

---

## Experimental results (sample benchmark)

The following table shows results from our local tests (your provided summary):

This Test took 33 minutes to finish with the results you see below:

| Test | Cities | Density | Alpha | Beta | Baseline | Solution | Improve % | Status |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 100 | 0.2 | 1 | 1 | 25266.405619 | 25266.405619 | -0.00% | Equal |
| 2 | 100 | 0.2 | 2 | 1 | 50425.309618 | 50425.309618 | +0.00% | Equal |
| 3 | 100 | 0.2 | 1 | 2 | 5334401.927003 | 48236.169897 | +99.10% | Better |
| 4 | 100 | 1.0 | 1 | 1 | 18266.185796 | 18266.185796 | +0.00% | Equal |
| 5 | 100 | 1.0 | 2 | 1 | 36457.918462 | 36457.918462 | -0.00% | Equal |
| 6 | 100 | 1.0 | 1 | 2 | 5404978.088996 | 35601.959271 | +99.34% | Better |
| 7 | 1000 | 0.2 | 1 | 1 | 195402.958104 | 195399.306350 | +0.00% | Better |
| 8 | 1000 | 0.2 | 2 | 1 | 390028.721263 | 390027.967438 | +0.00% | Better |
| 9 | 1000 | 0.2 | 1 | 2 | 37545927.702135 | 338677.888992 | +99.10% | Better |
| 10 | 1000 | 1.0 | 1 | 1 | 192936.233777 | 192925.978256 | +0.01% | Better |
| 11 | 1000 | 1.0 | 2 | 1 | 385105.641496 | 385101.490910 | +0.00% | Better |
| 12 | 1000 | 1.0 | 1 | 2 | 57580018.868725 | 380593.549304 | +99.34% | Better |

Interpretation:
- For `beta = 1`, improvements are typically ~0% (the cost is linear in carried weight, so partial splitting helps much less).
- For `beta = 2`, partial pickups strongly reduce the non‑linear penalty, giving very large improvements.

---
