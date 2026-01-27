# Gold Collection Routing 
This Project was done by me (s339414) and my friend (s339144)

This is my solution for the “collect all gold and return to base” routing problem defined by the provided `Problem.py`.

The short version:
- The map is a **graph** of cities (nodes) connected by roads (edges).
- City `0` is the **base**.
- Each other city has some amount of **gold**.
- I must produce a **legal path** that collects **all** gold and ends back at the base.
- Traveling while carrying gold is expensive because the cost grows **non‑linearly** with the carried weight.

This README explains what the code does and why.


## Problem recap (in my own words)

### Output format
My solver must return a list of tuples:

```python
[(city, gold_taken), (city, gold_taken), ...]
```

## Cost model (the key difficulty)

For a single move along an edge `(i → j)` while carrying weight `g`:

\[
c = d_{ij} + \left( \alpha \cdot d_{ij} \cdot g \right)^{\beta}
\]

Or in code:
```python
cost = dist + (alpha * dist * weight) ** beta
```

Where:
- `d_{ij}` (or `dist`) is the road distance between cities `i` and `j`
- `g` (or `weight`) is how much gold I'm carrying at that moment
- `α` (`alpha`) and `β` (`beta`) are fixed parameters provided by the instance

- Moving empty (`g = 0`) is cheap: cost = `d_{ij}`
- Moving while heavy can become **very** expensive, especially when `β > 1`

- Moving empty is cheap.
- Moving while heavy can become **very** expensive, especially when `beta > 1`.


## What the baseline does (and how I try to beat it)

The baseline strategy (inside `Problem.py`) basically does:
- For each city independently: `0 → city → 0` and pick all gold in one go.

That is always valid, but not always optimal because:
- It ignores the possibility of **partial pickups** (split heavy loads).
- It ignores the possibility of smart chaining when it is beneficial.

My code tries to outperform the baseline mainly using **partial pickups** and a **baseline-aware decision rule**.


## How my solver works (high-level)

### 1) Precompute shortest paths (speed)
The solver calls `nx.single_source_dijkstra_path` and `nx.single_source_dijkstra_path_length`
from every node once, and stores:
- shortest paths between all pairs
- shortest distances between all pairs

This makes later evaluation fast because I can reuse shortest paths instead of recomputing them.

### 2) Trip-based collection
I structure the solution as repeated “trips”:
- Start at base
- Visit one or more cities (sometimes picking partial gold)
- Return to base and unload
- Repeat until all gold is collected

### 3) Candidate filtering (keep decisions local)
At each step, I don’t evaluate all cities (that’s slow and often unnecessary).
I select candidates using:
- a “radius” rule: cities relatively closer to my current position than to base
- plus the `K` nearest cities (I set `K = 6`)

This keeps the solver fast and focuses on reasonable local moves.

### 4) Partial pickup (important improvement)
If `beta > 1`, carrying a huge amount in one go can be worse than splitting it into lighter trips.
So when I decide to visit a city, I may take **only part** of its gold.

In code:
- I compute an “optimal-ish” load cap based on the city’s distance to base.
- Then I take:
  ```python
  take_amount = min(remaining_gold[city], max(1.0, optimal_load_cap))
  ```
- If a city still has gold left, it stays in the remaining set and can be revisited later.

### 5) Baseline-aware decision rule
When I’m considering a city `city`, I compare two options:

**Option A (baseline-style later):**
- Return to base now
- Later do a separate trip `0 → city → 0` for the chosen pickup amount

**Option B (do it now):**
- Go to `city` now with current carried weight
- Return to base with increased carried weight

I compute:
```python
delta = option_b - option_a
```

- If `delta < 0`, doing it now is better than postponing it baseline-style.
- If `delta >= 0`, baseline-style is better, so I tend to stop the trip and return to base.

I also add a small penalty term to discourage choosing extremely heavy and far cities too early,
just to break ties and steer decisions gently.


## What I consider “success”
- The returned path is always legal (edge-by-edge moves via shortest paths).
- All gold is eventually collected (including partial pickups).
- The path ends at base `(0, 0)`.
- I beat the baseline on at least some instances (especially where partial pickups help).

Note: On some instances (high `beta`, very large gold values), the baseline can already be close to optimal,
so matching baseline is not necessarily a bug.


## File overview
- `s339414.py`: my solver. It exposes:
  ```python
  def solution(p: Problem):
      ...
      return path
  ```


## Small personal note
I set `K = 6` because it’s my favourite number


## Future improvements (if I had more time)
- Tune the candidate filtering (radius/K) based on `beta`
- Allow slight “epsilon” chaining even when `delta` is slightly positive (to avoid being too conservative)
- Add local search / swapping inside a trip for better ordering
