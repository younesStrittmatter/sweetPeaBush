from itertools import product
from typing import Dict, Sequence, List, Any, Union, Tuple, Iterable, Optional
import numpy as np
from joblib import Parallel, delayed


# ============================================================
# Domain construction (with lag) and data aggregation
# ============================================================

def enumerate_domain_with_lag(
    factors: Dict[str, Sequence[Any]],
    lag: int = 0
) -> Tuple[List[str], List[Tuple[Any, ...]], Dict[Tuple[Any, ...], int]]:
    """
    Build a domain over windows of length lag+1 using the provided subset of factors.
    Order of tuple key is: (vals_t, vals_t-1, ..., vals_t-lag), flattened by factor order.

    Returns:
        names: factor names in the insertion order of 'factors'
        domain: list of flattened tuples (rows)
        row_to_idx: mapping key -> row index
    """
    names = list(factors.keys())
    per_t = [list(factors[n]) for n in names]                 # one time step
    one = list(product(*per_t))                                # tuples of len(names)
    window = list(product(*([one] * (lag + 1))))               # (lag+1)-fold cartesian
    flat_domain: List[Tuple[Any, ...]] = []
    for row in window:  # row is (vals_t, vals_t-1, ..., vals_t-lag), each a tuple of len(names)
        flat = []
        for r in row:
            flat.extend(r)
        flat_domain.append(tuple(flat))
    row_to_idx = {row: i for i, row in enumerate(flat_domain)}
    return names, flat_domain, row_to_idx


def trials_to_indices_with_lag(
    trials: List[Dict[str, Any]],
    names: List[str],
    row_to_idx: Dict[Tuple[Any, ...], int],
    lag: int = 0
) -> np.ndarray:
    """
    Map trials to domain row indices with lag windows.
    For each t in [lag .. T-1], build key = (vals at t, t-1, ... , t-lag) over 'names'.
    """
    idxs: List[int] = []
    T = len(trials)
    for t in range(lag, T):
        key: List[Any] = []
        for offset in range(0, lag + 1):
            tt = t - offset
            row = trials[tt]
            key.extend(row[n] for n in names)
        k = tuple(key)
        idxs.append(row_to_idx[k])
    return np.asarray(idxs, dtype=np.int32)


def aggregate_by_row(N: int, idx: np.ndarray, rt: np.ndarray, acc: np.ndarray):
    counts  = np.bincount(idx, minlength=N).astype(np.int64)
    sum_rt  = np.bincount(idx, weights=rt, minlength=N).astype(float)
    sum_rt2 = np.bincount(idx, weights=rt * rt, minlength=N).astype(float)
    sum_acc = np.bincount(idx, weights=acc, minlength=N).astype(float)
    return counts, sum_rt, sum_rt2, sum_acc


# ============================================================
# Fast stats from aggregates
# ============================================================

def t_from_sums(n, s, s2):
    m = s / np.maximum(n, 1)
    v = (s2 - n * m * m) / np.maximum(n - 1, 1)  # unbiased
    return m, v

def welch_t(n1, s1, s12, n2, s2, s22):
    m1, v1 = t_from_sums(n1, s1, s12)
    m2, v2 = t_from_sums(n2, s2, s22)
    se = np.sqrt(v1 / np.maximum(n1, 1) + v2 / np.maximum(n2, 1))
    return (m1 - m2) / np.where(se > 0, se, np.inf)

def z_two_prop(x1, n1, x2, n2):
    p1 = x1 / np.maximum(n1, 1)
    p2 = x2 / np.maximum(n2, 1)
    p  = (x1 + x2) / np.maximum(n1 + n2, 1)
    se = np.sqrt(p * (1 - p) * (1 / np.maximum(n1, 1) + 1 / np.maximum(n2, 1)))
    return (p1 - p2) / np.where(se > 0, se, np.inf)


# ============================================================
# RGS enumeration of exactly-k partitions (+ safe prefixing)
# ============================================================

def rgs_prefixes(N: int, k: int, depth: int, min_block: int = 1) -> List[List[int]]:
    """
    Canonical prefixes for splitting the RGS search tree across workers.
    Clamps depth to [1..N].
    """
    depth = max(1, min(depth, N))
    if depth == 1:
        return [[0]]
    prefixes = [[0]]  # g[0]=0
    for _ in range(1, depth):
        new = []
        for p in prefixes:
            max_used = max(p) if p else -1
            upper = min(max_used + 1, k - 1)
            for lab in range(0, upper + 1):
                new.append(p + [lab])
        prefixes = new
    return prefixes


def complete_from_prefix(prefix: List[int], N: int, k: int, min_block: int = 1) -> Iterable[List[int]]:
    """
    Finish an RGS partition from a given prefix (length m <= N).
    Ensures labels in 0..k-1; final partition uses exactly k labels (surjection) with min_block size.
    """
    m = len(prefix)
    if m > N:
        return  # nothing to yield
    if m == 0:
        prefix = [0]
        m = 1

    if m == N:
        max_used = max(prefix) if prefix else -1
        if max_used == k - 1:
            sizes = [0] * k
            for lab in prefix:
                if 0 <= lab < k:
                    sizes[lab] += 1
                else:
                    return
            if all(s >= min_block for s in sizes[:k]):
                yield prefix[:]
        return

    # Initialize g and sizes
    g = prefix[:] + [0] * (N - m)
    sizes = [0] * k
    for lab in prefix:
        if 0 <= lab < k:
            sizes[lab] += 1
        else:
            return
    max_used = max(prefix) if prefix else 0

    # Backtrack from position m-1, so nxt starts at m
    def bt(i: int, max_used_local: int):
        nxt = i + 1
        upper = min(max_used_local + 1, k - 1)
        for lab in range(0, upper + 1):
            g[nxt] = lab
            sizes[lab] += 1
            remain = (N - 1) - nxt
            new_max = max(max_used_local, lab)
            needed = (k - 1 - new_max)
            feasible = (remain >= max(0, needed))
            if feasible:
                if nxt == N - 1:
                    if new_max == k - 1 and all(s >= min_block for s in sizes[:k]):
                        yield g[:]
                else:
                    yield from bt(nxt, new_max)
            sizes[lab] -= 1

    yield from bt(m - 1, max_used)


# ============================================================
# Lag necessity filtering
# ============================================================

def _projection_classes(domain: List[Tuple[Any, ...]], names: List[str], factors: Dict[str, Sequence[Any]], lag_from: int, lag_to: int) -> List[List[int]]:
    """
    Build equivalence classes of the lag_from domain after projecting to a smaller lag_to.
    We keep the first (lag_to+1)*len(names) entries of each flattened tuple key.
    """
    assert 0 <= lag_to < lag_from
    step = len(names)
    keep_len = (lag_to + 1) * step
    buckets: Dict[Tuple[Any, ...], List[int]] = {}
    for i, row in enumerate(domain):
        key = row[:keep_len]
        buckets.setdefault(key, []).append(i)
    return list(buckets.values())

def _partition_is_union_of_classes(g: List[int], classes: List[List[int]]) -> bool:
    """
    True iff each class is contained within a single block label in partition g.
    """
    for cls in classes:
        if not cls:
            continue
        label0 = g[cls[0]]
        for j in cls[1:]:
            if g[j] != label0:
                return False
    return True

def partition_needs_lag(
    g: List[int],
    domain: List[Tuple[Any, ...]],
    names: List[str],
    factors: Dict[str, Sequence[Any]],
    lag: int
) -> bool:
    """
    A partition 'needs' lag if it cannot be expressed for any smaller lag'<lag.
    """
    if lag <= 0:
        return True
    for l in range(0, lag):
        classes = _projection_classes(domain, names, factors, lag_from=lag, lag_to=l)
        if _partition_is_union_of_classes(g, classes):
            # Expressible at smaller lag l -> does NOT need lag
            return False
    return True


# ============================================================
# Evaluate a partition (exactly-k groups) from aggregates
# ============================================================

def eval_partition(
    g: List[int],
    counts: np.ndarray,
    sum_rt: np.ndarray,
    sum_rt2: np.ndarray,
    sum_acc: np.ndarray,
    min_block_trials: int = 1
):
    k = max(g) + 1
    N = len(g)
    n  = np.zeros(k, dtype=float)
    s  = np.zeros(k, dtype=float)
    s2 = np.zeros(k, dtype=float)
    xa = np.zeros(k, dtype=float)
    for i in range(N):
        c = counts[i]
        if c == 0:
            continue
        lab = g[i]
        n[lab]  += c
        s[lab]  += sum_rt[i]
        s2[lab] += sum_rt2[i]
        xa[lab] += sum_acc[i]

    if np.any(n < min_block_trials):
        return None

    best_t = 0.0
    best_z = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            t = abs(welch_t(n[a], s[a], s2[a], n[b], s[b], s2[b]))
            z = abs(z_two_prop(xa[a], n[a], xa[b], n[b]))
            if t > best_t: best_t = t
            if z > best_z: best_z = z

    score = max(best_t, best_z)
    return score, best_t, best_z


# ============================================================
# Scan exactly-k partitions in parallel (with lag)
# ============================================================

def scan_exactly_k_partitions(
    factors: Dict[str, Sequence[Any]],
    trials: List[Dict[str, Any]],
    rt: Union[List[float], np.ndarray],
    acc: Union[List[int], np.ndarray],
    *,
    k: int,
    lag: int = 0,
    require_needs_lag: bool = True,
    prefix_depth: int = 6,
    min_block_cells: int = 1,
    min_block_trials: int = 1,
    top_k: int = 100,
    n_jobs: int = -1
):
    if k < 2:
        return {"domain": [], "top": [], "prefixes": 0, "scanned_shards": 0}

    names, domain, row_to_idx = enumerate_domain_with_lag(factors, lag=lag)
    N = len(domain)
    if N == 0 or k > N:
        return {"domain": domain, "top": [], "prefixes": 0, "scanned_shards": 0}

    # Build indices and trim data according to lag
    idx = trials_to_indices_with_lag(trials, names, row_to_idx, lag=lag)
    rt_arr  = np.asarray(rt)[lag:]  # align to windows
    acc_arr = np.asarray(acc)[lag:]
    counts, sum_rt, sum_rt2, sum_acc = aggregate_by_row(N, idx, rt=rt_arr, acc=acc_arr)

    depth_eff = max(1, min(prefix_depth, N))
    prefixes = rgs_prefixes(N, k, depth=depth_eff, min_block=min_block_cells)

    def worker(prefix):
        best = []
        for g in complete_from_prefix(prefix, N, k, min_block=min_block_cells):
            if require_needs_lag and lag > 0:
                if not partition_needs_lag(g, domain, names, factors, lag):
                    continue
            r = eval_partition(g, counts, sum_rt, sum_rt2, sum_acc, min_block_trials=min_block_trials)
            if r is None:
                continue
            score, tmax, zmax = r
            best.append((score, tmax, zmax, g))
            if len(best) > top_k * 2:
                best.sort(key=lambda x: -x[0])
                best = best[:top_k]
        best.sort(key=lambda x: -x[0])
        return best[:top_k]

    if len(prefixes) == 0:
        return {"domain": domain, "top": [], "prefixes": 0, "scanned_shards": 0}

    shards = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(worker)(p) for p in prefixes)
    pool = [item for shard in shards for item in shard]
    pool.sort(key=lambda x: -x[0])

    return {
        "domain": domain,
        "top": pool[:top_k],   # (score, tmax, zmax, g)
        "prefixes": len(prefixes),
        "scanned_shards": len(shards)
    }


# ============================================================
# High-level scan (k can be int, list, or 'ALL')
# ============================================================

def scan(
    factors: Dict[str, Sequence[Any]],
    trials: List[Dict[str, Any]],
    rt: Union[List[float], np.ndarray],
    acc: Union[List[int], np.ndarray],
    *,
    k: Union[int, List[int], str] = 2,
    lag: int = 0,
    require_needs_lag: bool = True,
    prefix_depth: int = 6,
    min_block_cells: int = 1,
    min_block_trials: int = 1,
    top_k: int = 50,
    n_jobs: int = -1,
) -> Dict[int, dict]:
    """
    Scan factorial design for candidate derived factors.

    - 'factors' is the subset you want to base the split on (e.g., {'color': [...]})
    - 'lag' >= 0:
        0 -> current trial only
        1 -> (curr, prev)
        2 -> (curr, prev, prev2)
      With require_needs_lag=True, we filter out partitions expressible at any smaller lag.
    - 'k' can be:
        * int (e.g., 2)
        * list of ints (e.g., [2,3,4])
        * "ALL" (meaning all k from 2..N where N is #cells in the lagged domain)
    """
    names, domain, _ = enumerate_domain_with_lag(factors, lag=lag)
    N = len(domain)

    if isinstance(k, int):
        ks = [k]
    elif isinstance(k, list):
        ks = k
    elif isinstance(k, str) and k.upper() == "ALL":
        ks = list(range(2, max(2, N) + 1))
    else:
        raise ValueError("k must be int, list[int], or 'ALL'")

    results: Dict[int, dict] = {}
    for kk in ks:
        res = scan_exactly_k_partitions(
            factors=factors,
            trials=trials,
            rt=rt,
            acc=acc,
            k=kk,
            lag=lag,
            require_needs_lag=require_needs_lag,
            prefix_depth=prefix_depth,
            min_block_cells=min_block_cells,
            min_block_trials=min_block_trials,
            top_k=top_k,
            n_jobs=n_jobs
        )
        results[kk] = res
    return results


# ============================================================
# Demo / sanity check: congruency (lag=0), task switch (lag=1)
# ============================================================

# ==== JUST THE INPUTS / DEMO ====
# Put this at the bottom of your file (or run after your scan() is defined)

if __name__ == "__main__":
    import numpy as np
    from typing import Dict, Any, List

    rng = np.random.default_rng(0)

    # Full factorial design
    factors_full = {
        "color": ["red", "green", "blue", "yellow"],
        "word":  ["red", "green", "blue", "yellow"],
        "task":  ["color", "word"],  # included just to show you can also scan task switches if you want
    }

    T = 7000
    trials: List[Dict[str, Any]] = []
    rt: List[float] = []
    acc: List[int] = []

    # ---- Simulate data with:
    #  (A) Congruency main effect (congruent faster, more accurate)
    #  (B) Sequential congruency effect (Gratton): the congruency effect is smaller after INCONGRUENT trials
    #
    #  Let C_t = 1 if current is incongruent else 0
    #      P_t = 1 if previous is incongruent else 0
    #  RT mean = 600 + 60*C_t - 30*(C_t * P_t)
    #            ^^^^^ congruency effect       ^^^^^ interaction (shrinks effect after incongruent)
    #  This yields means:  CC=600, IC=660, CI=600, II=630  (effect 60 after prev congruent, 30 after prev incongruent)
    #
    #  We’ll also add modest accuracy differences in the same pattern.

    last_color = None
    last_word  = None

    for t in range(T):
        c = rng.choice(factors_full["color"])
        w = rng.choice(factors_full["word"])
        task = rng.choice(factors_full["task"])  # not used in the sequential effect demo, but present

        trials.append({"color": c, "word": w, "task": task})

        # congruency indicators
        cong_now  = int(c == w)                 # 1 = congruent, 0 = incongruent
        cong_prev = int((last_color == last_word)) if (last_color is not None and last_word is not None) else 0

        # recode to C_t (1=incongruent), P_t (1=prev incongruent)
        C_t = 1 - cong_now
        P_t = 1 - cong_prev

        # RT mean with sequential congruency interaction
        mu_rt = 600.0 + 60.0*C_t - 30.0*(C_t * P_t)
        rti = rng.normal(mu_rt, 50.0)

        # Accuracy with a parallel pattern (worse when incongruent; interaction reduces the penalty after incongruent)
        p_acc = 0.92 - 0.08*C_t + 0.04*(C_t * P_t)
        p_acc = float(np.clip(p_acc, 0.05, 0.99))
        acci = rng.choice([0, 1], p=[1.0 - p_acc, p_acc])

        rt.append(rti)
        acc.append(acci)

        last_color, last_word = c, w

    # ---------- Scan 1: find Congruency (lag=0) over color×word, k=2 ----------
    print("\n=== Scan: Congruency (color×word), lag=0, k=2 ===")
    res_cong = scan(
        factors={"color": factors_full["color"], "word": factors_full["word"]},
        trials=trials, rt=rt, acc=acc,
        k=[2],            # 2-way splits only
        lag=0,            # current trial only
        prefix_depth=6,
        min_block_cells=1,
        min_block_trials=80,
        top_k=5,
        n_jobs=-1
    )

    best2 = res_cong[2]["top"][0]    # (score, tmax, zmax, g)
    score, tmax, zmax, g = best2

    # Decode the groups for readability
    names0, domain_cw, _ = enumerate_domain_with_lag({"color": factors_full["color"], "word": factors_full["word"]}, lag=0)
    grp0 = [domain_cw[i] for i, lab in enumerate(g) if lab == 0]
    grp1 = [domain_cw[i] for i, lab in enumerate(g) if lab == 1]
    print("Best 2-way split score =", score)
    print("Group 0 (first 8):", grp0[:8], "...")
    print("Group 1 (first 8):", grp1[:8], "...")
    # With the simulated data, you should see the split align with congruent vs incongruent pairs.

    # ---------- Scan 2: find Sequential Congruency (lag=1) over color×word, k=2 ----------
    # NOTE: We are *not* changing your scan; just passing lag=1 on the same factors.
    # The best split will often show up either as:
    #   (a) previous congruent vs previous incongruent, or
    #   (b) congruency repeat vs switch.
    # Both are equivalent encodings of the interaction driving the Gratton effect.
    print("\n=== Scan: Sequential congruency (color×word), lag=1, k=2 ===")
    res_seq = scan(
        factors={"color": factors_full["color"], "word": factors_full["word"]},
        trials=trials, rt=rt, acc=acc,
        k=[2],
        lag=1,            # include previous trial’s (color, word)
        prefix_depth=6,
        min_block_cells=1,
        min_block_trials=80,
        top_k=5,
        n_jobs=-1
    )

    best2_seq = res_seq[2]["top"][0]
    score_s, tmax_s, zmax_s, g_s = best2_seq

    # Decode the lag=1 groups: domain rows are tuples (color_t, word_t, color_{t-1}, word_{t-1})
    names1, domain_cw_lag1, _ = enumerate_domain_with_lag({"color": factors_full["color"], "word": factors_full["word"]}, lag=1)
    grp0_s = [domain_cw_lag1[i] for i, lab in enumerate(g_s) if lab == 0]
    grp1_s = [domain_cw_lag1[i] for i, lab in enumerate(g_s) if lab == 1]
    print("Best 2-way split score =", score_s)

    # For interpretability, summarize each group by (prev_cong, curr_cong) counts
    def cong_from_pair(pair):
        # pair = (color, word)
        return int(pair[0] == pair[1])  # 1 if congruent else 0
    def summarize_group(rows):
        # rows are tuples: (c_t, w_t, c_prev, w_prev)
        cats = {("prevC_currC",0):0, ("prevC_currI",1):0, ("prevI_currC",2):0, ("prevI_currI",3):0}
        for (ct,wt,c0,w0) in rows:
            prevC = cong_from_pair((c0,w0))
            currC = cong_from_pair((ct,wt))
            key = ("prevC_currC" if prevC==1 and currC==1 else
                   "prevC_currI" if prevC==1 and currC==0 else
                   "prevI_currC" if prevC==0 and currC==1 else
                   "prevI_currI")
            cats[(key, ["prevC_currC","prevC_currI","prevI_currC","prevI_currI"].index(key))] += 1
        # sort by the canonical order to print nicely
        ordered = [(k[0], cats[k]) for k in sorted(cats.keys(), key=lambda x: x[1])]
        return ordered

    print("Group 0 composition (by prev/curr congruency):", summarize_group(grp0_s))
    print("Group 1 composition (by prev/curr congruency):", summarize_group(grp1_s))

    # In our simulated pattern, you should see that one group concentrates more on "prev congruent" cases,
    # or equivalently separates "repeat vs switch" of congruency. Either way, that's the sequential congruency effect emerging.
