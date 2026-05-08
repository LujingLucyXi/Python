```python
# 1) QUESTION: Given a list of integers, return a dict mapping each integer to its frequency.
#    Write it using a for-loop and if/else (no collections.Counter).
def freq_map(nums):
    freq = {}
    for x in nums:
        if x in freq:
            freq[x] += 1
        else:
            freq[x] = 1
    return freq

# Solution demo:
# print(freq_map([1, 2, 2, 3, 3, 3]))  # {1: 1, 2: 2, 3: 3}


# 2) QUESTION: Flatten a 2D list (list of lists) into a 1D list using loops.
def flatten_2d(grid):
    out = []
    for row in grid:
        for item in row:
            out.append(item)
    return out

# print(flatten_2d([[1, 2], [], [3, 4]]))  # [1, 2, 3, 4]


# 3) QUESTION: Given a string, return the first non-repeating character (or None if none exists).
def first_non_repeating_char(s):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1

    for ch in s:
        if counts[ch] == 1:
            return ch
    return None

# print(first_non_repeating_char("swiss"))  # 'w'


# 4) QUESTION: Merge two dictionaries by summing values for matching keys.
#    Example: {"a":2,"b":1} + {"a":3,"c":4} -> {"a":5,"b":1,"c":4}
def merge_sum_dicts(a, b):
    out = {}
    for k, v in a.items():
        out[k] = v
    for k, v in b.items():
        if k in out:
            out[k] += v
        else:
            out[k] = v
    return out

# print(merge_sum_dicts({"a": 2, "b": 1}, {"a": 3, "c": 4}))  # {'a': 5, 'b': 1, 'c': 4}


# 5) QUESTION: Deduplicate a list while preserving order (use a set + loop).
def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out

# print(dedupe_preserve_order([3, 3, 1, 2, 1, 4]))  # [3, 1, 2, 4]


# 6) QUESTION: Given a list of (name, score), return the top-k by score descending,
#    tie-break by name ascending (use sorted()).
def top_k_scores(pairs, k):
    return sorted(pairs, key=lambda p: (-p[1], p[0]))[:k]

# print(top_k_scores([("bob", 10), ("ann", 10), ("zoe", 7)], 2))  # [('ann', 10), ('bob', 10)]


# 7) QUESTION: Implement binary search on a sorted list. Return index or -1 if not found.
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# print(binary_search([1, 3, 5, 7, 9], 7))  # 3


# 8) QUESTION: Rotate a square matrix 90 degrees clockwise (in-place).
def rotate_matrix_clockwise(mat):
    n = len(mat)
    # transpose
    for r in range(n):
        for c in range(r + 1, n):
            mat[r][c], mat[c][r] = mat[c][r], mat[r][c]
    # reverse each row
    for r in range(n):
        mat[r].reverse()
    return mat

# m = [[1,2,3],[4,5,6],[7,8,9]]
# rotate_matrix_clockwise(m)
# print(m)  # [[7,4,1],[8,5,2],[9,6,3]]


# 9) QUESTION: Given edges of an undirected graph, build an adjacency list dict and run BFS
#    from a start node, returning visit order.
def bfs_order(edges, start):
    adj = {}
    for a, b in edges:
        if a not in adj:
            adj[a] = []
        if b not in adj:
            adj[b] = []
        adj[a].append(b)
        adj[b].append(a)

    # deterministic traversal
    for node in adj:
        adj[node] = sorted(adj[node])

    visited = set()
    order = []
    queue = [start]
    visited.add(start)

    while queue:
        node = queue.pop(0)
        order.append(node)
        for nei in adj.get(node, []):
            if nei not in visited:
                visited.add(nei)
                queue.append(nei)
    return order

# print(bfs_order([(1,2),(2,3),(1,3),(3,4)], 1))  # [1, 2, 3, 4]


# 10) QUESTION: Parse a CSV-like string into a list of dict rows.
#     Input:
#       "name,age\nAnn,30\nBob,25"
#     Output:
#       [{"name":"Ann","age":"30"}, {"name":"Bob","age":"25"}]
def parse_simple_csv(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return []

    header = [h.strip() for h in lines[0].split(",")]
    rows = []

    for i, line in enumerate(lines[1:], start=1):
        parts = [p.strip() for p in line.split(",")]
        row = {}
        for j, col in enumerate(header):
            row[col] = parts[j] if j < len(parts) else ""
        rows.append(row)

    return rows

# csv_text = "name,age\nAnn,30\nBob,25"
# print(parse_simple_csv(csv_text))
```
