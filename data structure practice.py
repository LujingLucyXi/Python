# 1) QUESTION: What is the difference between a list and a tuple in Python, and when would you choose one over the other?
# Solution (talking points):
# - list: mutable, dynamic operations (append/pop), slightly larger overhead
# - tuple: immutable, hashable if elements are hashable (can be dict key), good for fixed records
# - choose list for collections that change; tuple for fixed-size records / safety / keys


# 2) QUESTION: How do you remove duplicates from a list while preserving order?
# Solution:
# - use a set to track seen elements and build an output list in one pass
# - discuss time complexity O(n) average

def dedupe_preserve_order(items):
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# 3) QUESTION: Explain how dictionaries work in Python (hash tables). What are common performance characteristics?
# Solution (talking points):
# - dict is a hash table: average O(1) insert/lookup/delete
# - worst-case O(n) but mitigated by good hashing and resizing
# - keys must be hashable (immutable in practice: str, int, tuple of hashables)


# 4) QUESTION: Given a string, how would you find the first non-repeating character?
# Solution:
# - count occurrences with dict, then scan again to find first with count==1

def first_non_repeating_char(s: str):
    counts = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in s:
        if counts[ch] == 1:
            return ch
    return None


# 5) QUESTION: What’s the difference between a shallow copy and a deep copy? Give an example with nested lists.
# Solution (talking points):
# - shallow copy copies outer container but references same inner objects
# - deep copy recursively copies nested objects
# - use copy.copy vs copy.deepcopy


# 6) QUESTION: How do you implement a stack and a queue in Python using built-in data structures?
# Solution:
# - stack: list with append/pop (end)
# - queue: collections.deque with append/popleft


# 7) QUESTION: How do you rotate an N×N matrix 90 degrees clockwise in place?
# Solution:
# - transpose then reverse each row

def rotate_clockwise(mat):
    n = len(mat)
    for r in range(n):
        for c in range(r + 1, n):
            mat[r][c], mat[c][r] = mat[c][r], mat[r][c]
    for r in range(n):
        mat[r].reverse()
    return mat


# 8) QUESTION: How would you detect if two strings are anagrams of each other?
# Solution:
# - compare frequency maps (dict) or sorted strings; dict approach is O(n)

def are_anagrams(a: str, b: str) -> bool:
    if len(a) != len(b):
        return False
    counts = {}
    for ch in a:
        counts[ch] = counts.get(ch, 0) + 1
    for ch in b:
        if ch not in counts:
            return False
        counts[ch] -= 1
        if counts[ch] == 0:
            del counts[ch]
    return len(counts) == 0


# 9) QUESTION: Explain BFS vs DFS. When would you prefer one over the other?
# Solution (talking points):
# - BFS: level-order, finds shortest path in unweighted graph; uses queue
# - DFS: explores deep paths, useful for topological sort/cycle detection; uses stack/recursion


# 10) QUESTION: Implement binary search and explain its time complexity.
# Solution:
# - works on sorted arrays/lists
# - O(log n) comparisons

def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
