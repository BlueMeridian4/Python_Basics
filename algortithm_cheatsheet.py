"""
ALGORITHM USE-CASE GUIDE — When to Use What

-------------------------------------------------------------
* SORTING ALGORITHMS
-------------------------------------------------------------
1 Bubble Sort
    ➤ Use when: Teaching, debugging, or understanding sorting basics.
    ➤ Avoid for real workloads (O(n²)).
    ➤ Simple logic; good for verifying correctness on small input.

2 Insertion Sort
    ➤ Use when: List is small or nearly sorted.
    ➤ Common in hybrid algorithms like TimSort (used in Python’s built-in sort).
    ➤ Avoid: Large, random datasets.

3 Merge Sort
    ➤ Use when: Stable sorting is needed (preserves order of equal elements).
    ➤ Ideal for linked lists, external sorting (large data on disk), or when stable sorting matters.
    ➤ Avoid: Memory-constrained environments (uses O(n) extra space).

4 Quick Sort
    ➤ Use when: You need fast, general-purpose sorting.
    ➤ Typically fastest in practice due to good cache performance.
    ➤ Avoid: Already sorted or small datasets (can degrade to O(n²)).

-------------------------------------------------------------
* SEARCHING ALGORITHMS
-------------------------------------------------------------
5 Linear Search
    ➤ Use when: Data is unsorted, small, or you only need a single scan.
    ➤ Example: Searching for a name in a short list or string.
    ➤ Avoid: Large datasets (O(n)).

6 Binary Search
    ➤ Use when: Data is sorted and random access is possible (arrays, not linked lists).
    ➤ Example: Searching for a value in a sorted price list or index.
    ➤ Avoid: Unsorted data or when data is changing frequently.

-------------------------------------------------------------
* GRAPH ALGORITHMS
-------------------------------------------------------------
7 Breadth-First Search (BFS)
    ➤ Use when: You need the shortest path in an unweighted graph.
    ➤ Example: Finding minimum hops in a social network, routing in unweighted networks.
    ➤ Avoid: Weighted graphs (use Dijkstra instead).

8 Depth-First Search (DFS)
    ➤ Use when: You need to explore all nodes, detect cycles, or traverse trees.
    ➤ Example: Maze solving, topological sort, connected components.
    ➤ Avoid: When shortest path or minimum distance matters.

9 Dijkstra’s Algorithm
    ➤ Use when: You need shortest paths in weighted graphs (non-negative weights).
    ➤ Example: GPS navigation, network routing (e.g., OSPF).
    ➤ Avoid: Graphs with negative edges (use Bellman-Ford).

-------------------------------------------------------------
* DYNAMIC PROGRAMMING (DP)
-------------------------------------------------------------
10 Fibonacci (Bottom-Up)
    ➤ Use when: You see overlapping subproblems (recursion calls itself with same inputs).
    ➤ Example: Number sequences, path counting, cost minimization.
    ➤ Avoid: Problems without subproblem overlap.

11 Knapsack (0/1)
    ➤ Use when: You must optimize selection under constraints.
    ➤ Example: Choosing items to pack (weight/value tradeoff), resource allocation.
    ➤ Avoid: Continuous optimization (not discrete), very large capacities (O(nW) memory).

-------------------------------------------------------------
* BACKTRACKING & COMBINATORICS
-------------------------------------------------------------
12 Subsets / Combinations / Permutations
    ➤ Use when: You must generate all possible combinations or configurations.
    ➤ Example: Generating power sets, scheduling, constraint satisfaction.
    ➤ Avoid: Large n (>20) — exponential growth (2^n).

-------------------------------------------------------------
* MISCELLANEOUS / HELPER ALGORITHMS
-------------------------------------------------------------
13 Two Sum (Hash Map)
    ➤ Use when: You need O(n) lookup for complementary values.
    ➤ Example: Financial transaction matching, target sum problems.
    ➤ Avoid: Very large datasets where memory is limited.

14 Factorial / Recursion
    ➤ Use when: Working with combinatorics, permutations.
    ➤ Example: Probability, number theory.
    ➤ Avoid: Deep recursion (stack overflow); use iterative for large n.

15 Palindrome Check
    ➤ Use when: Validating string symmetry, reversible data.
    ➤ Example: DNA sequence validation, data integrity checks.

-------------------------------------------------------------
* GENERAL STRATEGY GUIDE
-------------------------------------------------------------
* Greedy Algorithms
    ➤ Choose locally optimal step each time.
    ➤ Use when: Greedy choice leads to global optimum (e.g., Dijkstra, Kruskal).
    ➤ Example: Huffman coding, coin change (if denominations allow).

* Divide and Conquer
    ➤ Split → Solve → Combine.
    ➤ Use when: Problem can be divided into smaller subproblems of same type.
    ➤ Example: Merge Sort, Quick Sort, Binary Search, FFT.

* Dynamic Programming
    ➤ Use when: Overlapping subproblems and optimal substructure.
    ➤ Example: Knapsack, Fibonacci, pathfinding on grids.

* Sliding Window
    ➤ Use when: Looking for subarrays/substrings with constraints (sum, length, distinctness).
    ➤ Example: Longest substring without repeating characters, max sum subarray.

* Two Pointers
    ➤ Use when: Sorted data and pair relationships (e.g., sums, distances).
    ➤ Example: Finding pair with target sum, merging sorted arrays.

* Union-Find (Disjoint Set)
    ➤ Use when: You need to track connected components efficiently.
    ➤ Example: Kruskal’s MST, dynamic connectivity, social networks.

* Recursion vs Iteration
    ➤ Use recursion for clarity, iteration for performance.
    ➤ Avoid deep recursion in Python unless memoized (default recursion limit ≈ 1000).

-------------------------------------------------------------
* GENERAL TIP:
If the problem mentions:
    - "minimum" → think **Dynamic Programming** or **Greedy**
    - "all possible combinations" → think **Backtracking**
    - "sorted list" or "search" → think **Binary Search / Two Pointers**
    - "shortest path" → think **BFS (unweighted)** or **Dijkstra (weighted)**
-------------------------------------------------------------
"""

from typing import List, Set, Any, Union


# =========================================================
# 🧮 BASIC SORTING ALGORITHMS
# =========================================================

def bubble_sort(array: Any) -> Any:
    """O(n^2) — Simple, good for teaching or small lists."""
    array_length = len(array)
    for iteration in range(array_length):
        for index in range(0, array_length - iteration - 1):
            if array[index] > array[index + 1]:
                array[index], array[index + 1] = array[index + 1], array[index]
    return array


def insertion_sort(array: Any) -> Any:
    """O(n^2), stable — Good for nearly sorted lists or small datasets."""
    for iteration in range(1, len(array)):
        key = array[iteration]
        index = iteration - 1
        while index >= 0 and array[index] > key:
            array[index + 1] = array[index]
            index -= 1
        array[index + 1] = key
    return array


def merge_sort(array: Any) -> List[Any]:
    """O(n log n), stable — Excellent general-purpose sort."""
    if len(array) <= 1:
        return array
    mid: int = len(array) // 2
    left: List[Any] = merge_sort(array[:mid])
    right: List[Any] = merge_sort(array[mid:])
    return _merge(left, right)


def _merge(left: List[Any], right: List[Any]) -> List[Any]:
    result: List[Any] = []
    iteration = index = 0
    while iteration < len(left) and index < len(right):
        if left[iteration] < right[index]:
            result.append(left[iteration]); iteration += 1
        else:
            result.append(right[index]); index += 1
    result.extend(left[iteration:]); result.extend(right[index:])
    return result


def quick_sort(array: Any) -> List[Any]:
    """O(n log n) avg, O(n^2) worst — Great for in-place sorting, large data."""
    if len(array) <= 1:
        return array
    pivot = array[len(array) // 2]
    left = [element for element in array if element < pivot]
    mid = [element for element in array if element == pivot]
    right = [element for element in array if element > pivot]
    return quick_sort(left) + mid + quick_sort(right)


# =========================================================
# 🔍 SEARCHING ALGORITHMS
# =========================================================

def linear_search(array: List[Any], target: Any) -> Union[Any, bool]:
    """O(n) — Use when data is unsorted or small."""
    for iteration, value in enumerate(array):
        if value == target:
            return iteration
    return -1


def binary_search(array: List[Any], target: Any) -> Union[Any, bool]:
    """O(log n) — Requires sorted data."""
    low, high = 0, len(array) - 1
    while low <= high:
        mid = (low + high) // 2
        if array[mid] == target:
            return mid
        elif array[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


# =========================================================
# 🔗 LINKED LIST UTILITIES
# =========================================================

class Node:
    def __init__(self, data: Any):
        self.data: Any = data
        self.next: Union[None, Any] = None


class LinkedList:
    """Basic linked list operations — useful for low-level data structure understanding."""
    def __init__(self):
        self.head = None

    def append(self, data: Any):
        if not self.head:
            self.head = Node(data)
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = Node(data)

    def print_list(self):
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")


# =========================================================
# 🧭 GRAPH ALGORITHMS (BFS / DFS / DIJKSTRA)
# =========================================================

from collections import deque, defaultdict
import heapq


def bfs(graph, start):
    """O(V + E) — Level-order traversal, shortest path in unweighted graphs."""
    visited: Set[Any] = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            print(node, end=" ")
            visited.add(node)
            queue.extend(graph[node] - visited)


def dfs(graph, start, visited=None):
    """O(V + E) — Depth-first traversal, good for connectivity, cycle detection."""
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=" ")
    for neighbor in graph[start] - visited:
        dfs(graph, neighbor, visited)


def dijkstra(graph, start):
    """O(E log V) — Shortest path in weighted graph with non-negative edges."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    pq = [(0, start)]
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance > distances[current_node]:
            continue
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
    return distances


# =========================================================
# 🧩 DYNAMIC PROGRAMMING
# =========================================================

def fibonacci_dp(n):
    """O(n) — Bottom-up DP example."""
    if n <= 1:
        return n
    dp = [0, 1]
    for i in range(2, n + 1):
        dp.append(dp[i - 1] + dp[i - 2])
    return dp[n]


def knapsack(weights, values, capacity):
    """O(n*W) — Classic 0/1 Knapsack Problem (DP Table)."""
    n = len(values)
    dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i][w] = max(values[i - 1] + dp[i - 1][w - weights[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][capacity]


# =========================================================
# 🧠 BACKTRACKING (e.g., N-Queens, Subset Generation)
# =========================================================

def subsets(nums):
    """O(2^n) — Generate all subsets (useful for combinatorial search)."""
    result = []

    def backtrack(start, path):
        result.append(path[:])
        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result


# =========================================================
# 🧱 COMMON DATA STRUCTURE UTILITIES
# =========================================================

def factorial(n):
    """Simple recursion, O(n) — Use iterative for performance in large n."""
    return 1 if n == 0 else n * factorial(n - 1)


def is_palindrome(s):
    """Check if string is palindrome."""
    return s == s[::-1]


# =========================================================
# 🧰 UTILITIES & TEMPLATES
# =========================================================

def swap(a, b):
    return b, a


def two_sum(nums, target):
    """O(n) — Hash map pattern; very common interview problem."""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []


# =========================================================
# 🧪 SAMPLE EXECUTION (for testing)
# =========================================================
if __name__ == "__main__":
    print("Merge Sort:", merge_sort([5, 2, 8, 3, 9]))
    print("Binary Search:", binary_search([1, 2, 3, 4, 5], 3))
    print("Subsets:", subsets([1, 2, 3]))
    print("Dijkstra:", dijkstra({
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }, 'A'))
