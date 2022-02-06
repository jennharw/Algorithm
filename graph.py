# 1. Euler's Theorem  - 모든 정점이 짝수개의 차수를 갖는다면 모든 다리를 한번씩만 건너서 도달하는 것이 성립
# Eulerian Trail, path
# 모든 간선을 한번씩 방문하는 유한 그래프

# 2. HamiltonPath,
# 오일러 경로는 간선을 기준으로
# 해밀턴 경로는 정점을 기준으로
# 히밀턴 경로는 각 정점을 한번씩 방문하는 무향 또는유향 그래프 경로
#
# 3. TSP
# 원래의 출발점으로 돌아오는 경로 Hamiltonian cycle - travelling salesman problem - 각 도시를 방문하고 돌아오는 가장 짧은 경로
# 다이나믹 프로그래밍의 경우 최적화

# 최단경로 다익스트라, 벨만포드, 플로이드와셜


# graph search
#
# dfs - stack, recursion
# bfs - queue, 최단경로 Dijkstra's algorithm is an algorithm for finding the shortest paths between nodes in a graph, which may represent, for example, road networks


# 인접행렬, 인접리스트
# adjacency matrix, adjacency list


#1 dfs (stack, recursion)
# import torch
# from torch_geometric.data import Data
#
# x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float) #feature
# edge_index = torch.tensor([[0, 1, 0, 9, 8, 7, 1, 8, 10, 7, 10, 11, 7, 11, 6, 7, 3, 7, 3, 5, 5, 6, 3, 4, 2, 3],
#                            [1, 0, 9, 0, 7, 8, 8, 1, 7, 10, 11, 10, 11, 7, 7, 6, 7, 3, 5, 3, 6, 5, 4, 3, 3, 2]], dtype=torch.long)
#
# data = Data(edge_index=edge_index)
#
# ret = []
# def dfs(node = ""): #모든 경우
#     if node == "":
#         node = 0
#     visited[node] = True
#     ret.append(node)
#     neighbors = edge_index[node][1]
#     for neighbor in neighbors:
#         if visited[neighbor] == False:
#             dfs(neighbor)
#     return ret, visited
#
# def dfs(G, node): # stack
#     visited= []
#     ret = []
#
#     ret.append(node)
#     stack = []
#     stack.append(node)
#     while stack:
#         p = stack.pop()
#         neighbors =  p 의 neiughbors
#         ret.append(p)
#
#
#         for neighbor in neighbors:
#             #validation .ㅑif ~
#                 stack.append(node)
#
#     return ret
#
#
#
# def bfs(graph, start, to): #최단경로
# def bfs(s, e):  # s = start node, e = end node
#     # BFS
#     brt, prev = solve(s)
#
#     return reconstructPath(s, e, prev)
#
#
# visited = [False] * n
# prev = [None] * n
#
#
# def solve(s):
#     q = collections.deque()
#     q.append(s)
#     visited[s] = True
#     brt = []
#     brt.append(s)
#
#     while q:
#         node = q.popleft()
#         neighbors = g[1][g[0] == node]
#         for next in neighbors:
#             if visited[next] == False:
#                 brt.append(next)
#                 q.append(next)
#                 visited[next] = True
#                 prev[next] = node
#     return brt, prev
#
#
# print(prev)
# print(solve(0))
#
#
# def reconstructPath(s, e, prev):
#     path = []
#     # Reconstruct path going backwards from e
#     at = e
#     while at != None:
#         path.append(at)
#         at = prev[at]
#     path.reverse()
#     if path[0] == s:
#         return path
#     return []
#
#
# print(bfs(0, 10))
import bisect
import collections


import heapq
import itertools

#12장 그래프
import sys


class GraphProblem:
    # 32 섬의 개수
    def numIslands(self, grid):
        #bfs 재귀 를 통해 component 구하기
        left  = len(grid)
        right = len(grid[0])

        visited = [[False for _ in range(right)] for _ in range(left)]

        #ret = [[False for _ in range(right)] for _ in range(left)]

        def dfs(l , r):

            #ret[l][r] = True
            visited[l][r] = True
            i = [1, 0, -1, 0]
            j = [0, 1, 0, -1]

            for k in range(len(i)):
                if 0 <= l + i[k] < left and 0 <=  r + j[k] < right:
                    if grid[l + i[k]][r + j[k]] == "1" and visited[l + i[k]][r + j[k]] == False:
                        dfs(l + i[k], r + j[k])
            return visited

        count = 0

        #print(dfs(l = 0, r = 0))
        for p in range(4):
            for q in range(5):
                if grid[p][q] == "1" and visited[p][q] == False:
                    dfs(p, q)
                    count+=1
        # print(visited)
        # print(count)
        # print(dfs(l=2, r=2))
        # print(dfs(l=3, r=3))
        return count
        #print(neighbors)

    #33) 순열, 조합 -> 수를 세는 것은 쉬우나, 조합을 생성할 때 dfs
    def letterCombinationsOfPhoneNumber(self, digits:str):
        #3 * 3
        def dfs(index, path):
            if len(path) == len(digits):
                result.append(path)
                return
            for i in range(index, len(digits)):
                for j in dict[digits[i]]:
                    dfs(i+1, path+j)

        if not digits:
            return []
        dict = {
            "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
            "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
        }
        result = []
        dfs(0, "")
        return result

    # 34)
    def permutation(self, l):
        # node = l[0]
        # print(node)
        # l.pop(0)
        print(l[:0] + l[1:] )
        print(l[:1] + l[2:] )
        print(l[:2] + l[3:] )
        # neighbors = l
        # print(neighbors)
        # result = []
        # path = []
        # def dfs(index, path):
        #     if path == len(l):
        #         print(path)
        #         result.append(path)
        #         print("result", result)
        #         return
        #     path.append(index)
        #     node = l[index]
        #     l.remove(node)
        #     for neighbor in neighbors:
        #         path.append(neighbor)
        #         dfs(neighbor, path)
        # print(dfs(0, path))
        # #
        # #


    def permutationItertools(self, n):

        return list(itertools.permutations(n))

    # 35)
    def comintationItertools(self, n, k):
        return list(itertools.combinations(range(1, n+1), k))

    def combination(self, n, k):
        result = []
        def dfs(nums, pth):
            if len(pth) == k:
                result.append(pth[:])
                return


            for i in range(nums, n+1):
                pth.append(i)
                dfs(i+1, pth)
                pth.pop()

        dfs(1, [])

        return result

    # 36)
    def combinationSum(self, candidates, target):
        result = []

        def dfs(n, pth):
            if sum(pth) > target:
                return
            if sum(pth) == target:
                result.append(pth[:])
                return

            for i in range(n, len(candidates)):
                pth.append(candidates[i])
                dfs(i, pth)
                pth.pop()

        dfs(0, [])

        return result
    #37)
    def subsets(self, nums):
        result = []

        def dfs(n, pth):
            if len(pth) > len(nums):
                return
            result.append(pth[:])
            for i in range(n, len(nums)):
                pth.append(nums[i])
                dfs(i+1, pth)
                pth.pop()

        dfs(0, [])

        return result

    # 38)
    def itinerary(self, schedules):
        grph = collections.defaultdict(list)
        for d, t in sorted(schedules):
            grph[d].append(t)
        print(grph)

        route = []

        def dfs(b):
            while grph[b]:
                dfs(grph[b].pop(0))
            route.append(b)

        dfs('JFK')

        return route[::-1]

    def itineraryStck(self, schedules):
        grph = collections.defaultdict(list)
        for d, t in sorted(schedules, reverse=True):
            grph[d].append(t)
        print(grph)

        route = []

        def dfs(b):
            while grph[b]:
                dfs(grph[b].pop())
            route.append(b)

        dfs('JFK')

        return route[::-1]

    def itineraryItertionStack(self, schedules):
        grph = collections.defaultdict(list)
        for d, t in sorted(schedules, reverse=True):
            grph[d].append(t)
        print(grph)

        route, stack = [], ['JFK']
        while stack:
            while grph[stack[-1]]:
                stack.append(grph[stack[-1]].pop())
            route.append(stack.pop())
        print(route)
        return route[::-1]

    #39)
    def courseSchedule(self, courses):
        #순환인가 cyclic -> false
        graph = collections.defaultdict(list)

        for x, y in courses:
            graph[x].append(y)

        traced = set()
        def dfs(i):
            if i in traced:
                return False

            traced.add(i)
            for y in graph[i]:
                if not dfs(y):
                    return False
            traced.remove(i)
            return True

        print(graph)
        for x in list(graph):#1, 0 0,1
            print(x)
            if not dfs(x):
                return False
        return True

    #가지치기
    def canFinish(self, courses):
        graph = collections.defaultdict(list)

        for x, y in courses:
            graph[x].append(y)

        traced = set()
        visited = set()

        def dfs(i):
            if i in traced:
                return False
            if i in visited:
                return True

            traced.add(i)
            for y in graph[i]:
                if not dfs(y):
                    return False
            traced.remove(i)
            visited.add(i)
            return True


        for x in list(graph):
            if not dfs(x):
                return False
        return True



    #bfs 최단경로 40)
    def networkDelayTime(self, times, N, K):
        #bfs queue 에 넣을 때마다 1?
        #거리 => 다익스트라 - 우선순위 queue heapq

        #(이웃, 거리) graph

        graph = collections.defaultdict(list)
        for u, v, t in times:
            graph[u].append((v,t))

        Q = [(0, K)]

        dist = collections.defaultdict(int)

        while Q:
            #pop?
            time, node = heapq.heappop(Q) #제일 작은 거 0,2
            if node not in dist:
                dist[node] = time
                for v, t in graph[node]: #(1, 1), (3, 1)
                    dis = time + t #

                    heapq.heappush(Q, (dis, v)) #heap
        if len(dist) == N: #노드들...
            return max(dist.values())
        return -1

    #)41
    def cheapestFlights(self, n, edges, src, dst, K):
        graph = collections.defaultdict()

        for u, v, e in edges:
            graph[u].append((v, e))

        Q = [(0 ,src, K)]

        k = 0
        while Q:
            dis, u, k = heapq.heappop(Q)
            if u == dst:
                return dis
            if k <= K:
                k+= 1
                for v, e in graph[u]:
                    dis = dis + e
                    heapq.heappush(Q, (dis, v, k))
        return -1



#14장 트리
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class TreeProblem:
    #42
    def maxDepth(self, root):
        if root is None:
            return 0
        queue = collections.deque([root])
        depth = 0
        while queue:
            depth+=1
            for _ in range(len(queue)):
                cur_root = queue.popleft()
                if cur_root.left:
                    queue.append(cur_root.left)
                if cur_root.right:
                    queue.append(cur_root.right)
        return depth

    #43
    longest = 0
    def diameterBT(self, root):
        def dfs(node):
            if not node:
                return -1
            left = dfs(node.left)
            right = dfs(node.right)
            self.longest = max(self.longest, left+ right+2)
            return max(left, right) + 1
        dfs(root)
        return self.longest

    #44
    result = 0
    def longestPath(self, root):
        #dfs ,  backtracking
        def dfs(node):
            if node is None:
                return 0
            left = dfs(node.left)
            right = dfs(node.right)

            if node.left and node.left.val == node.val:
                left += 1
            else:
                left = 0
            if node.right and node.right.val == node.val:
                right += 1
            else:
                right = 0
            self.result = max(self.result, left+right)
            return max(left, right)
        dfs(root)
        return self.result

    #45) invert

    #python 방식
    def invert(self, root):
        if root:
            root.left, root.right = self.invert(root.right), self.invert(root.left)
            return root
        return None

    def invertTreeBFS(self, root):
        #bfs
        queue = collections.deque([root])
        while queue:
            node = queue.popleft()
            if node:
                node.left, node.right = node.right, node.left
                queue.append(node.left)
                queue.append(node.right)
        return root

    def invertTreeDFS(self, root):
        #dfs
        queue = collections.defaultdict([root])
        while queue:
            node = queue.pop() #가장 뒤에
            if node:
                node.left, node.right = node.right, node.left
                queue.append(node.left)
                queue.append(node.right)
        return root

    #46
    def margeTree(self, t1, t2):
        if t1 and t2:
            node = TreeNode(t1.val, t2.val)
            node.left = self.mergeTree(t1.left, t2.left)
            node.right = self.mergeTree(t1.right, t2.right)
            return node
        else:
            return t1 or t2

    #47
    def serialize(self, root):
        #bfs
        queue = collections.queue([root])
        result = ['#']
        while queue:
            node = queue.popleft()
            if node:
                queue.append(node.left)
                queue.append(node.right)
                result.append(str(node.val))
            else:
                result.append('#')
        return ' '.join(result)

    # def deSeriazlier(self, data : str):
    #     if data == "# #":
    #         return None
    #     nodes = data.split()
    #     root = TreeNode(int(nodes[1]))
    #
    #     queue = collections.deque([root])
    #     index = 2
    #
    #     while queue:
    #         node = queue.popleft()
    #         if nodes[index] is not '#':
    #             node.left = TreeNode(int(nodes[index]))
    #             queue.append(node.left)
    #         index +=1
    #         if nodes[index] is not '#':
    #             node.right = TreeNode(int(nodes[index]))
    #             queue.append(node.right)
    #         index+=1
    #     return root

    #48
    def isBalanced(self, root):
        def check(root):
            if not root:
                return 0
            left = check(root.left)
            right = check(root.right)

            if left == -1 or right == -1 or abs(left-right)> 1:
                return -1
            return max(left, right) +1
        return check(root) != -1

    #49
    def minimumHeightTree(self, n, edges):
        #dfs
        if n<=1 :
            return [0]

        graph = collections.defaultdict(list)
        for i, j in edges:
            graph[i].append(j)
            graph[j].append(i)

        leaves = []
        for i in range(n+1):
            if len(graph[i])==1:
                leaves.append(i)

        while n>2:
            n -= len(leaves)
            new_leaves = []
            for leaf in leaves:
                neighbor = graph[leaf].pop()
                graph[neighbor].remove(leaf)

                if len(graph[neighbor] == 1):
                    new_leaves.append(neighbor)
            leaves = new_leaves
        return leaves

# class TreeNode:
#     nodeLHS = None
#     nodeRHS = None
#     nodeParent = None
#     value = None
#
#     def __init__(self, value, nodeParent):
#         self.value = value
#         self.nodeParent = nodeParent
#
#     def getLHS(self):
#         return self.nodeLHS
#
#     def getRHS(self):
#         return self.nodeRHS
#
#     def getValue(self):
#         return self.value
#
#     def getParent(self):
#         return self.nodeParent
#
#     def setLHS(self, LHS):
#         self.nodeLHS = LHS
#
#     def setRHS(self, RHS):
#         self.nodeRHS = RHS
#
#     def setValue(self, value):
#         self.value = value
#
#     def setParent(self, nodeParent):
#         self.nodeParent = nodeParent
#
# # Definition for a Node.
# class Node:
#     def __init__(self, val, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
#BST. Binary Search (Tree)
class BSTP:
    #50)
    def convertSortedArraytoBST(self, nums):
        if not nums:
            return None
        mid = len(nums)//2

        node = TreeNode(nums[mid])
        node.left = self.convertSortedArraytoBST(nums[:mid])
        node.right = self.convertSortedArraytoBST(nums[mid+1:])
        return node

    #51
    def bst_sum(self, root):
        #bst 옆이 나보다 큰 !

        if root:
            self.bst_sum(root.right)
            self.val += root.val
            root.val = self.val
            self.bst_sum(root.left)
        return root

    # #52
    # def rangeSumBst(self, root, L, R):
    #     if not root:
    #         return 0
    #
    #     return root.val if L <= root.val <= R else 0 + self.(root.left) + rangeSumBst(root.right)
    #
    # def rangeSumBstDfs(self, root, L, R):
    # def rangeSumBstStack(self, root, L, R):
    #
    # #53
    # prev = 0
    # result = 0
    # def minimumDistancebst(self, root):
    #
    #     if root.left:
    #         self.convertSortedArraytoBST(root.left)
    #
    #     self.result = min(self.result, root.val - self.prev)
    #     self.prev= root.val
    #
    #     if root.right:
    #         self.minimumDistancebst(root.right)
    #
    #     return self.result
    #
    # #54 전위, 중위 순회회
    # def buidTree(self, preorder, inorder):
    #     if inorder:
    #         index = inorder.index(preorder.pop(0))
    #
    #         node = TreeNode(inorder[index])
    #         node.left = buidTree
    #
    #         return node
#18장 BinarySearch
class BinarySearch:
    #2/4
    #65
    def binarySearch(self, nums, target):
        l, r = 0, len(nums)
        while l <= r:
            mid = (l + r) //2
            if nums[mid] == target:
                return mid
            if nums[mid] < target:
                l = mid + 1
            if nums[mid] > target:
                r = mid - 1
        return -1
    def binarySearchRecursion(self, nums, target):
        def binary_search(l, r):
            if l <= r:
                mid = (l + r) //2
                if nums[mid] == target:
                    return mid
                if nums[mid] < target:
                    return binary_search(mid + 1, r)
                if nums[mid] > target:
                    return binary_search(l, mid+1)
            else:
                return -1
        return binary_search(0, len(nums)-1)
    def binarySearchBisect(self, nums, target):
        index = bisect.bisect_left(nums, target)
        if index < len(nums) and nums[index] == target:
            return index
        else:
            return -1
    def binarySearchIndex(self, nums, target):
        try:
            return nums.index(target)
        except:
            return -1
    #66
    def searchInRotatedSortedArray(self, nums, target):
        l, r = 0, len(nums)-1
        while l <= r:
            mid = l + (r-l) //2 #자료형을 초과하지 않는 중앙 위치 계산
            if nums[mid] == target:
                return mid

            if nums[l] <= nums[mid]: # left sorted
                if nums[l] > target or nums[mid] < target :
                    l = mid + 1
                else: #nums[l] < target , nums[mid] > target
                    r = mid - 1
            else: # nums[l] < nums[mid]
                if nums[r] < target or nums[mid] > target :
                    r = mid - 1
                else: #nums[r] > target
                    l = mid + 1

        return -1


    #67
    def intersectionOfTwoArrays(self, nums1, nums2):
        result = set()
        nums2.sort()
        for n1 in nums1:
            i2 = bisect.bisect_left(nums2, n1)
            if len(nums2) > 0 and len(nums2) > i2 and n1 == nums2[i2]:
                result.add(n1)
        return result

    def intersectionOfTwoArraysTwoPointer(self, nums1, nums2):
        result = set()
        nums2.sort()
        nums1.sort()
        i = j = 0
        while i < len(nums1) and j < len(nums2):
            if nums1[i] == nums2[j]:
                result.add(nums1[i])
                i += 1
                j += 1
            elif nums1[i] < nums2[j]:
                i += 1
            else:
                j += 1

        return result


    # 68
    def twoSum(self, numbers, target):
        #Two pointer - 배열이 정렬
        l ,  r = 0, len(numbers)  -1
        while l < r:
            if numbers[l] + numbers[r] < target:
                l += 1
            elif numbers[l] + numbers[r] > target:
                r -= 1
            else:
                return l + 1, r + 1

    def twoSumBinarySearch(self, numbers, target):
        for k, v in enumerate(numbers):
            l, r = k, len(numbers) - 1
            expected = target - v
            while l <= r:
                mid = l + (r - l) // 2
                if numbers[mid] < expected:
                    l = mid + 1
                elif numbers[mid] > expected :
                    r = mid - 1
                else:
                    return k + 1, mid + 1
    # 또는 bisect

    def twoSumBisect(self, nums, target):
        for k, v in enumerate(nums):
            expected = target - v
            i = bisect.bisect_left(nums, expected, k+1) #l0 , slicing 필요 없음
            if len(nums) > i and expected == nums[i]:
                return k+1, i+1


    # 69
    def search2D(self, matrix, target):
        return any(target in row for row in matrix)
    def search2DBS(self, matrix, target):
        if not matrix:
            return False
        row = 0
        col = len(matrix) - 1
        while row <= len(matrix) - 1 and col >= 0:
            if target == matrix[row][col]:
                return True
            elif matrix[row][col] < target:
                row += 1 #아래로 이동
            else: #왼쪽으로 이동
                col -= 1
        return False

    # 2/5
    # 고정점
    def fixedPoint(self, lst):  # 최대 거리 출력
        # 이진탐색
        def binary_search(array, start, end):
            if start > end:
                return -1
            mid = (start + end) // 2
            if array[mid] == mid:
                return mid
            elif array[mid] < mid:
                return binary_search(array, mid + 1, end)
            else:
                return binary_search(array, start, mid - 1)

        return binary_search(lst, 0, len(lst)-1)

    # 공유기 https://www.acmicpc.net/problem/2110
    def wifi(self, N, C, lst):
        # import sys
        #
        # N, C = map(int, sys.stdin.readline().split())
        # lst = [int(sys.stdin.readline()) for _ in range(N)]

        lst.sort()

        # 공유기 사이 거리 최솟값
        start = 1
        # 공유기 사이 거리 최댓값
        end = lst[-1] - lst[0]

        while start <= end:
            count = 1
            mid = (start + end) // 2
            value = lst[0]  # 공유기가 설치된 집의 위치
            for i in range(1, N):
                if value + mid <= lst[i]:  # 공유기 설치
                    count += 1
                    value = lst[i]
            if count >= C:  # mid 값에 따라 설치된 공유기의 개수가 c 보다 많거나 같으면
                start = mid + 1  # 거리를 늘린다.
                result= mid
            else:
                end = mid - 1  # c 보다 작으면 거리를  줄인다.

        print(result)
        return result


    # https://www.acmicpc.net/problem/2512 예산
    def budget(self, lst, M):
        return 0
    #     import sys
    #     input = sys.stdin.readline
    #
    #     N = int(input())
    #     cities = list(map(int, input().split()))
    #     cities.sort()
    #     M = int(input())  # 예산
    #     start, end = 1, cities[-1]  # 시작 점, 끝 점
    #
    #     # 이분 탐색
    #     while start <= end:
    #         mid = (start + end) // 2
    #         bdgt = M  # 총 지출 양
    #         for i in cities:
    #             bdgt -= mid if i > mid else i
    #         if bdgt >= 0:  # 지출 양이 예산 보다 작으면
    #             start = mid + 1
    #         else:  # 지출 양이 예산 보다 크면
    #             end = mid - 1
    #     print(end)
    #     N = int(input())
    #     cities = list(map(int, input().split()))
    #     cities.sort()
    #     M = int(input())  # 예산
    #     start, end = 0, cities[-1]  # 시작 점, 끝 점
    #
    #     # 이분 탐색
    #     while start <= end:
    #         mid = (start + end) // 2
    #         bdgt = M  # 총 지출 양
    #         for i in cities:
    #             bdgt -= mid if i > mid else i
    #         if bdgt >= 0:  # 지출 양이 예산 보다 작으면
    #             start = mid + 1
    #         else:  # 지출 양이 예산 보다 크면
    #             end = mid - 1
    #     print(end)
    #
    #
    #
    #
    #                 # N = int(input())
    #         lst = list(map(int, input().split()))
    #         M = int(input())
    #         #이진탐색
    #         lst.sort()
    #         start = 1
    #         end = lst[-1]
    #
    #         while start <= end:
    #             bdgt = M
    #             mid = end - start // 2 #130
    #             for i in lst:
    #                 bdgt -= mid if i > mid else i
    #             if bdgt >= 0 :
    #                 start = mid + 1
    #
    #                 # mid += (v // mid_i)
    #                 # return mid
    #             else:
    #                 end = mid - 1
    #
    #                 # mid += v //mid_i
    #                 # if mid > end:
    #                 #     return end
    #         print(end)
    #         return end
    #
    # # https://www.acmicpc.net/problem/2805 나무 자르기
    # def cutTree(self, lst, M):
    #     # N, M = map(int, input().split())
    #     # lst = list(map(int, input().split()))
    #     lst.sort()
    #     start, end = 1, lst[-1]  # 이분탐색 검색 범위 설정
    #
    #     while start <= end:  # 적절한 벌목 높이를 찾는 알고리즘
    #         mid = (start + end) // 2
    #
    #         result = 0  # 벌목된 나무 총합
    #         for i in lst:
    #             result += i - mid if i > mid else 0
    #
    #         # 벌목 높이를 이분탐색
    #         if result >= M:
    #             start = mid + 1
    #         else:
    #             end = mid - 1
    #     print(end)
    #     return end



    # https://www.acmicpc.net/problem/1654 랜선자르기
    def cutLan(self, lst, M):
        #K개의 랜선 N개의 같은 길이의 랜선 -> 최대 길이
        N, M = map(int, input().split())
        lst = list(map(int, input().split()))
        lst.sort()
        end = sum(lst) // M  #max(max(lst)
        start = lst[-1] // (M - len(lst) - 1) #1????

        while start <= end:
            mid = (start + end) // 2
            count = 0
            for i in lst:
                count += i // mid
            if count >= M:
                start = mid + 1
            else:
                end = mid - 1
        print(end)
        return end


    def networkDelayTime(self, times, N, K):
        graph = collections.defaultdict(list)
        for (u, v, w) in times:
            graph[u].append((v,w))
        Q = [(0, K)]

        dist = collections.defaultdict(int)

        while Q:
            time, node = heapq.heappop(Q)
            if node not in dist:
                dist[node] = time
                for v, w in graph[node]:
                    alt = time + w
                    heapq.heappush(Q, (alt, v))
        if len(dist) == N:
            return max(dist.values())
        return -1

    def cheapestFlights(self, n, edges, src, dst, K):
        graph = collections.defaultdict(list)
        for (u, v, w) in edges:
            graph[u].append((v, w))

        Q = [(0, src, K)]

        while Q:
            distance, node, k = heapq.heappop(Q)
            if node == dst:
                return distance
            if k >= 0 :

                for v, w in graph[node]:
                    alt = distance + w
                    heapq.heappush(Q, (alt, v, k-1))
        return -1

    def shortestPath(self, src, edges):
        graph = collections.defaultdict(list)
        for (u, v, w) in edges:
            graph[u].append((v,w))
        Q = [(0, src)]

        #visited
        visited = collections.defaultdict(int)

        while Q:
            dist, node = heapq.heappop(Q)

            if node not in visited:
                visited[node] = dist
                for (v, w) in graph[u]:
                    alt = dist + w
                    heapq.heappush(Q, (alt+w, node))
        print(visited)

    def checkMars(self, edges):
        i, j = 0, 0
        i_lst = [ 0, 1, 0]
        j_lst = [-1, 0, 1]

        visited = [[0 for _ in range(len(edges[0]))] for _ in range(len(edges))]

        Q = [(edges[0][0], (0, 0))]
        #heapq.heappush(Q)
        while Q:
            #print(heapq.heappop(Q))
            energy, (w_i, w_j) = heapq.heappop(Q)

            #print("energy", energy)
            #print(w_j, w_i)

            if (w_i, w_j) not in visited:
                visited[w_j][w_i] = energy

                #이동
                for l in range(3):
                    p =  w_i + i_lst[l]
                    q = w_j + j_lst[l]

                    if p == len(edges) - 1 and q == len(edges) - 1:
                        return energy + edges[q][p]

                    if ( 0 <= p <= len(edges)-1 ) and ( 0 <= q <= len(edges)-1 ):
        #                        print(q, p)

#                        print(energy + edges[q][p])

                        heapq.heappush(Q , (energy + edges[q][p], (p, q)))

        print(visited)
        return -1


    def hideSeek(self, edges):
        grph = collections.defaultdict(list)
        for v, u in edges:
            grph[v].append(u)
            grph[u].append(v) #양방향

        Q = [(0, 1)]

        #dist = collections.defaultdict(int)
        visited = collections.defaultdict(int)
        while Q:
            dist, src = heapq.heappop(Q)

            if src not in visited:
                visited[src] = dist

                for vertex in grph[src]:
                    alt = dist + 1
                    heapq.heappush(Q, (alt, vertex))

        print(visited)
        print(max(visited.values()))
        print(list(visited.values()).index(2))

        print(list(visited.keys())[list(visited.values()).index(2)])  # Prints george


    def floyd(self, N, edges):
        INF = int(1e9)
        visited = [[INF for _ in range(N)] for _ in range(N)]
        for s, v, u in edges: #가장짧은 경우!

            visited[s-1][s-1] = 0
            if u < visited[s-1][v-1]:
                visited[s-1][v-1] = u
        print(visited)

        #플로이드워셜
        for k in range(N):
            for a in range(N):
                for b in range(N):
                    visited[a][b] = min(visited[a][b] , visited[a][k] + visited[k][b]) #k 거쳐서 가는 경우
        print(visited)

    #def futureCities(self, ):




class DynamicProgramming:
    def fibonacciRecursion(self, N):
        #recursion
        def fibonacci(n):
            if n == 0:
                return 1
            if n ==1:
                return 1
            return fibonacci(n-1) + fibonacci(n-2)
        return fibonacci(N)

    def fibonacciMemoization(self, N):
        dp = collections.defaultdict(int)
        def fib(N):
            print('f(' + str(N) + ')', end = ' ')
            if N <= 1:
                return N
            if dp[N]:
                return dp[N]
            dp[N] = fib(N-1) + fib(N-2)
            return dp[N]
        return fib(N)

    def bibonfibonacciTabulation(self, N):
        dp = collections.defaultdict(int)
        def fib(N):
            dp[1] = 1
            dp[2] = 2
            for i in range(2, N+1):
                dp[i] = dp[i-1] + dp[i-2]
            return dp[N]
        return fib(N)

    def fib(self, N):
        x, y = 0, 1
        for i in range(0, N):
            x, y = y, x+y
        return x

    def maximumSubarray(self, nums):
        #연속
        #Memoization
        sums = [nums[0]]
        for i in range(1, len(nums)):
            sums.append(nums[i] + (sums[i-1] if sums[i-1] > 0 else 0))

        return max(sums)

    def Kadane(self, nums):
        best_sum = -sys.maxsize
        current_sum = 0
        for num in nums:
            current_sum = max(num, current_sum + num)
            best_sum = max(best_sum, current_sum)
        return best_sum


    def climbRecursion(self, n):
        if n == 1:
            return 1
        if n == 2:
            return 2
        return self.climbRecursion(n-1) + self.climbRecursion(n-2)

    def climb(self, N):
        dp = collections.defaultdict(int)
        def climbStairs(n):
            if N <= 2:
                return n
            if dp[n]:
                return dp[n]
            dp[n] = climbStairs(n-1) + climbStairs(n-2)
            return dp[n]
        return climbStairs(3)

    def houseRecursion(self, nums):
        def _rob(i):
            if i < 0:
                return 0
            return max(_rob(i-1), _rob(i-2) + nums[i])
        return _rob(len(nums)-1)

    def house(self, nums):
        dp = collections.OrderedDict()
        if not nums:
            return 0
        if len(nums)<=2:
            return max(nums)
        dp[0], dp[1] = nums[0], max(nums[:2])
        for i in range(2, len(nums)):
            dp[i] = max(dp[i-1], dp[i-2]+nums[i])
        return dp.popitem()[1]

    # 2 / 7
    # 최단경로, 다익스트라, 화성탐사, 숨바꼭질
    # 네트워크 딜레이, 가장 저렴한 항공권
    # 2 / 8
    # 플로이드워셜
    # - [boj][https: // www.acmicpc.net / problem / 1753](https: // www.acmicpc.net / problem / 1753)
    # - [boj][https: // www.acmicpc.net / problem / 1956](https: // www.acmicpc.net / problem / 1956)
    # - [boj][https: // www.acmicpc.net / problem / 4485](https: // www.acmicpc.net / problem / 4485)
    # 2 / 9
    # dp, 피보나치, 정수삼각형, 퇴사
    # 최대서브배열, 계단오르기, 집도둑
    def triangle(self, tri ):

        dp = [[0 for _ in tri[i]] for i in len(tri)]
        def tris(n):
            if n== 0:
                dp[0][0] = tri[0][0]

            #i
            for i in range(1, len(tri)):
                for j in range(i+1):
                    if j != 0 :
                        dp[i][j] = max(dp[i-1][j], dp[i-1][j-1]) + tri[i][i]
                    if j == 0:
                        dp[i][j] = dp[i - 1][j] + tri[i][i]
                    if j == i:
                        dp[i][j] = dp[i - 1][j - 1] + tri[i][i]

            return max(dp[n])
        return tris(5)

    def compny(self, schedules, N):
        dp = collections.defaultdict(int)

        dp        dp[1] = schedules[1]












# 15 장 힙
class BinaryHeap2:
    def __init__(self):
        self.items  = [None]
    def __len__(self):
        return len(self.items) -1
    def percolate_up(self):
        i = len(self)
        parent = i // 2
        while parent > 0:
            if self.items[i] < self.items[parent]:
                self.items[parent], self.items[i] = self.items[i], self.items[parent]
                i = parent
                parent = i//2
    def percolate_down(self, idx):
        left = idx * 2
        right = idx * 2
        smallest = idx
        if left <= len(self) and self.items[left] < self.items[smallest]:
            smallest = left
        if right <= len(self) and self.items[right] < self.items[smallest]:
             smallest = right
        if smallest != idx:
            self.items[idx] , self.items[smallest] = self.items[smallest], self.items[idx]
        self.percolate_down(smallest)
    def extract(self):
        extracted = self.items[1]
        self.items[1] = self.items[len(self)]
        self.items.pop()
        self.percolate_down(1)
        return extracted
class BinaryHeap:
    def __init__(self):
        self.arrPriority = [0] * 99
        self.arrValue = [0] * 99
        self.size = 0

    # insert - Percolation
    def enqueueWithPrioirty(self, value, prioirty):
        self.arrPriority[self.size] = prioirty
        self.arrValue[self.size] = value
        self.size = self.size + 1
        self.percolateUp(self.size - 1)

    def percolateUp(self, idxPercolate):
        if idxPercolate == 0:
            return
        parent = int((idxPercolate - 1) / 2)
        if self.arrPriority[parent] < self.arrPriority[idxPercolate]:
            self.arrPriority[parent], self.arrPriority[idxPercolate] = self.arrPriority[idxPercolate], self.arrPriority[
                parent]
            self.arrValue[parent], self.arrValue[idxPercolate] = self.arrValue[idxPercolate], self.arrValue[parent]
            self.percolateUp(parent)

    def dequeueWithPrioirty(self):
        if self.size == 0:
            return
        retPrioirty = self.arrPriority[0]
        retValue = self.arrValue[0]
        self.arrPriority[0] = self.arrPriority[self.size - 1]
        self.arrValue[0] = self.arrValue[self.size - 1]
        self.size -= 1
        self.percolateDown(0)
        return retValue

    def percolateDown(self, idxPercolate):
        if 2 * idxPercolate + 1 > self.size:
            return
        else:
            leftChild = idxPercolate * 2 + 1
            leftPrioirty = self.arrPriority[leftChild]
        if 2 * idxPercolate + 2 > self.size:
            return
        else:
            rightChild = idxPercolate * 2 + 2
            rightPriority = self.arrPriority[rightChild]

        if leftPrioirty > rightPriority:
            biggerChild = leftChild
        else:
            biggerChild = rightChild

        if self.arrPriority[idxPercolate] < self.arrPriority[biggerChild]:
            self.arrPriority[idxPercolate], self.arrPriority[biggerChild] = self.arrPriority[biggerChild], \
                                                                            self.arrPriority[idxPercolate]
            self.arrValue[idxPercolate], self.arrValue[biggerChild] = self.arrValue[biggerChild], self.arrValue[
                idxPercolate]
            self.percolateDown(biggerChild)

class HeapProblem:
    #55)
    def KLargest(self,l, k):
        q = []
        for n in l:
            heapq.heappush(q, -n) #가장 작은 수
        for _ in range(1, k):
            heapq.heappop(q)
        return -heapq.heappop(q)

    def KL(self, l, k):
        heapq.heapify(l)
        for _ in range(len(l)-k):
            heapq.heappop(l)
        return heapq.heappop(l)

    def kl22(self, l, k):
        return heapq.nlargest(k, l)[-1]

    def kl3(self, l, k):
        return sorted(l, reverse=True)[k-1]


# 16 장 트라이
class TrieNode:
    def __init__(self):
        self.word = False
        self.children = {}

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, s):
        node = self.root

        for char in s:
            if char not in node.children:

                node.children[char] = TrieNode()
            node = node.children[char]
        node.word = True


    def search(self, s):
        node = self.root
        for char in s:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.word


    def startWith(self, s):
        node = self.root
        for char in s:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


class TrieNode2:
    def __init__(self):
        self.word_id = -1 #index구현해야
        self.children = collections.defaultdict(TrieNode2)
        self.hasPalindrome_ids = []

class Trie2:
    def __init__(self):
        self.root = TrieNode2()

    @staticmethod
    def is_palindrome(word):
        return word[::] == word[::-1]

    def insert(self, index, s):
        node = self.root

        for i, char in enumerate(reversed(s)): #뒤집어서
            if self.is_palindrome(s[0:len(s)-i]):
                node.hasPalindrome_ids.append(index)
            node = node.children[char] #defaultdict
        node.word_id = index

    def search(self,index, word): #순서대로 -> index, word_id
        result = []
        node = self.root
        while word:
            if node.word_id >= 0:
                if self.is_palindrome(word):
                    result.append([index, node.word_id])
            if not word[0] in node.children:
                return result
            node = node.children[word[0]]
            word = word[1:]

        if node.word_id >= 0 and node.word_id != index:
            result.append([index, node.word_id])

        for palinderome_word_id in node.hasPalindrome_ids:
            result.append([index, palinderome_word_id])

        return result

# 트라이 Palindrome
class Solution:
    #Brute Force - Timeout
    def palindromePairs(self, words):
        def is_palindrome(word):
            return word == word[::-1]
        #n**2
        output = []
        for i, word1 in enumerate(words):
            for j, word2 in enumerate(words):
                if i == j:
                    continue
                if is_palindrome(word1+word2):
                    output.append([i, j])

        return output


    def palindromePairsTrie(self, words):
        trie = Trie2()

        for i, word in enumerate(words):
            trie.insert(i, word)

        result = []

        for i, word in enumerate(words):
        #O(n) 3가지
            result.extend(trie.search(i, word))

        return result


## BackJoon + 이코테
# graph
# https://www.acmicpc.net/problem/1197
# https://www.acmicpc.net/problem/1260
# https://www.acmicpc.net/problem/2178
# - [https://www.acmicpc.net/problem/2667](https://www.acmicpc.net/problem/2667)
# - [https://www.acmicpc.net/problem/2606](https://www.acmicpc.net/problem/2606)
# backtracking
# - [https://www.acmicpc.net/problem/9663](https://www.acmicpc.net/problem/9663)
# - [https://www.acmicpc.net/problem/9095](https://www.acmicpc.net/problem/9095)
# - [https://www.acmicpc.net/problem/1759](https://www.acmicpc.net/problem/1759)

# - 다익스트라
# - 벨만 포드
# - 플로이드 와셜
#
# - 과제 최단경로 dp?
# - [boj][https: // www.acmicpc.net / problem / 1753](https: // www.acmicpc.net / problem / 1753)
# - [boj][https: // www.acmicpc.net / problem / 1956](https: // www.acmicpc.net / problem / 1956)
# - [boj][https: // www.acmicpc.net / problem / 4485](https: // www.acmicpc.net / problem / 4485)
# - [boj][https: // www.acmicpc.net / problem / 11404](https: // www.acmicpc.net / problem / 11404)

# 힙

#- [boj] [https://www.acmicpc.net/problem/1927](https://www.acmicpc.net/problem/1927)
#- [boj] [https://www.acmicpc.net/problem/11279](https://www.acmicpc.net/problem/11279)
#- [boj] [https://www.acmicpc.net/problem/2512](https://www.acmicpc.net/problem/2512)
#- [boj] [https://www.acmicpc.net/problem/2805](https://www.acmicpc.net/problem/2805)
#- [boj] [https://www.acmicpc.net/problem/1654](https://www.acmicpc.net/problem/1654)

#정렬
# - https://www.acmicpc.net/problem/5052 삽입
# - [https://www.acmicpc.net/problem/11650](https://www.acmicpc.net/problem/11650)
# - [boj] [https://www.acmicpc.net/problem/11651](https://www.acmicpc.net/problem/11651) 큌
# - [boj] [https://www.acmicpc.net/problem/1181](https://www.acmicpc.net/problem/1181) 병합

# - [boj] [https://www.acmicpc.net/problem/10814](https://www.acmicpc.net/problem/10814)
# - [boj] [https://www.acmicpc.net/problem/2751](https://www.acmicpc.net/problem/2751)

# 이진트리
## 이진트리
# - [https://www.acmicpc.net/problem/11725](https://www.acmicpc.net/problem/11725)
# - [https://www.acmicpc.net/problem/1068](https://www.acmicpc.net/problem/1068)
## 18장 이진탐색
# - [boj] [https://www.acmicpc.net/problem/1920](https://www.acmicpc.net/problem/1920)
# - [boj] [https://www.acmicpc.net/problem/12015](https://www.acmicpc.net/problem/12015)
# - [boj] [https://www.acmicpc.net/problem/2512](https://www.acmicpc.net/problem/2512)
# - [boj] [https://www.acmicpc.net/problem/2805](https://www.acmicpc.net/problem/2805)
# - [boj] [https://www.acmicpc.net/problem/1654](https://www.acmicpc.net/problem/1654)
# # https://programmers.co.kr/learn/courses/30/lessons/43238
# # https://programmers.co.kr/learn/courses/30/lessons/43236

# 분할정복

# - [boj][https: // www.acmicpc.net / problem / 11444](https: // www.acmicpc.net / problem / 11444)
# - [boj][https: // www.acmicpc.net / problem / 1992](https: // www.acmicpc.net / problem / 1992)


#dp


# Eulerian Path
# Hamiltonian Path
# TSP

# 1 graphAlgorithm
# 2 jenkins
# 3 DP

#

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gp = GraphProblem()
    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]
    grid2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]

    #print(gp.numIslands(grid))
    #print(gp.numIslands(grid2))

    #print(gp.letterCombinationsOfPhoneNumber("23"))

    #print(gp.permutation([1,2,3]))
    #print(gp.permutationItertools([1,2,3]))
    #print(gp.comintationItertools(4,2))
    #print(gp.combination(4,2))
    #print(gp.combination(5,3))
    #print(gp.combinationSum([2,3,6,7], 7))
    #print(gp.combinationSum([2,3,5], 8))
    #print(gp.subsets([1,2,3]))
    #print(gp.itinerary([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))
    #print(gp.itinerary([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]))
    #print(gp.itineraryStck([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))
    #print(gp.itineraryItertionStack([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))
    #print(gp.itineraryItertionStack([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]))

    #print(gp.courseSchedule([[1,0]]))
    #print(gp.courseSchedule([[1,0], [0,1]]))
    #print(gp.courseSchedule([[0, 1], [0, 2], [1, 2]]))
    #print(gp.canFinish([[0, 1], [0, 2], [1, 2]]))


    #print(gp.networkDelayTime([[2,1,1], [2,3,1], [3,4,1]], N=4, K=2))
    #print(gp.cheapestFlights(n=3, edges=[[0,1,100], [1, 2, 100], [0,2,500]], src=0, dst=2, K=0))

    # trie = Trie()
    # trie.insert("apple")
    # trie.search("apple")
    # trie.search("app")
    # trie.startWith("app")
    # trie.insert("app")
    # trie.search("app")

    tP = Solution()
    #print(tP.palindromePairs(["abcd", "dcba", "lls", "s", "sssll"]))  # [[0,1],[1,0],[3,2],[2,4]]
    # print(tP.palindromePairs(["bat", "tab", "cat"]))  # [[0,1],[1,0]]
    # print(tP.palindromePairs(["a", ""]))  # [[0,1],[1,0]]
    #
    #print(tP.palindromePairsTrie(["abcd","dcba","lls","s","sssll"])) #[[0,1],[1,0],[3,2],[2,4]]
    # print(tP.palindromePairsTrie(["bat","tab","cat"])) # [[0,1],[1,0]]
    # print(tP.palindromePairsTrie(["a",""])) #[[0,1],[1,0]]
    #print(tP.palindromePairsTrie(["d", "cbbcd", "dcbb", "dcbc", "cbbc", "bbcd"]))

    #print(tP.palindromePairs(["d", "cbbcd", "dcbb", "dcbc", "cbbc", "bbcd"]))

    #Binary Tree
    bt = TreeProblem()
#     print(bt.maxDepth(TreeNode(val=3, left=TreeNode(val=9), right=TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7)))))
#
#     print(bt.diameterBT(TreeNode(val= 1, left= TreeNode(val= 2, left= TreeNode(val= 4), right= TreeNode(val= 5)), right= TreeNode(val= 3))))
#
#     print(bt.invertTree(TreeNode{val: 4, left: TreeNode{val: 2, left: TreeNode{val: 1, left: None, right: None}, right: TreeNode{val: 3, left: None, right: None}}, right: TreeNode{val: 7, left: TreeNode{val: 6, left: None, right: None}, right: TreeNode{val: 9, left: None, right: None}}}
# ))
#
#     print(bt.margeTree(TreeNode{val: 1, left: TreeNode{val: 3, left: TreeNode{val: 5, left: None, right: None}, right: None}, right: TreeNode{val: 2, left: None, right: None}} TreeNode{val: 2, left: TreeNode{val: 1, left: None, right: TreeNode{val: 4, left: None, right: None}}, right: TreeNode{val: 3, left: None, right: TreeNode{val: 7, left: None, right: None}}}
#     ))
#
#     print(bt.isBalanced(TreeNode{val: 3, left: TreeNode{val: 9, left: None, right: None}, right: TreeNode{val: 20, left: TreeNode{val: 15, left: None, right: None}, right: TreeNode{val: 7, left: None, right: None}}}
# ))
#     print(bt.minimumHeightTree(4, [[1,0],[1,2],[1,3]]))
#     print(bt.minimumHeightTree(6, [[0, 3],[1,3],[2,3], [4,3], [5,4]]))

    BS = BinarySearch()
    # print(BS.binarySearch(nums = [-1,0,3,5,9,12], target= 9))
    # print(BS.binarySearchRecursion(nums = [-1,0,3,5,9,12], target= 9))
    # print(BS.binarySearchBisect(nums = [-1,0,3,5,9,12], target= 9))
    # print(BS.binarySearchIndex(nums = [-1,0,3,5,9,12], target= 9))
    # print(BS.searchInRotatedSortedArray(nums = [4,5,6,7,0,1,2], target= 1))
    # print(BS.searchInRotatedSortedArray(nums = [4,5,6,7,0,1,2], target= 3))
    # print(BS.searchInRotatedSortedArray(nums = [1], target= 0))
    # print(BS.intersectionOfTwoArrays([1,2,2,1], [2,2]))
    # print(BS.intersectionOfTwoArrays([4,9,5], [9,4,9,8,4,]))
    # print(BS.intersectionOfTwoArraysTwoPointer([1,2,2,1], [2,2]))
    # print(BS.intersectionOfTwoArraysTwoPointer([1,2,2,1], [2,2]))

    # print(BS.twoSum([2,7,11,15], 9))
    # print(BS.twoSumBinarySearch([2,7,11,15], 9))
    # print(BS.twoSumBisect([2,7,11,15], 9))
    #
    # print(BS.search2D([[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=3))
    # print(BS.search2DBS([[1,3,5,7],[10,11,16,20],[23,30,34,60]], target=13))
    #print(BS.fixedPoint([-15, -4, 2, 8, 13]))
    # print(BS.wifi(5, 3, [1,2,8,4,9]))
    #print(BS.budget([120, 110, 140, 150], 485))
   # print(BS.cutTree([20, 15, 10, 17], 7))
   # print(BS.cutTree([4, 42, 40, 26, 46], 20))
   #  print(BS.cutLan([802,743,457,539],11))
   #  print(BS.budget([120, 110, 140, 150], 485))
   #  print(BS.budget([70, 80, 30, 40, 100], 450))

    #print(BS.checkMars([[5,5,4],[3,9,1],[3,2,7]]))
    #print(BS.checkMars([[3,7,2,0,1],[2,8,0,9,1],[1,2,1,8,1],[9,8,9,2,0],[3,6,5,1,5]]))
    #print(BS.checkMars([[9,0,5,1,1,5,3],[4,1,2,1,6,5,3],[0,7,6,1,6,8,5],[1,1,7,8,3,2,3],[9,4,0,7,6,4,1],[5,8,3,2,4,8,3],[7,4,8,4,8,3,4]]))
    #print(BS.hideSeek([[3,6], [4,3], [3,2], [1,3], [1,2],[2,4],[5,2]]))
    print(BS.floyd(5, [[1,2,2], [1,3,3], [1,4,1], [1,5,10], [2,4,2], [3,4,1], [3,5,1], [4,5,3], [3,5,10],[3,1,8],[1,4,2],[5,1,7],[3,4,2],[5,2,4]]))