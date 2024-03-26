#3/23 Dynamic Programming
##파이썬코딩인터뷰
##이코테
##문제은행 + 문제
##programmers
##백준
##leetcode
##hackerrank (week13)

import bisect
import collections
import copy
import functools
import heapq
import itertools
import re


def fibonacci(n):
    dp = [] * n
    if n == 1:
        dp[1] = 1
    if n ==2:
        dp[2] = 2
    for i in range(3, len(n)+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

def zero_one_knapsack(cargo):
    capacity = 15

    pack = []
    #[[4,12],[2,1],[10,4], [1,1],[2,2]]

    for i in range(len(cargo) + 1):
        pack.append([])
        for j in range(capacity+1):

            if i == 0 or j == 0:
                pack[i].append(0)

            elif cargo[i-1][1] < j :
                pack[i].append(max(
                    cargo[i-1][0] + pack[i-1][j-cargo[i-1][1]],
                    pack[i-1][j]
                ))

            else:
                pack[i].append(pack[i-1][j])

    return pack[-1][-1]

#최대서브배열
def maximumSubarray(nums):
    dp = [0] * len(nums)

    for i in range(len(nums)):
        if i == 0:
            dp[0] = nums[i]
        elif nums[i] > 0:
            dp[i] = dp[i-1]+nums[i]
        else:
            dp[i] = dp[i-1]
    return dp[len(nums)-1]

#가장 긴 증가하는 부분수열 https://www.acmicpc.net/problem/12015
def longestSubarray(nums):
    #dp 로 풀기 순서 X
    dp = [1 for _ in range(len(nums))]

    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j]+1)
    print(max(dp))
    # for i in range(1, len(nums)):

    #
    #     if dp[i-1][-1] < nums[i]:
    #         dp[i-1].append(nums[i])
    #     else:

    #binary search 로 풀기
    dp = list()
    dp.append(nums[0])

    for i in range(1, len(nums)):
        if dp[-1] < nums[i]:
            dp.append(nums[i])
        else:
            idx = bisect.bisect_left(dp, nums[i])
            dp[idx] = nums[i]
    return dp

#계단오르기
def stand(n):
    dp = [0 for _ in range(n+1)]

    dp[1] = 1

    dp[2] = 2
    for i in range(3, n+1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

#집도둑
def theif(nums):
    dp = [0 for _ in range(len(nums))]
    dp[0] = nums[0]
    for i in range(1, len(nums)):
        dp[i] = max(
            dp[i-1],
            dp[i-2] + nums[i]
        )

##
#1로 만들기
def make1(X):
    dp = collections.defaultdict(list)
    dp[0] = X

    i = 1
    while True: #X 일 때 까지

        for x in dp[i - 1]:
            if x ==  1:
                return i - 1
            dp[i].append(x - 1)
            if x % 5 == 0:
                dp[i].append(x // 5)
            if x % 3 == 0:
                dp[i].append(x // 3)
            if x % 2 == 0:
                dp[i].append(x // 2)
        i += 1
#개미전사
def ant(store):
    dp = [0 for _ in range(len(store))]

    dp[0] = store[0]
    dp[1] = max(dp[0], store[1])

    for i in range(2, len(store)):
        dp[i] = max(dp[i-1], dp[i-2] + store[i])
    return dp[i]

#바닥공사
def floor(N):
    dp = [0 for _ in range(N)]
    dp[1] = 1
    dp[2] = 2
    for i in range(3, N):
        dp[i] = dp[i-1] + dp[i-2]  * 2
    return dp[N]

#화폐구성
def currency(N, M):
    dp = [10001 for _ in range(len(N)+1)]
    dp[0] = 0

    # for i in range(len(N)+1):
    #     if dp[i] != 10001:
    #         for k in M:
    #             dp[i+k] = min(dp[i+k] , dp[i]+1)

    for i in range(len(M)):
        for j in range(len(N)):
            if dp[i-N[j]] != 10001:
                dp[i] = min(dp[i], dp[i-N[j]]+1)

    return dp[M]

#금광
def gold(store):
    #[[1,3,3],[2,2,1],[4,1,0],[6,4,7]]
    dp = [[0 for _ in range(len(store[0])+1)] for _ in range(len(store)+1)]

    count = 1
    for i in range(1, len(store[0])+1):
        for j in range(1, len(store)+1):
            if i == 1 and j == 1:
                dp[1][1] = store[0][0]
            else:
                dp[i][j] = max(dp[i-1][j-1], dp[i][j-1]) + store[i+1][j-1]
            count+=1
            if count == len(store) :
                return max(dp)

#정수삼각형
def triangle(N):
    n = int(input())
    d = []
    for i in range(n):
        d.append(list(map(int, input().split())))

    for i in range(1, len(d)):
        for j in range(i+1):
            if j == 0 :
                d[i][j] = d[i][j] + d[i-1][0]
            elif j == i:
                d[i][j] = d[i][j] + d[i-1][i-1]

            else:
                d[i][j] = d[i][j] + max(d[i-1][j-1], d[i-1][j])
    return max(d[i])

#퇴사
def exit():
    n = 7
    timeTable = [[3, 10], [5, 20], [1, 10], [1, 20], [2, 15], [4, 40], [2, 200]]

    dp = [0 for _ in range(n+1)]

    for i in range(n-1, -1, -1):
        if timeTable[i][0] + i <= n:
            dp[i] = max(
                dp[i+1],
                timeTable[i][1] + dp[i + timeTable[i][0]]
            )
        else:
            dp[i] = dp[i+1]

    print(dp[0])
    print(dp)
    #return max(dp)

#병사배치
def soldier(): #LIS
    # 병사 수
    import sys
#N = int(sys.stdin.readline())
# 병사들의 전투력
#p = list(map(int, sys.stdin.readline().split()))
    N = 7
    p = [15, 11, 4,8,5,2,4]
    # for power in powers:
    #         p.append(-power)
    p.reverse()

    l = list()
    l.append(p[0])
    for i in range(1, N):
        if l[-1] < p[i]:
            l.append(p[i])
        else:
            idx = bisect.bisect_left(l, p[i])
            l[idx] = p[i]
    print(N - len(l))
    return len(p) - len(l)

#못생긴 수
def uglyNumber():
    # n = 9
    # ugly = [0] * n
    # ugly[0] = 1
    #
    # # 2,3,5의 배수의 각 인덱스
    # i2 = i3 = i5 = 0
    # next2, next3, next5 = 2, 3, 5
    #
    # for l in range(1, n):
    #     ugly[l] = min(next2, next3, next5)
    #     if ugly[l] == next2:
    #         i2 += 1
    #         next2 = ugly[i2] * 2
    #     if ugly[l] == next3:
    #         i3 += 1
    #         next3 = ugly[i3] * 3
    #     if ugly[l] == next5:
    #         i5 += 1
    #         next5 = ugly[i5] * 5
    # print(ugly)
    # print(ugly[n - 1])

    N = 9
    i = j = k = 0
    x = 2
    y = 3
    z = 5
    dp = [0 for _ in range(N)]
    dp[0] = 1

    for p in range(1, N):
        dp[p] = min(x, y, z)
        if dp[p] == x:
            i += 1
            x = dp[i] * 2
            print("x", x)

        if dp[p] == y:
            j += 1
            y = dp[j] * 3
            print("y", y)

        if dp[p] == z:
            k += 1
            z = dp[k] * 5
    print(dp)
    print(dp[N-1])

#최소 편집, 편집 거리
def editDistance(A, B):
    n = len(A)
    m = len(B)

    dp = [[0] * (m+1) for _ in range(n+1)]

    for i in range(1, n+1):
        dp[i][0] = i

    for j in range(1, m+1):
        dp[0][j] = j


    for i in range(1, n+1):
        for j in range(1, m+1):
            if A[i-1] == B[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
    return dp[-1][-1]



#가장긴증가하는부분수열

#신나는함수실행
def w(a,b,c): #dp?

    #dp = [[[1 for _ in range(a)] for _ in range(b)] for _ in range(c)]

    if a <= 0 or b <= 0 or c <= 0 :
        return 1
    if a > 20 or b > 20 or c > 20:
        return w(20, 20, 20)
    if dp[a][b][c]:
        return dp[a][b][c]
    if a<b<c:
        dp[a][b][c] = w(a, b, c-1) + w(a, b-1, c-1) - w(a, b-1, c)
        return dp[a][b][c]
    dp[a][b][c] = w(a-1, b, c) + w(a-1, b-1, c) + w(a-1, b, c-1) - w(a-1, b-1, c-1)
    return dp[a][b][c]
dp = [[[0] * 21 for _ in range(21)] for _ in range(21)]

# while True:
#     a, b, c = map(int, input().split())
#     if a == -1 and b == -1 and c == -1:
#         break
#     print(f'w({a}, {b}, {c}) = {w(a, b, c)}')


#파도반수열
def padoban(N):
    dp = [0] * (N+1)

    dp[1] = 1
    dp[2] = 1
    dp[3] = 1
    dp[4] = 2
    dp[5] = 2
    for i in range(6, N+1):
        dp[i] = dp[i-1] + dp[i-5]
    print(dp[-1])
    return dp[-1]

#RGB거리
def RGBDistance(N, colors):
    # N = int(input())
    # colors = []
    # for i in range(N):
    #     colors.append(list(map(int, input().split())))

    dp = [[0 for _ in range(3)] for _ in range(N)]
    dp[0][0] = colors[0][0]
    dp[0][1] = colors[0][1]
    dp[0][2] = colors[0][2]

    for i in range(1, N):

        dp[i][0] = colors[i][0] + min(dp[i-1][1], dp[i-1][2])
        dp[i][1] = colors[i][1] + min(dp[i-1][0], dp[i-1][2])
        dp[i][2] = colors[i][2] + min(dp[i-1][0], dp[i-1][1])

    print(min(dp[-1]))
    return min(dp[N])



##programmers
##백준
##leetcode
##hackerrank (week13)
def shortPalindrome(s):
    mod = 1000000007
    result = 0
    c1 = [0 for i in range(26)]
    c2 = [[0 for i in range(26)] for j in range(26)]
    c3 = [[[0 for i in range(26)] for j in range(26)] for k in range(26)]
    for j in s:
        c  = ord(j) - ord('a')
        for i in range(26):
            result += c3[c][i][i] # ciic c3[6][0][0]
            c3[i][c][c] += c2[i][c] # icc
            c2[i][c] += c1[i] # ic
        c1[c] += 1 # c
    return result % mod

#3/24
##binarysearch

# 회전 정렬된 배열 검색
def search_in_rotated_array(nums, target):
    start = 0
    end = len(nums)-1

    while start <= end:
        mid = (start + end) // 2

        if nums[mid] == target:
            return mid

        if nums[mid] < target: # start 를 mid + 1
            if nums[mid] < nums[start]:
                end = mid - 1
            else:
                start = mid + 1


        else:
            if nums[mid] > nums[end]:
                start = mid + 1
            else:
                end = mid - 1
    return -1

# 두 배열의 교집합
def intersection(listb, listc):
    #n*m

    listb.sort()
    listc.sort()
    intersection = []
    for b in listb:
        idx = bisect.bisect_left(listc, b)
        if b not in intersection and listc[idx] == b:
            intersection.append(b)
    return intersection

# 두 수의 합 II
def sumtwo(listc, target):

    for i in range(len(listc)):
        val = target - listc[i]
        idx = bisect.bisect_left(listc, val, i+1)
        if idx < len(listc) and listc[idx] == val:
            return [i, idx]

def two_sum(nums, target):
    for i, v in enumerate(nums):
        val = target - v

        copied = nums[i+1:]
        if val in copied:
            return [i, copied.index(val) + i + 1]

# 2D 행렬 검색 II
def search2D(matrix, target):
    m = len(matrix)
    n = len(matrix[0])
    end = m - 1
    start = 0
    # 예외 처리
    if not matrix:
        return False

    while 0 <= end and start < n:
        if target == matrix[end][start]:
            return True
        if target < matrix[end][start]:
            end -= 1
        else:
            start += 1
    return False

    #return any(target in mat for mat in matrix)



## 이코테
def find_item(nums, customer):
    nums.sort()
    for cus in customer:
        idx = bisect.bisect_left(nums, cus)
        if idx < len(nums) and nums[idx] == cus:
            print("yes")
        else:
            print("no")

def teokbokki(tlist, N):
    tlist.sort()
    start = 0
    end = tlist[-1]
    while start <= end:
        mid = (start + end) // 2
        sumt = 0

        for t in tlist:
            if t- mid > 0:
                sumt += t - mid
        if sumt == N:
            return mid
        if sumt < N:
            end = mid - 1
        else:
            start = mid + 1
    return mid

def count_target(targets,t):
    lo = bisect.bisect_left(targets, t)
    ro = bisect.bisect_right(targets,t)
    if lo-ro == 0:
        return -1
    return ro - lo

def fixed_point(a):
    start = 0
    end = len(a) - 1

    while start <= end:
        mid = (start + end) // 2
        if a[mid] == mid:
            return mid
        if a[mid] > mid:
            end = mid - 1
        else:
            start = mid + 1
    return -1

import sys
def wifi(distances, wfs):
    # N, wfs = map(int, sys.stdin.readline().split())
    # distances = [int(sys.stdin.readline()) for _ in range(N)]
    distances.sort()
    start, end = 0, distances[-1]

    while start <= end:
        mid = (start + end) // 2 #가장 인접한 두 공유기의 최대 거리

        count = 0
        val = distances[0] #설치
        for i in range(1, len(distances)):
            if distances[i] - val >= mid: #공유기 설치
                val = distances[i]
                count += 1

        if count > wfs:
            end = mid - 1
        else:
            start = mid + 1
            result = mid #최적
    print(result)

#programmers
#입국심사
def immigration(n, times):
    times.sort()
    left, right = 0, times[-1] * n
    #최적
    while left <= right:
        mid = (left + right) // 2
        ppl = 0
        for time in times:
            ppl += mid // time
        if ppl < n :
            left = mid + 1
        else:
            result = mid
            right = mid - 1
    print(result)


# 징검다리
def stones(distance, rocks, n):
    # 각 지점 사이의 거리의 최솟값 중 가장 큰 값
    start = 0
    end = distance
    rocks.sort()
    while start <= end:
        mid = (start + end) // 2

        val = 0
        count = 0
        for rock in rocks:
           if rock-val < mid:
               count += 1
           else:
               val = rock

        if count > n:
            end = mid - 1
        else:
            start = mid + 1
            result = mid
    print(result)

# 공유기설치
# 가장긴증가하는부분수열2

# k번째수

def kth_number(n, k):
    left = 0
    right = k

    while left <= right:
        mid = (left + right) // 2
        count = 0
        for i in range(n):
            count += min(mid // (i+1), n)
        if count < k:
            left = mid + 1
        else:
            right = mid - 1
            result = mid
    print(result)

# 나무자르기
def cut_tree(trees, target):
    trees.sort()
    left = 0
    right = trees[-1]

    while left <= right:
        mid = (left + right) // 2

        c = 0
        for tree in trees:
            c += tree - mid if tree - mid > 0 else 0
        if c < target:
            right = mid - 1
        else:
            left = mid + 1
            result = mid
    print(result)


# https://www.acmicpc.net/problem/1654 랜선자르기
def cut_lan(lans, k):
    lans.sort()
    left = 0
    right = lans[-1]
    while left <= right:
        mid = (left + right) // 2
        count = 0
        for lan in lans:
            count += lan // mid
        if count < k:
            right = mid - 1
        else:
            result = mid
            left = mid + 1
    print(result)

# https://www.acmicpc.net/problem/2512 예산
def budget(bdgts, t):
    bdgts.sort()
    left = 0
    right = bdgts[-1]
    while left <= right:
        mid = (left + right) // 2
        count = 0
        for bdgt in bdgts:
            count +=mid if bdgt > mid else bdgt
        if count > t:
            right = mid - 1
        else:
            result = mid
            left = mid + 1
    print(result)



from bisect import bisect_left, bisect_right

def count_range(arr, left, right):
    l_o = bisect_left(arr, left)
    r_o = bisect_right(arr, right)
    return r_o - l_o
array = collections.defaultdict(list)
rev_array = collections.defaultdict(list)
def search_lyrics(words, queries):
    answer = []

    # 1) 이진탐색
    for word in words:
        array[len(word)].append(word)
        rev_array[len(word)].append(word[::-1])
    for i in array:
        array[i].sort()
        rev_array[i].sort()
    for query in queries:
        if query[-1] == '?':
            res = count_range(array[len(query)], query.replace('?', 'a'), query.replace('?', 'z'))
        else:
            res = count_range(rev_array[len(query)], query[::-1].replace('?', 'a'), query[::-1].replace('?', 'z'))
        answer.append(res)
    return answer

    #2) trie
# class TrieNode:
#     def __init__(self):
#         self.len = []
#         self.children = collections.defaultdict(TrieNode)
#         self.val = ''
#
# class Trie:
#     def __init__(self):
#         self.root = TrieNode()
#
#     def insert(self, word):
#         node = self.root
#         l = len(word)
#         for w in word:
#             node = node.children[w]
#             node.val = w
#             node.len.append(l)
#
#     def search(self, query, length):
#         node = self.root
#         while query:
#             if query[0] == '?':
#                 return node.len.count(length)
#             elif query[0] in node.children:
#                 node = node.children[query[0]]
#                 query= query[1:]
#             else:
#                 return 0

def make_trie(trie, words):
    for word in words:
        cur = trie
        l = len(word)
        for w in word:
            if w in cur:
                cur = cur[w]
                cur['!'].append(l)
            else:
                cur[w] = {}
                cur = cur[w]
                cur['!'] = [l]
    return trie
def search_trie(trie, query, length):
    count = 0
    if query[0] == '?':
        print(trie)
        return trie['!'].count(length)
    elif query[0] in trie:
        count += search_trie(trie[query[0]], query[1:], length)
    return count

def search_lyrics_trie(words, queries):

    answer = []

    trie, rev_trie = Trie(), Trie()
    rev_words, counted = [], []
    for w in words:
        trie.insert(w)
        rev_trie.insert(w[::-1])
        counted.append(len(w))

    for query in queries:
        if query[0] == '?' and query[-1] == '?':
            answer.append(counted.count(len(query)))
        elif query[-1] =='?':
            answer.append(trie.search(query,len(query)))
        elif query[0] == '?':
            answer.append(rev_trie.search(query[::-1], len(query)))
    return answer
    # answer = []
    # rev_words, counted = [], []
    # for w in words:
    #     rev_words.append(w[::-1])
    #     counted.append(len(w))
    # trie = make_trie({}, words)
    # rev_trie = make_trie({}, rev_words)
    #
    # for query in queries:
    #     if query[0] == '?' and query[-1] == '?':
    #         answer.append(counted.count(len(query)))
    #     elif query[-1] == '?':
    #         answer.append(search_trie(trie, query, len(query)))
    #     elif query[0] == '?':
    #         answer.append(search_trie(rev_trie, query[::-1], len(query)))
    # print(answer)

# trie
class TrieNode:
    def __init__(self):
        self.val = ''
        self.word_id = -1 #
        self.palindrome_word_ids = []
        self.children = collections.defaultdict(TrieNode)
class Trie:
    def __init__(self):
        self.root = TrieNode()

    @staticmethod
    def is_palindrome(word):
        return word[::] == word[::-1]

    def insert(self, word, index):
        node = self.root
        for i, w in enumerate(reversed(word)):
            if self.is_palindrome(word[0:len(word)-i]):
                node.palindrome_word_ids.append(index)
            node = node.children[w]
            node.val = w
        node.word_id = index

    def search(self, query, index):
        result = []
        node = self.root

        while query:
            # 3) 경우
            if node.word_id >= 0:
                if self.is_palindrome(query):
                    # print("경우3")
                    result.append([index, node.word_id])
            if query[0] not in node.children:
                return result
            node = node.children[query[0]]
            query = query[1:]
        #return [index, node.word_id]
        #1) 경우
        if node.word_id >= 0 and node.word_id != index:
            # print("경우1")
            result.append([index, node.word_id])
        #2) 경우
        for palindrome in node.palindrome_word_ids:
            # print("경우 2")
            result.append([index, palindrome])
        return result


def palindromeParis(words):
    trie = Trie()
    for i, word in enumerate(words):
        trie.insert(word, i)
    results = []
    for i, word in enumerate(words):
        # print(word)
        results.extend(trie.search(word, i))
    return results

# hackerrank week 12


#3/25 그래프 bfs/dfs 최단경로

##파이썬코딩인터뷰
def component(x, y, visited, grid):
    x_lst = [0, 1, 0, -1]
    y_lst = [1, 0, -1, 0]
    # print("check", visited[x][y], x, y)
    visited[x][y] = True

    if grid[x][y] == '1' :
        for i in range(4):
            x_new = x + x_lst[i]
            y_new = y + y_lst[i]
            if 0 <= x_new < len(grid) and 0 <= y_new < len(grid[0]) and visited[x_new][y_new] == False:
                #print("위치", x_new, y_new)
                #ODO: visited 확인
                component(x + x_lst[i], y+y_lst[i], visited, grid)

    return visited


def numIslands(grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]

    count = 0

    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == '1' and visited[x][y] == False:
                print("새로시작-------")
                visited = component(x, y, visited, grid)
                count += 1
    return count

def letterCombinations(digits): #23
    dic = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
           "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
    if not digits:
        return []
    results = []
    def dfs(index, path):
        if len(digits) == len(path):
            results.append(path)
            return
        for i in range(index, len(path)): #0 1
            for j in dic[digits[i]]:   #2 - abc def
                dfs(i, path+j)

    dfs(0, "")
    return results

#순열
def permutation(nums):
    print(list(itertools.permutations(nums)))
    #return itertools.permutations

# #조합
def combination(n, k):
    print(list(itertools.combinations([1,2,3,4], 2)))
    #return itertools.combinations

def combinationSum(candidates, target): #can be used multiple times
    #print(list(itertools.combinations(candidates, 2)))
    def dfs(csum, index, result):
        if csum < 0 :
            return
        if csum == 0:
            results.append(result)
            return

        for i in range(index, len(candidates)):
            dfs(csum-candidates[i], i, result+[candidates[i]])
    results = []
    dfs(target, 0, [])

    return results

def subsets(nums):
    # print(list(itertools.combinations(nums, 2)))

    results = []
    def dfs(index, result):
        # if len(result) == 3:
        #     results.append(result)
        #
        #     return
        results.append(result)

        for i in range(index, len(nums)):
            dfs(i+1, result + [nums[i]])

    dfs(0, [])
    return results

def findItinerary(tickets): # the smallest lexical order when read as a single string.
    #dictionary
    results = []
    tickets_dict = collections.defaultdict(list)
    for ticket in sorted(tickets, reverse=True):
        tickets_dict[ticket[0]].append(ticket[1])

    #dfs
    src = "JFK"
    #neighbor
    visited = []
    def dfs(result):
        while tickets_dict[result]:
            dfs(tickets_dict[result].pop())
        results.append(result)

    dfs("JFK")

    return results[::-1]

def canFinish(numCourses, prerequisites): #Course Schedule
    #Circled link 확인
    graph = collections.defaultdict(list)
    for x, y in prerequisites:
        graph[x].append(y)

    traced = set()
    visited = set()

    def dfs(i):
        #순환구조이면 false
        if i in traced:
            return False

        #이미 방문했던 노드
        if i in visited:
            return True
        traced.add(i)
        for y in graph[i]:
            if not dfs(y):
                return False

        traced.remove(i) #순환 확인 후 제거
        visited.add(i)
        return True

    for x in list(graph):
        if not dfs(x):
            return False  #순환구조 O
    return True #순환구조 X

#최단경로 heapq
def networkDelayTime(times, n, k):
    graph = collections.defaultdict(list)
    for time in times:
        graph[time[0]].append(([time[1], time[2]]))
    Q = [(0, k)]
    ##
    dist = collections.defaultdict(list)
    # heapq.heappush(Q, ((0, k, 0)))
    while Q:
        t, src = heapq.heappop(Q)
        if src not in dist:
            dist[src] = t
            for tar, t in graph[src]:
                alt = t + t
                heapq.heappush(Q, (alt, tar))
    #모든 노드 최단 경로 존재 여부 판별
    if len(dist) == n:
        return max(dist.values())

    return -1

def findCheapestPrice(n, flights, src, dst, k):
    graph = collections.defaultdict(list)
    for s, v, c in flights:
        graph[s].append((v, c))
    # heapq
    Q = [(0, src, 0)]
    heapq.heapify(Q)
    while Q:
        price, src, K = heapq.heappop(Q)
        if K > k:
            return -1 #False
        for v, c in graph[src]:
            if v == dst:
                return c + price
            heapq.heappush(Q, (price+c, v, K+1))

    return -1 #over, no



##이코테
def component_j(x, y, visited, grid):
    x_lst = [0, 1, 0, -1]
    y_lst = [1, 0, -1, 0]
    # print("check", visited[x][y], x, y)
    visited[x][y] = True

    if grid[x][y] ==0 :
        for i in range(4):
            x_new = x + x_lst[i]
            y_new = y + y_lst[i]
            if 0 <= x_new < len(grid) and 0 <= y_new < len(grid[0]) and visited[x_new][y_new] == False:
                #print("위치", x_new, y_new)
                #ODO: visited 확인
                component_j(x + x_lst[i], y+y_lst[i], visited, grid)

    return visited


def juiceComponent(grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]

    count = 0

    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 0 and visited[x][y] == False:
                visited = component_j(x, y, visited, grid)
                count += 1
    return count

#미로탈출 ... > (시간이 오래걸리는데, heapq X) dfs / dp
def maze(n, m, grid): #bfs queue deque ???
    #2) deque
    x_lst = [1, 0, 0, -1]
    y_lst = [0, 1, -1, 0]
    queue = collections.deque()
    queue.append((0, 0))
    while queue:
        x, y = queue.popleft()
        for i in range(4):
            newx = x + x_lst[i]
            newy = y + y_lst[i]
            if 0 <= newx < n and 0 <= newy < m:
        # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
                if grid[newx][newy] == 1:
                    grid[newx][newy] = grid[x][y] + 1
                    queue.append((newx, newy))
    return grid[n-1][m-1]


    #1) heapq
    #visited
    x_lst = [1,0]
    y_lst = [0,1]
    Q = []
    heapq.heappush(Q, (0, (0, 0)))
    #heapq.heapify(Q)
    while Q:
        t, (x, y) = heapq.heappop(Q)
        print((x, y))
        if x == n-1 and y == m -1:
            return t + 1
        for i in range(2):
            new_x = x + x_lst[i]
            new_y = y + y_lst[i]
            if 0 <= new_x < n and 0 <= new_y < m:
                if grid[new_x][new_y] == 1:
                    heapq.heappush(Q, (t+1, (new_x, new_y)))
    return -1


#Floyd Warshall Algorithm
def Floyd(n, m, wlist):
    INF = int(1e9)

    graph = [[INF for _ in range(n+1)] for _ in range(n+1)]
    for s, r, v in wlist:
        graph[s][r] = v
    for i in range(n+1):
        graph[i][i] = 0

    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                graph[i][j] = min(graph[i][j] , graph[i][k] + graph[k][j])
    print(graph)

def future(n,m, flist, p, q): # n 노드 m link / 1 시작 p 거쳐 q 도착
    INF = int(1e9)
    #최단경로 - 다익스트라 알고리즘, 플로이드워셜 알고리즘
    graph = [[INF for _ in range(n+1)] for _ in range(n+1)]
    for x, y in flist:
        graph[x][y] = min(graph[x][y], 1)
    for i in range(n+1):
        graph[i][i] = 0

    # k, x ? => 플로이드 워셜 문제
    for k in range(1, n+1):
        for a in range(1, n+1):
            for b in range(1, n+1):
                graph[a][b] = min(graph[a][b], graph[a][k] + graph[k][b])

    distance = graph[1][p] + graph[p][q]

    if distance >= INF:
        print("-1")
    else:
        print(distance+1)

def telegraph(N, M, start, gr): #출력, C에서 메시지받는 도시의 개수, 걸리는 시간
    #플로이드워셜
    #다익스트라
    INF = int(1e9)  # 무한을 의미하는 값으로 10억을 설정
    dist = [INF] * (N + 1)
    graph = collections.defaultdict(list)
    for x,y,z in gr:
        graph[x].append((y, z))

    Q = []
    heapq.heappush(Q, (0, start))
    dist[start] = 0
    while Q:
        d, n = heapq.heappop(Q)
        if dist[n] < d:
            continue
        for i , j in graph[n]:
            d_n = d + j
            if d_n < dist[i]:
                dist[i] = d_n
                heapq.heappush(Q, (d_n, i))

    count = 0
    max_distance = 0
    for d in dist:
        if d != 1e9:
            count += 1
            max_distance = max(max_distance, d)
    print(count-1, max_distance)



def find_parent(parent, x):
    if parent[x] != x:
        parent[x] = find_parent(parent, x)
    return parent[x]

def union_parent(parent, a,b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

def kruskal(n,m,klist):
    parent = [0] * (n + 1)
    for i in range(1, n+1):
        parent[i] = i
    graph = []
    for s, v, l in klist:
        graph.append((l, s, v))
    graph.sort() #비용순으로 정렬
    result = 0
    for edge in graph:
        cost, a, b = edge
        if find_parent(parent, a) != find_parent(parent, b):
            union_parent(parent, a, b)
            result += cost

    print(result)

def topology_sort(v, e, tlist):
    indegree = [0] * (v + 1)

    graph = collections.defaultdict(list)
    for s, vw in tlist:
        indegree[vw] += 1
        graph[s].append(vw)
    #진입차수

    result = []
    q = collections.deque()

    for i in range(1, v+1):
        if indegree[i] == 0:
            q.append(i)

    while q:
        now = q.popleft()
        result.append(now)

        for n in graph[now]:
            indegree[n] -= 1
            if indegree[n] == 0:
                q.append(n)
    print(result)

def find_parent(parent, a):
    if parent[a] != a:
        parent[a]  = find_parent(parent, parent[a])
    return parent[a]

def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

def check_team(parent, a, b):
    return find_parent(parent, a) == find_parent(parent, b)
#팀결성
def team(n, m, tlist):
    kpa = [0] * (n + 1)
    for i in range(1, n+1):
        kpa[i] = i

    for i, s, v in tlist:
        if i == 0:
            union_parent(kpa, s, v)
        else:
            print(check_team(kpa, s, v))

def city_plan(n, m, clist):
    parent = [0] * (n + 1)
    for i in range(1, n+1):
        parent[i] = i
    graph = []
    for s, v, cost in clist:
        graph.append((cost, s, v))

    graph.sort()
    print(graph)
    #
    cost = 0
    last = 0
    for p, s, v in graph:
        if find_parent(parent, s) != find_parent(parent, v):
            union_parent(parent, s, v)
            #print(cost)

            cost += p
            last = p
    print(cost-last)

def curriculum(n, clist):
    indegree = [0] * (n+1)
    graph = collections.defaultdict(list)
    time = [0] * (n + 1)

    for i, li in enumerate(clist):
        time[i+1] = li[0]
        for pre in li[1:]:
            graph[pre].append(i+1)
            indegree[i+1] += 1

    print(time)
    print(indegree)
    print(graph)
    #topology
    result = copy.deepcopy(time)
    q =  collections.deque()

    for i in range(1, n+1): #진입차수가 0
        if indegree[i] == 0:
            q.append(i)

    while q:
        now = q.popleft()
        for i in graph[now]: #2,3,4
            result[i] = max(result[i], result[now] + time[i])
            indegree[i] -= 1
            if indegree[i] == 0:
                q.append(i)

    #결과
    for i in range(1, n+1):
        print(result[i])


#-알고리즘유형별기출문제
#특정거리의 도시 찾기
def find_shortest(N, M, K, X, flist):# 도시의 개수, 도로의 개수, 거리 정보, 출발 도시 번호
    #특정거리에 있는 도시 찾기 BFS
    q = collections.deque()

    graph = collections.defaultdict(list)
    for s, v in flist:
        graph[s].append(v)
    q.append(X)
    s = [int(1e9)] * (N+1)
    r = 0
    s[X] = 0
    while q:
        st = q.popleft()
        r += 1
        for neighbor in graph[st]:
            s[neighbor] = min(r, s[neighbor])
            if r > K:
                break
            q.append(neighbor)
    for i in range(1, N + 1):
        if s[i] == K:
            print(i)

    return

    # N, M, K, X = map(int, input().split())
    # graph = collections.defaultdict(list)
    # #
    # # 모든 도로 정보 입력 받기
    # for _ in range(M):
    #     a, b = map(int, input().split())
    #     graph[a].append(b)
    # INF = int(1e9)
    # visited = [False] * (N+1)
    graph = collections.defaultdict(list)
    for s, v in flist:
        graph[s].append(v)
    distances = [-1] * (N+1)
    distances[X] = 0

    #bfs
    q = collections.deque([X])
    while q:
        now = q.popleft()
        for neighbor in graph[now]:

            if distances[neighbor] == -1:
                distances[neighbor] = distances[now] + 1
                if distances[neighbor] < K :
                    q.append(neighbor)


    # #dfs
    # def dfs(X, graph, visited, distances):
    #     visited[X] = True
    #
    #     print(X, graph, visited, distances)
    #     for neighbor in graph[X]:
    #         distances[neighbor] = min(distances[X] + 1, distances[neighbor])
    #         if distances[neighbor] > K:
    #             return
    #         dfs(neighbor, graph, visited, distances)
    #
    # dfs(X, graph, visited, distances)

    check = False
    for i in range(len(distances)):
        if distances[i] == K:
            print(i)
            check = True
    # if K not in distances:
    if check == False:
        print(-1)


#PyPy3 연구소

result = 0
def laboratory(n, m, grid):
    #temp = [[0] * m for _ in range(n)]  # 벽을 설치한 뒤의 맵 리스트

    v = []

    for i in range(n):
        for j in range(m):
            if grid[i][j] == 2:
                v.append([i, j])

    def get_score(temp):
        score = 0
        for i in range(n):
            for j in range(m):
                if temp[i][j] == 0:
                    score +=1
        return score

    def virus(x, y, temp):
        x_list = [0, 0, -1, 1]
        y_list = [-1, 1, 0, 0]

        for i in range(4):
            nx = x + x_list[i]
            ny = y + y_list[i]
            if 0 <= nx < n and 0 <= ny < m:
                if temp[nx][ny] == 0:
                    temp[nx][ny] = 2
                    virus(nx, ny, temp)

    def dfs(count):
        global result
        temp = [[0] * m for _ in range(n)]
        if count == 3:
            for i in range(n):
                for j in range(m):
                    temp[i][j] = grid[i][j]
            # 각 바이러스의 위치에서 전파 진행
            for i in range(n):
                for j in range(m):
                    if temp[i][j] == 2:
                        virus(i, j, temp)
            result = max(result, get_score(temp))
            return
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    count += 1
                    dfs(count)
                    count -= 1
                    grid[i][j] = 0

    dfs(0)
    return result


     # 벽을 설치한 뒤의 맵 리스트

    def get_score(temp):
        score = 0
        for i in range(n):
            for j in range(m):
                if temp[i][j] == 0:
                    score += 1
        return score

    def virus(x, y, temp):
        x_list = [0, 0, -1, 1]
        y_list = [-1, 1, 0, 0]
        for i in range(4):
            nx = x + x_list[i]
            ny = y + y_list[i]
            if 0 <= nx < n and 0 <= ny < m:
                if temp[nx][ny] == 0:
                    temp[nx][ny] = 2
                    virus(nx, ny, temp)

    def dfs(count):
        global result
        temp = [[0] * m for _ in range(n)]
        if count == 3:
            for i in range(n):
                for j in range(m):
                    temp[i][j] = grid[i][j]
            #바이러스 전파
            for i in range(n):
                for j in range(m):
                    if temp[i][j] == 2:
                        virus(i, j, temp)
            result = max(result, get_score(temp))
            return

        #벽 설치
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 0:
                    grid[i][j] = 1
                    count += 1
                    dfs(count)
                    grid[i][j] = 0
                    count -= 1
    print("Start", grid)
    dfs(0)
    print(result)

#경쟁적 전염
def contagion(n,m, grid, S, x, y):
    #dfs
    q = list()
    for i in range(n):
        for j in range(m):
            if grid[i][j] != 0:
                q.append([grid[i][j], i, j])
    q.sort() #

    dq = collections.deque()
    for v, i, j in q:
        dq.append([i, j, 0])
    print(dq)
    r = 0
    x_list = [-1, 1, 0, 0]
    y_list = [0, 0, 1, -1]


    while dq:
        i, j, r = dq.popleft()
        print(r, i, j)
        if r == S:
            print(grid)
            return grid[x-1][y-1]
        for a in range(4):
            nx = i + x_list[a]
            ny = j + y_list[a]
            if 0 <= nx < n and 0 <= ny < m:
                if grid[nx][ny] == 0:
                    grid[nx][ny] = grid[i][j]
                    dq.append([nx, ny, r+1])
    return
    from collections import deque

    n, k = map(int, input().split())

    graph = []  # 전체 보드 정보를 담는 리스트
    data = []  # 바이러스에 대한 정보를 담는 리스트

    for i in range(n):
        # 보드 정보를 한 줄 단위로 입력
        graph.append(list(map(int, input().split())))
        for j in range(n):
            # 해당 위치에 바이러스가 존재하는 경우
            if graph[i][j] != 0:
                # (바이러스 종류, 시간, 위치 X, 위치 Y) 삽입
                data.append((graph[i][j], 0, i, j))

    # 정렬 이후에 큐로 옮기기 (낮은 번호의 바이러스가 먼저 증식하므로)
    data.sort()
    q = deque(data)

    target_s, target_x, target_y = map(int, input().split())

    # 바이러스가 퍼져나갈 수 있는 4가지의 위치
    dx = [-1, 0, 1, 0]
    dy = [0, 1, 0, -1]

    # 너비 우선 탐색(BFS) 진행
    while q:
        virus, s, x, y = q.popleft()
        # 정확히 s초가 지나거나, 큐가 빌 때까지 반복
        if s == target_s:
            break
        # 현재 노드에서 주변 4가지 위치를 각각 확인
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            # 해당 위치로 이동할 수 있는 경우
            if 0 <= nx and nx < n and 0 <= ny and ny < n:
                # 아직 방문하지 않은 위치라면, 그 위치에 바이러스 넣기
                if graph[nx][ny] == 0:
                    graph[nx][ny] = virus
                    q.append((virus, s + 1, nx, ny))

    print(graph[target_x - 1][target_y - 1])


    x_list = [-1,1,0,0]
    y_list = [0,0,-1,1]

    def dfs(x, y, visited, count):
        visited[x][y] = True
        for i in range(4):
            nx = x+x_list[i]
            ny = y+ y_list[i]
            if 0 <= nx < n and 0<= ny <m and visited[nx][ny] == False:
                if grid[nx][ny] == 0:
                    grid[nx][ny] = grid[x][y]
                    visited[nx][ny] = True
                else:
                    dfs(nx, ny, visited, count)

    # while all(visited) != True:
    for _ in range(S):
        visited = [[False] * m for _ in range(n)]  # [[False] * m] for _ in range(n)]
        for i in range(n):
            for j in range(m):
                if visited[i][j] == False and grid[i][j]!=0:
                    dfs(i,j,visited, 0)

    return grid[x-1][y-1]

#괄호변환
def bracket(b):

    def checkRight(b):
        if b.startswith(")") :
            return False
        #stck
        stck = []
        for bs in b:
            if bs == "(":
                stck.append(bs)
            else:
                if stck[-1] != "(":
                    return False
        return True

    def balanced_index(b):
        stck = []
        bstck = []
        for i, bs in enumerate(b):
            if bs == "(":
                stck.append(bs)
            else:
                bstck.append(bs)
            if len(stck) == len(bstck):
                return i

    nswer = ""
    def bcket(b):
        answer = ''
        if b == '':
            return answer
        index = balanced_index(b)
        u = b[:index+1]
        w = b[index+1:]

        if checkRight(u):
            answer = u + bcket(w)

        else:

            answer = "("
            answer += bcket(w)
            answer += ")"
            u = list(u[1:-1])
            for i in range(len(u)):
                if u[i] == "(":
                    u[i] = ")"
                else:
                    u[i] = "("
            answer += "".join(u)
        return answer

    return bcket(b)

    #균형

    u = ""
    def checkbalanced(b):
        stck = []
        bstck = []
        for i, bs in enumerate(b):
            print(i, bs)
            if bs == "(":
                stck.append(bs)
            else:
                bstck.append(bs)
            if len(stck) == len(bstck):
                u = b[:i+1]
                w = b[i+1:]
                return u, w
    def checkRight(b):
        #stck
        stck = []
        for bs in b:
            if bs == "(":
                stck.append(bs)
            else:
                if stck[-1] != "(":
                    return True
    if checkRight(b) == False:
        for i in range(len(b)):
            if i ==0:
                nswer = "("
            elif b[i] == "(":
                nswer += ")"
            elif b[i] == ")":
                nswer += "("
            elif b[i] == len(b):
                nswer += ")"

    # print(u, v)
    u, v = checkbalanced(b)
    u2, v2 = checkbalanced(v)
    u3, v3 = checkbalanced(v2)

    print(u, v)
    print(u2, v2)
    print(u3, v3)


#연산자끼워넣기
maxx = -10000000
minn = 10000000
def pushdfs(data, p, m, t, d):
    # ls = []
    # for _ in range(p):
    #     ls.append("+")
    # for _ in range(m):
    #     ls.append("-")
    # for _ in range(t):
    #     ls.append("*")
    # for _ in range(d):
    #     ls.append("//")
    # #def dfs ():
    # print(list(itertools.permutations(ls)))

    results = []
    prev_elements = []
    # def dfs(ls):
    #
    #     if len(ls) == 0:
    #         results.append(prev_elements[:])
    #     for e in ls:
    #         next_elements = ls[:]
    #         next_elements.remove(e)
    #
    #         prev_elements.append(e)
    #         dfs(next_elements)
    #         prev_elements.pop()
    # dfs(ls)


    n = len(data)
    def dfs(i, ls, p, m, t, d):
        global minn, maxx
        if i == n:
            minn = min(minn, ls)
            maxx = max(maxx, ls)
            print(minn, maxx)
        if p > 0:
            p -= 1
            dfs(i+1, ls + data[i], p,m, t,d)
            p += 1
        if m>0:
            m -= 1
            dfs(i+1, ls - data[i],p,m, t,d)
            m += 1
        if t>0:
            t -= 1
            dfs(i+1, ls * data[i], p,m, t,d)
            t += 1
        if d>0:
            d -= 1
            dfs(i+1, int(ls / data[i]), p,m, t,d)
            d += 1
    dfs(1, data[0], p,m, t,d)
    print(minn, maxx)

#감시피하기
# 특정 방향으로 감시를 진행 (학생 발견: True, 학생 미발견: False)
def void_techer(tx, ty, grid):
    x_list = [0,0,-1,1]
    y_list = [-1,1,0,0]

    for j in range(4):
        for i in range(1, len(grid)):
            nx = tx+x_list[j]*i
            ny = ty+y_list[j]*i
            if 0 <= nx < len(grid) and 0<= ny <len(grid) :
                if grid[nx][ny] == 'S':
                    return True
                if grid[nx][ny] == "O":
                    break
    return False

def student_teacher(n, grid):
    txy = []
    bxy = []
    for i in range(n):
        for j in range(n):
            if grid[i][j] == 'T':
                txy.append([i, j])
            elif grid[i][j] == 'X':
                bxy.append([i, j])
   #list(itertools.combinations(bxy , 3))

    def process():
        for tx, ty in txy:
            if void_techer(tx, ty, grid):
                return True
            return False
    temp = []
    find = False # 학생이 한 명도 감지되지 않도록 설치할 수 있는지의 여부

    for data in itertools.combinations(bxy , 3): #
        for x, y in data:
            grid[x][y] = 'O'
        if not process():
            find = True
            break
        for x, y in data:
            grid[x][y] = 'X'



    # for tx, ty in txy:
    #     voidtecher(tx, ty, grid)

    # def dfs(count):
    #
    #     if count==3:
    #         # 장애물 설치 이후에, 한 명이라도 학생이 감지되는지 검사
    #         if not process():
    #             find = True
    #             break
    #             # 원하는 경우를 발견한 것임
    #         return
    #
    #     for i in range(n):
    #         for j in range(n):
    #             if grid[i][j] == 'X':
    #                 grid[i][j] = 'O'
    #                 count += 1
    #                 dfs(count)
    #                 count -= 1
    #                 grid[i][j] = 'X'
    # 원하는 경우를 발견한 것임

    # count = 0
    # for i in range(n):
    #     for j in range(n):
    #         if grid[i][j] == 'X':
    #             grid[i][j] = 'O'
    #             count += 1
    #             print(count)
    #             if count == 3:
    #                 print(grid)
    #                 # 장애물 설치 이후에, 한 명이라도 학생이 감지되는지 검사
    #                 if not process():
    #                     print("00000000000")
    #                     find = True
    #                     break
    #             count -= 1
    #             grid[i][j] = 'X'

    if find:
        return "YES"
    return "NO"

def population(N, L, R, plist): #연합 , BFS
    graph = plist

    x_list = [1, 0, -1, 0]
    y_list = [0,1, 0, -1]

    def union(x, y, index):
        visited[x][y] = True
        union = graph[x][y]
        unionxy = [[x, y]]
        count = 1

        # 너비 우선 탐색 (BFS)을 위한 큐 라이브러리 사용
        q = collections.deque()
        q.append((x, y))

        while q:
            x, y = q.popleft()

            for i in range(4):
                nx = x + x_list[i]
                ny = y + y_list[i]
                if 0 <= nx < N and 0 <= ny < N and visited[nx][ny] == False:
                    if L <=  abs(graph[x][y] - graph[nx][ny]) <= R:
                        q.append((nx, ny))

                        union += graph[nx][ny]
                        unionxy.append([nx, ny])
                        count += 1
                        visited[nx][ny] = True

        #인구분배
        for i, j in unionxy:
            graph[i][j] = union//count

    total_count = 0

    while True:
        visited = [[False] * N for _ in range(N)]
        index = 0
        for i in range(N):
            for j in range(N):
                if visited[i][j] == False:
                    union(i, j, index)
                    index += 1
        if index == N*N:
            break #모든 인구 이동이 종료
        total_count += 1
    return total_count

    # union = list
    # unionxy = []
    # x_list = [1,0]
    # y_list = [0,1]
    # for i in range(N-1):
    #     for j in range(N-1):
    #         if L <= abs(plist[i][j] - plist[i][j+1]) <= R:
    #             union.append(plist[i][j])
    #             union.append(plist[i][j+1])
    #             unionxy.append([i, j])
    #             unionxy.append([i, j+1])
    #         if L <= abs(plist[i+1][j] - plist[i][j]) <= R:
    #             union.append(plist[i][j])
    #             union.append(plist[i][j+1])
    #             unionxy.append([i, j])
    #             unionxy.append([i, j+1])
    # print(union)
    # print(unionxy)

    #O(2N*N)



#블록이동하기
def robot(board): #2
    def get_new_pos(pos, board):
        dx = [0,0,1,-1]
        dy = [-1,1,0,0]
        new_pos = []
        pos = list(pos)

        pos1_x, pos1_y, pos2_x, pos2_y = pos[0][0], pos[0][1], pos[1][0], pos[1][1]

        for i in range(4):
            n1x, n1y, n2x, n2y = pos1_x + dx[i], pos1_y + dy[i], pos2_x + dx[i], pos2_y + dy[i]
            if board[n1x][n1y] == 0 and board[n2x][n2y] == 0:
                new_pos.append({(n1x, n1y), (n2x, n2y)})

        #가로인 경우
        if pos1_x == pos2_x:
            #위쪽, 아래쪽으로 회전
            for i in [-1, 1]:
                if board[pos1_x+i][pos1_y] == 0 and board[pos2_x+i][pos2_y] == 0: # 위쪽 혹은 아래쪽 두 칸이 모두 비어 있다면
                    new_pos.append({(pos1_x, pos1_y), (pos1_x+i, pos1_y)})
                    new_pos.append({(pos2_x, pos2_y), (pos2_x+i, pos2_y)})
        #세로인 경우
        elif pos1_y == pos2_y:
            #오른쪽, 왼쪽으로 회전
            for i in [-1, 1]:
                if board[pos1_x][pos1_y+i]==0 and board[pos2_x][pos2_y+i] == 0: # 왼쪽 혹은 오른쪽 두 칸이 모두 비어 있다면
                    new_pos.append({(pos1_x, pos1_y), (pos1_x, pos1_y+i)})
                    new_pos.append({(pos2_x, pos2_y), (pos2_x, pos2_y+i)})
        return new_pos



    n = len(board)
    new_board = [[1] * (n+2) for _ in range(n+2)]
    for i in range(n):
        for j in range(n):
            new_board[i+1][j+1] = board[i][j]
    Q = []
    pos = {(1, 1), (1, 2)}
    visited = []
    heapq.heappush(Q, (0, pos))
    visited.append(pos)

    while Q:
        cost, pos = heapq.heappop(Q)

        if (n, n) in pos:
            return cost
        for next_pos in get_new_pos(pos, new_board):
            if next_pos not in visited:
                heapq.heappush(Q, (cost + 1, next_pos))
                visited.append(next_pos)
    return

    #
    # def get_next_pos(pos, board):
    #     next_pos = []
    #     pos = list(pos)
    #     pos1_x, pos1_y, pos2_x, pos2_y = pos[0][0], pos[0][1], pos[1][0], pos[1][1]
    #     # (상, 하, 좌, 우)로 이동하는 경우에 대해서 처리
    #     dx = [-1, 1, 0, 0]
    #     dy = [0, 0, -1, 1]
    #     for i in range(4):
    #         pos1_next_x, pos1_next_y, pos2_next_x, pos2_next_y = pos1_x + dx[i], pos1_y + dy[i], pos2_x + dx[i], pos2_y + dy[i]
    #         # 이동하고자 하는 두 칸이 모두 비어 있다면
    #         if board[pos1_next_x][pos1_next_y] == 0 and board[pos2_next_x][pos2_next_y] == 0:
    #             next_pos.append({(pos1_next_x, pos1_next_y), (pos2_next_x, pos2_next_y)})
    #     if pos1_x == pos2_x: #가로인 경우
    #         for i in [-1, 1]:  # 위쪽으로 회전하거나, 아래쪽으로 회전
    #             if board[pos1_x + i][pos1_y] == 0 and board[pos2_x + i][pos2_y] == 0:  # 위쪽 혹은 아래쪽 두 칸이 모두 비어 있다면
    #                 next_pos.append({(pos1_x, pos1_y), (pos1_x + i, pos1_y)})
    #                 next_pos.append({(pos2_x, pos2_y), (pos2_x + i, pos2_y)})
    #         # 현재 로봇이 세로로 놓여 있는 경우
    #     elif pos1_y == pos2_y:
    #         for i in [-1, 1]:  # 왼쪽으로 회전하거나, 오른쪽으로 회전
    #             if board[pos1_x][pos1_y + i] == 0 and board[pos2_x][pos2_y + i] == 0:  # 왼쪽 혹은 오른쪽 두 칸이 모두 비어 있다면
    #                 next_pos.append({(pos1_x, pos1_y), (pos1_x, pos1_y + i)})
    #                 next_pos.append({(pos2_x, pos2_y), (pos2_x, pos2_y + i)})
    #         # 현재 위치에서 이동할 수 있는 위치를 반환
    #     return next_pos
    #
    # n = len(board)
    # # 맵의 외곽에 벽을 두는 형태로 맵 변형
    # new_board = [[1] * (n + 2) for _ in range(n + 2)]
    # for i in range(n):
    #     for j in range(n):
    #         new_board[i + 1][j + 1] = board[i][j]
    # # 너비 우선 탐색(BFS) 수행 - 최단경로
    # q = collections.deque()
    # visited = []
    # pos = {(1, 1), (1, 2)}  # 시작 위치 설정
    # q.append((pos, 0))  # 큐에 삽입한 뒤에
    # visited.append(pos)  # 방문 처리
    #
    # while q:
    #     pos, cost = q.popleft()
    #     if (n, n) in pos:
    #         return cost
    #     for next_pos in get_next_pos(pos, new_board):
    #         visited.append(next_pos)
    #         q.append((next_pos, cost+1))


def floydvo(n, flist):
    INF = int(1e9)
    dist = [[INF for _ in range(n+1)] for _ in range(n+1)]
    for x, y, d in flist:
        if d < dist[x][y]:
            dist[x][y] = d
    for i in range(n):
        dist[i+1][i+1]=0
    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                dist[i][j] = min(dist[i][j], dist[i][k]+dist[k][j])

    # 수행된 결과를 출력
    for a in range(1, n + 1):
        for b in range(1, n + 1):
            # 도달할 수 없는 경우, 0을 출력
            if dist[a][b] == INF:
                print(0, end=" ")
            # 도달할 수 있는 경우 거리를 출력
            else:
                print(dist[a][b], end=" ")
        print()

# 정확한순위
def ranking(n, rlist):
    #노드가 다른 노드와 모두 연결되어 있다면
    INF = int(1e9)
    graph = [[INF] * (n+1) for _ in range(n+1)]
    for i in range(n):
        graph[i+1][i+1] = 0
    for x, y in rlist:
        graph[x][y] = 1

    for k in range(1, n+1):
        for i in range(1, n+1):
            for j in range(1, n+1):
                graph[i][j] = min(graph[i][j], graph[i][k] + graph[k][j])
    # 점화식에 따라 플로이드 워셜 알고리즘을 수행

    result = 0
    # 각 학생을 번호에 따라 한 명씩 확인하며 도달 가능한지 체크
    for i in range(1, n + 1):
        count = 0
        for j in range(1, n + 1):
            if graph[i][j] != INF or graph[j][i] != INF:
                count += 1
        if count == n:
            print(i, j)
            result += 1
    print(result)
    #
    count = 0
    print(graph)
    for p in range(1, n+1):
        FIND = True
        for q in range(1, n+1):
            if FIND != True:
                break
            if graph[p][q] >= INF and graph[q][p] >= INF:
                FIND = False
        print(FIND)
        if FIND:
            count += 1
    return count

# 화성탐사
def mars_exploration(n, mlist):
    x_list = [-1,1,0,0]
    y_list = [0,0,1,-1]

    INF = int(1e9)
    distance = [[INF] * n for _ in range(n)]
    distance[0][0] = mlist[0][0]

    Q = []
    cost = mlist[0][0]
    heapq.heappush(Q, (cost, (0, 0)))
    while Q:
        cost, (x, y) = heapq.heappop(Q)

        if distance[x][y] < cost:
            continue

        if x == n-1 and y == n-1:
            return cost
        for i in range(4):
            nx = x + x_list[i]
            ny = y + y_list[i]
            if 0<= nx < n and 0 <= ny < n:
                if cost + mlist[nx][ny] < distance[nx][ny]:
                    distance[nx][ny] = cost + mlist[nx][ny]
                    heapq.heappush(Q, (cost+mlist[nx][ny], (nx, ny)))

# 숨바꼭질
def hideandseek(n, hlist):

    #최단거리가 가장 먼...

    INF = int(1e9)
    distance = [INF] * (n+1)

    graph = collections.defaultdict(list)
    for x, y in hlist:
        graph[x].append(y)
        graph[y].append(x)
    print(graph)

    distance[1] = 0
    Q = []
    heapq.heappush(Q, (0, 1))

    while Q:
        dist, now = heapq.heappop(Q)

        if distance[now] < dist:
            continue
        for neighbor in graph[now]:
                cost = dist + 1
                if distance[neighbor] > cost:
                    distance[neighbor] = cost
                    heapq.heappush(Q, (cost, neighbor))

    # 가장 최단 거리가 먼 노드 번호(동빈이가 숨을 헛간의 번호)
    max_node = 0
    # 도달할 수 있는 노드 중에서, 가장 최단 거리가 먼 노드와의 최단 거리
    max_distance = 0
    # 가장 최단 거리가 먼 노드와의 최단 거리와 동일한 최단 거리를 가지는 노드들의 리스트
    result = []

    for i in range(1, n + 1):
        if max_distance < distance[i]:
            max_node = i
            max_distance = distance[i]
            result = [max_node]
        elif max_distance == distance[i]:
            result.append(i)
    print(max_node, max_distance, len(result))


def find_parent(parents, x):
    if parents[x] != x:
        parents[x] = find_parent(parents, parents[x] )
    return parents[x]

def union_parent(parents, a, b):
    a = find_parent(parents, a)
    b = find_parent(parents, b)
    if a < b:
        parents[b] = a
    else:
        parents[a] = b

# 여행계획
def travel_plan(n, tlist, itinerary):
    parent = [0] * (n+1)
    for i in range(1, n+1):
        parent[i] = i
    #여행이 가능한지  - > floyd? kruskal?
    #kruskal
    for i in range(n):
        for j in range(n):
            if tlist[i][j] == 1:
                union_parent(parent, i+1, j+1)

    result = True
    for i in range(len(itinerary)-1):
        if find_parent(parent, itinerary[i])  != find_parent(parent, itinerary[i+1]):
            result = False
    if result:
        print("YES")
    else:
        print("NO")

# 탑승구
def gate(G, P, ilist):
    parents = [0] * (G+1)
    for i in range(G+1):
        parents[i] = i
    count = 0
    for i in ilist:
        if find_parent(parents, i) == 0:
            break
        union_parent(parents, i, i-1)
        count += 1
    return count



# 어두운길
def dark(N, roads):
    #최소신장트리 - kruskal
    road = []
    su = 0
    for x, y, d in roads:
        road.append([d, x, y])
        su += d

    road.sort()
    print(road)

    parent = [0] * (N+1)
    for i in range(N+1):
        parent[i] = i
    #
    ds = 0
    for d, x, y in road:
        if find_parent(parent, x) != find_parent(parent, y):

            union_parent(parent, x, y)
            ds += d
    print(ds)
    print(su)
    print(su - ds)

# 행성터널
def planet_turnel(n, planet):
    parent = [0] * n
    for i in range(n):
        parent[i] = i
    edges = []

    for i in range(n):
        for j in range(n):
            d  = min(abs(planet[i][0] - planet[j][0]), abs(planet[i][1] - planet[j][1]), abs(planet[i][2] - planet[j][2]))
            edges.append([d, i, j])
    edges.sort()
    result = 0
    for d, i, j in edges:
        if find_parent(parent, i) != find_parent(parent, j):
            union_parent(parent, i, j)
            result += d
    print(result)

# 최종순위
def final_ranking(T, past_year, c_pairs):
    graph = [[0] * (T+1) for _ in range(T+1)]
    indegree = [0] * (T + 1)

    #작년
    for i, p in enumerate(past_year):
        for i in range(i+1, T):
            graph[p][past_year[i]] = 1
            indegree[past_year[i]] += 1

    #바뀜
    for x, y in c_pairs:
        if graph[y][x] == 1:
            graph[y][x] = 0
            graph[x][y] = 1
            indegree[y] += 1
            indegree[x] -= 1
        else:
            graph[y][x] = 1
            graph[x][y] = 0
            indegree[y] -= 1
            indegree[x] += 1


    #topological sort
    #graph 에 진입차수가 0 인것
    q = collections.deque()
    result = []
    for i in range(1, T+1):
        if indegree[i] == 0:
            q.append(i)

    certain = True #위상정렬결과가 오직 하나인지 여부
    cycle = False # 그래프 내 사이클이 존재하는지 여부

    for i in range(T):
        if len(q) == 0:
            cycle = True
            break
        if len(q) > 2:
            certain = False
            break
        now = q.popleft()
        result.append(now)

        for j in range(1, T+1):
            if graph[now][j] == 1:
                indegree[j] -= 1
                if indegree[j] == 0:
                    q.append(j)

    if cycle:
        print("IMPOSSIBLE")
    elif not certain:
        print("?")
    else:
        for i in result:
            print(i, end = " ")
        print()

    #exception queue 에 없는 경우,순환 , stack 에 두개 이상 들어오는 경우



# 아기상어
def baby_shark(sea):
    INF = 1e9  # 무한을 의미하는 값으로 10억을 설정
    for a in range(len(sea)):
        for b in range(len(sea[0])):
            if sea[a][b] == 9:
                sea[a][b] = 0
                now_x, now_y = a, b
                break

    now_size = 2
    dx = [0,-1,1,0]
    dy = [1,0,0,-1]
    n = len(sea)
    def bfs(): #현재위치에서 모든 위치까지 최단 거리 테이블 반환
        visited = [[False] * (len(sea[0])) for _ in range(len(sea))]
        distance = [[-1] * (len(sea[0])) for _ in range(len(sea))]
        distance[now_x][now_y] = 0
        visited[now_x][now_y] = True
        Q = []
        heapq.heappush(Q, (0, now_x, now_y))
        while Q:
            dist, x, y = heapq.heappop(Q)

            for i in range(4):
                nx = x + dx[i]
                ny = y + dy[i]

                if 0 <= nx < n and 0 <= ny < n and sea[nx][ny] <= now_size and visited[nx][ny] == False:
                    #visited
                    visited[nx][ny] = True
                    distance[nx][ny] = dist + 1
                    heapq.heappush(Q, (dist+1 ,nx, ny))
        return distance

    #물고기 찾기
    def find(distance):
        min_dist = INF
        x, y = 0, 0
        for a in range(n):
            for b in range(n):
                if distance[a][b] != -1 and 1 <= sea[a][b] < now_size:
                    if distance[a][b] < min_dist:
                        x, y = a, b
                        min_dist = distance[a][b]
        if min_dist == INF:
            return None
        else:
            return x, y, min_dist

    result = 0
    ate = 0

    while True:
        value = find(bfs())
        if value == None:
            break
        else:

            now_x, now_y = value[0], value[1]

            result += value[2]
            sea[now_x][now_y] = 0
            ate += 1
            if ate >= now_size:
                now_size+=1
                ate = 0




# 청소년상어
result = 0
def adolescent_shark(grid):
    dx = [-1,-1,0,1, 1, 1, 0, -1]
    dy = [0,-1, -1,-1, 0 , 1,1,1]
    #위 i+1

    direction = grid[0][1]
    sx, sy = 0, 0

    #1 )
    sea = [[0] * (4) for _ in range(4)]
    fish = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    # total = grid[0][0]
    # fish.remove(grid[0][0])
    #
    # grid[0][0] = "상어"

    dq = collections.defaultdict(list)
    for j, fish_d in enumerate(grid):
        for i in range(4):
            if fish_d[2 * i] != 0:
                dq[fish_d[2 * i]].append(fish_d[2 * i + 1])
                dq[fish_d[2 * i]].append(j)
                dq[fish_d[2 * i]].append(i)  # 물고기 번호, 물고기 방향 -1 , 위치 저장
                sea[j][i] = fish_d[2 * i]
    dq['상어'].append(0)
    dq['상어'].append(0)
    dq['상어'].append(0)

    def fish_moving(dq, sea, fish):
        print("---------------------------------------------")
        #물고기 이동
        #1) 번호가 작은 물고기
        print("dq---", dq)
        print("sea---", sea)
        #작은 수 부터 이동
        for f in fish:
           td, tx, ty  = dq[f]
           p =0
           while True:
               p += 1
               if p ==8:
                   print()
                   break
               nx =  tx + dx[td-1]
               ny =  ty + dy[td-1]

               if 0<= nx < 4 and 0 <= ny < 4 and sea[nx][ny] != "상어":
                   sea[tx][ty] = sea[nx][ny]
                   sea[nx][ny] = f
                   dq[f] = [td,nx, ny]

                   if sea[tx][ty] != 0:
                       dq[sea[tx][ty]][1] = tx
                       dq[sea[tx][ty]][2] = ty
                   break
               else:
                   td += 1
                   if td == 9:
                        td = 1

        return dq, sea, fish

    # 상어가 현재 위치에서 먹을 수 있는 모든 물고기의 위치 반환

    def shark_moving(dq, result):
        positions = []

        s_max = -1

        direction, sx, sy = dq['상어']
        mx = -1
        n_sx = sx
        n_sy = sy
        for i in range(3):
            n_sx += dx[direction-1] #+ i*1 if dx[direction-1] > 0 else sx + dx[direction-1] - i*1
            n_sy += dy[direction-1] #+ i*1 if dy[direction-1] > 0 else sy + dy[direction-1] - i*1
            #최댓값 아니 dfs
            if 0 <= n_sx < 4 and 0 <= n_sy < 4:
                if sea[n_sx][n_sy] > 0:
                    positions.append([n_sx, n_sy])
                    # mx, my = n_sx, n_sy
                    # s_max = sea[n_sx][n_sy]
        return positions

    def dfs(dq, fish,sea, now_x, now_y, total):
        print("시작-------", fish, sea, now_x, now_y, total,"&***********************")
        global result
        sea = copy.deepcopy(sea)
        fish = copy.deepcopy(fish)
        dq = copy.deepcopy(dq)
        if sea[now_x][now_y] == 0:
            return
        total += sea[now_x][now_y]
        direction = dq[sea[now_x][now_y]][0]
        fish.remove(sea[now_x][now_y])
        _, sx, sy = dq['상어']



        dq["상어"][0] = direction
        dq["상어"][1] = now_x
        dq["상어"][2] = now_y
        sea[sx][sy] = 0
        sea[now_x][now_y] = "상어"

        dq, sea, fish = fish_moving(dq, sea, fish)
        positions = shark_moving(dq, result)
        print("positions",positions)

        #물고기 먹고
        if len(positions) == 0:
            print("total", total)
            result = max(result, total)
            print(result)
            return
        for next_x, next_y in positions:
            dfs(dq, fish, sea, next_x, next_y, total)
        # result += sea[mx][my]
        #
        # #방향
        # direction = dq[sea[mx][my]][0]
        #
        # fish.remove(sea[mx][my])
        # dq["상어"][0] = direction
        # dq["상어"][1] = mx
        # dq["상어"][2] = my
        # sea[sx][sy] = 0
        # sea[mx][my] = "상어"
        # #del dq[sea[mx][my]]
        # return True, dq, sea, fish, result

    # 청소년 상어의 시작 위치(0, 0)에서부터 재귀적으로 모든 경우 탐색
    dfs(dq, fish, sea, 0, 0, 0)
    print("ttttttttt",result)
    # bool = True
    # while bool:
    #     dq, sea, fish, result = fish_moving()
    #     bool, dq, sea, fish, result = shark_moving(dq, result)
    # return result





        #이동할수 없으면 종료

from collections import deque
#내가 놓친 것 1) 우선순위 방향 (빈 칸, 자기 자신 칸도!) 2) 향기가 사라질수도 더해질수도 3) 바로전만 아니면 이동 못함
def adult_shark(N, M, k, grid, shark_direction, shark_priority):
    count = 0 #몇 초 후
    shark_list = [i+1 for i in range(M)] #상어 리스트

    #scent 상어 냄새 위치
    S = [[[0, 0]] * (N) for _ in range(N)]
    #grid 상어 위치

    dq = collections.defaultdict(deque) #상어 정보
    s_klist = collections.defaultdict(list)  # 향기리스트
    for i in range(N):
        for j in range(N):
            for p in range(1,M+1):
                if grid[i][j] == p:
                    dq[p].append([shark_direction[p-1],i, j]) #방향 위치 i j
                    # S[i][j] = [p, k+1]
                    s_klist[p].append([i, j])


    #if count >= k:  # update smell #한번가기도 함...

        # keys = sorted(s_klist.keys(), reverse=True)
        # for sk in keys:
        #     if s_klist[sk]:
        #         for i in range(1, len(s_klist[sk])):
        #             dex, dey = s_klist[sk][i]
        #             S[dex][dey] = sk
        #         s_klist[sk] = s_klist[sk][1:]

    x_list = [-1, 1, 0, 0]
    y_list = [0, 0, -1, 1]
    shark_list.sort(reverse=True)

    while len(shark_list) > 1:
        for i in range(N):
            for j in range(N):

                if S[i][j][1] > 0:
                    S[i][j][1] -= 1

                if grid[i][j] != 0:
                    S[i][j] = [grid[i][j], k]
                # 상어가 존재하는 해당 위치의 냄새를 k로 설정


        f = 0
        for shark in shark_list:
            d = dq[shark][0][0]
            # 우선순위 #방향일 때 우선순위
            x = dq[shark][0][1]
            y = dq[shark][0][2]
            check = True
            for p in shark_priority[shark - 1][d - 1]:
                nx = x + x_list[p - 1]
                ny = y + y_list[p - 1]
                if 0 <= nx < N and 0 <= ny < N and S[nx][ny][1] == 0:
                    if grid[nx][ny] == 0:
                        check = False
                        grid[nx][ny] = shark
                        s_klist[shark].append([nx, ny])
                        break

                #if 0 <= nx < N and 0 <= ny < N and  0 < grid[nx][ny] > shark :
                    else:
                        check = False
                        f = grid[nx][ny]
                        grid[nx][ny] = shark
                        s_klist[shark].append([nx, ny])
                        #shark_list.remove(f)
                        break
            # 있는거
            # 없으ㅕㄴ 되돌아가 -> 방향
            if check:
                for p in shark_priority[shark - 1][d - 1]:
                    nx = x + x_list[p - 1]
                    ny = y + y_list[p - 1]
                    if 0 <= nx < N and 0 <= ny < N and S[nx][ny][0] == shark:
                        grid[nx][ny] = shark
                        s_klist[shark].append([nx, ny])
                        break

                # 현재 상어의 방향
                # 자신의 냄새가 있는 곳이면
                # 해당 상어의 방향 이동시키기
                # !!! 우선순위에 따라

            d, dx, dy = dq[shark].pop()
            grid[dx][dy] = 0
            dq[shark].append([p, nx, ny])
        if f !=0 :
            shark_list.remove(f)

        count += 1

        if count >= 1000:
            print(-1)
            break
    return count

    #방향에 ㅏㅈ춰 이동



def adult_shark2(n, m, k, array, directions , priorities):
    smell = [[[0, 0]] * n for _ in range(n)]

    # 특정 위치에서 이동 가능한 4가지 방향
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]

    # 모든 냄새 정보를 업데이트
    def update_smell():
        # 각 위치를 하나씩 확인하며
        for i in range(n):
            for j in range(n):
                # 냄새가 존재하는 경우, 시간을 1만큼 감소시키기
                if smell[i][j][1] > 0:
                    smell[i][j][1] -= 1
                # 상어가 존재하는 해당 위치의 냄새를 k로 설정
                if array[i][j] != 0:
                    smell[i][j] = [array[i][j], k]

    # 모든 상어를 이동시키는 함수
    def move():
        # 이동 결과를 담기 위한 임시 결과 테이블 초기화
        new_array = [[0] * n for _ in range(n)]
        # 각 위치를 하나씩 확인하며
        for x in range(n):
            for y in range(n):
                # 상어가 존재하는 경우
                if array[x][y] != 0:
                    direction = directions[array[x][y] - 1]  # 현재 상어의 방향
                    found = False
                    # 일단 냄새가 존재하지 않는 곳이 있는지 확인
                    for index in range(4):
                        nx = x + dx[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        ny = y + dy[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        if 0 <= nx and nx < n and 0 <= ny and ny < n:
                            if smell[nx][ny][1] == 0:  # 냄새가 존재하지 않는 곳이면
                                # 해당 상어의 방향 이동시키기
                                directions[array[x][y] - 1] = priorities[array[x][y] - 1][direction - 1][index]
                                # 상어 이동시키기 (만약 이미 다른 상어가 있다면 번호가 낮은 것이 들어가도록)
                                if new_array[nx][ny] == 0:
                                    new_array[nx][ny] = array[x][y]
                                else:
                                    new_array[nx][ny] = min(new_array[nx][ny], array[x][y])
                                found = True
                                break
                    if found:
                        continue
                    # 주변에 모두 냄새가 남아 있다면, 자신의 냄새가 있는 곳으로 이동
                    for index in range(4):
                        nx = x + dx[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        ny = y + dy[priorities[array[x][y] - 1][direction - 1][index] - 1]
                        if 0 <= nx and nx < n and 0 <= ny and ny < n:
                            if smell[nx][ny][0] == array[x][y]:  # 자신의 냄새가 있는 곳이면
                                # 해당 상어의 방향 이동시키기
                                directions[array[x][y] - 1] = priorities[array[x][y] - 1][direction - 1][index]
                                # 상어 이동시키기
                                new_array[nx][ny] = array[x][y]
                                break
        return new_array

    time = 0
    while True:
        update_smell()  # 모든 위치의 냄새를 업데이트
        new_array = move()  # 모든 상어를 이동시키기
        array = new_array  # 맵 업데이트
        time += 1  # 시간 증가
        print("나동빈", time)
        print(smell)
        print(array)
        # 1번 상어만 남았는지 체크
        check = True
        for i in range(n):
            for j in range(n):
                if array[i][j] > 1:
                    check = False
        if check:
            print(time)
            break

        # 1000초가 지날 때까지 끝나지 않았다면
        if time >= 1000:
            print(-1)
            break


#문제은행
#프로그래머스

#백준
#leetcode
#hackerrank

answer = 0
def target_number(numbers, target):
    n = len(numbers)
    def dfs(i, result):
        global answer
        if (i) == n:
            if result == target:
                print("AAAA")
                #nonlocal answer
                answer += 1
            return
        else:
            dfs(i+1, result - numbers[i])
            dfs(i+1, result + numbers[i])

    dfs(0,0)

    return answer



def find_parents(parents, a):
    if parents[a] != a:
        parents[a] = find_parents(parents, parents[a])
    return parents[a]

def union_parents(parents, a, b):
    a = find_parents(parents, a)
    b = find_parents(parents, b)

    if a <= b:
        parents[b] = a
    else:
        parents[a] = b

def network(n, computers):
    parents = [(i) for i in range(n+1)]

    for i in range(n):
        for j in range(i+1, n):
            if computers[i][j] == 1:
                union_parents(parents, i+1, j+1)

    results = set()
    for i in range(1, n+1):
        results.add(find_parents(parents, i))
    return len(results)
    #return len(collections.Counter(parents)) - 1

def word_transition(begin, target, words):
    def dfs(begin, index, words):
        global answer
        print("--", begin, index, words)
        if begin == target:
            print("wjd",index)
            answer = index
            return
        if len(words) == 0:
            return

        for p, word in enumerate(words):
            for i in range(len(begin)):
                if begin[:i] + begin[i+1:] == word[:i] + word[i+1:]:
                    n_words = words[:p] + words[p+1:]
                    dfs(word, index+1, n_words)

    dfs(begin, 0, words)

    return answer

def findItinerary(tickets):
    route = []
    graph = collections.defaultdict(list)
    for x, y in tickets:
        graph[x].append(y)

    for v in graph:
        graph[v].sort(reverse=True)

    def dfs(start):
        route.append(start)
        while graph[start]:
            dfs(graph[start].pop())

    dfs("ICN")

    return route



def tomato(M, N, grid): #토마토가 다 익는 최소 일
    dx = [0,0,-1,1]
    dy = [-1,1,0,0]
    #출발확인

    positions = []
    for i in range(N):
        for j in range(M):
            if grid[i][j] == 1:
                positions.append([i, j])

    def check_grid(grid):
        for i in range(N):
            for j in range(M):
                if grid[i][j] == 0:
                    return False
        return True

#출발 dfs? bfs?
# def dfs(x, y, count):
#         print(count, x, y)
#         #확인
#         if check_grid(grid):
#             print(count)
#             return count
#
#         for i in range(4):
#             nx = x + dx[i]
#             ny = y + dy[i]
#
#             if 0<= nx < N and 0 <= ny <  M and grid[nx][ny] == 0:
#                 grid[nx][ny] = 1
#                 dfs(nx, ny, count+1)
#         return -1
    Q = []
    for x, y in positions:
        heapq.heappush(Q, (0,(x,y)))

    while Q:
        c, (x, y) = heapq.heappop(Q)
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < N and 0 <= ny < M and grid[nx][ny] == 0:
                grid[nx][ny] = 1
                heapq.heappush(Q, (c+1, (nx, ny)))


    if check_grid(grid):
        print(c)
    else:
        print(-1)

#유기농 배추
def organic_(M, N, K, cabbage_positions):
    grid = [[0] * M for _ in range(N)]
    for x, y in cabbage_positions:
        grid[y][x] = 1
    visited = [[False] * M for _ in range(N)]

    def component(x, y, visited):
        visited[x][y] = True
        dx = [-1, 1, 0, 0]
        dy = [0, 0, 1, -1]
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < N and 0 <= ny < M and visited[nx][ny] == False and grid[nx][ny] == 1:
                component(nx, ny, visited)

    #시작
    count = 0
    for x, y in cabbage_positions:
        if visited[y][x] == False:
            component(y, x, visited)
            count += 1

    return count

def dfsbfs(N, M, V, vlist):
    graph = collections.defaultdict(list)
    for s, v in vlist:
        graph[s].append(v)
        graph[v].append(s)

    def dfs():
        visited = [False] * (N + 1)

        for s in graph:
            graph[s] = sorted(graph[s], reverse=True)
        dfs_l = []
        #stack
        q = []
        q.append(V)
        while q:
            v = q.pop()
            if visited[v] == False:
                dfs_l.append(v)
            visited[v] = True
            for n in graph[v]:
                if visited[n] == False:
                    q.append(n)
        return dfs_l

    def bfs():
        visited = [False] * (N + 1)

        for s in graph:
            graph[s].sort()
        #queue
        bfs_l = []
        q = collections.deque()
        q.append(V)
        while q:
            v =  q.popleft()
            if visited[v] == False:
                bfs_l.append(v)
            visited[v] = True
            for n in graph[v]:
                if visited[n] == False:
                    q.append(n)
        return bfs_l

    print(dfs())
    print(bfs())

#바이러스
def virus(N, K, c):
    parents = [i for i in range(N+1)]

    for x, y in c:
        union_parent(parents, x, y)
    print(parents)
    count = 0
    for p in parents:
        if p == 1:
            count += 1

    return count - 1

#단지번호붙이기
def makeNumber(N, grid):

    visited = [[False] * N for _ in range(N)]

    def component(x, y, visited):
        visited[x][y] = True
        dx = [-1, 1, 0, 0]
        dy = [0, 0, 1, -1]
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            if 0 <= nx < N and 0 <= ny < N and visited[nx][ny] == False and grid[nx][ny] == 1:
                nonlocal count
                count += 1
                component(nx, ny, visited)

    # 시작
    answer = 0
    count_list = []
    for i in range(N):
        for j in range(N):
            if visited[i][j] == False and grid[i][j] == 1:
                count = 1
                component(i, j, visited)
                count_list.append(count)
                answer += 1
    count_list.sort()
    print(answer)
    for i in count_list:
        print(i)

    #return answer


def NM(N, M):
    print(itertools.combinations(3, 1))


#하노이

def hanoi(n, start, end):
    if n == 1:
        print(start, end)
        return

    hanoi(n-1, start, 6 - start - end)
    print(start, end)
    hanoi(n-1, 6 - start - end, end)

#통계학
def sort_cordinates(lst):
    lst = sorted(lst, key=lambda x: x[0])
    lst = sorted(lst, key = lambda x:x[1])
    return lst



###정렬
# 버블
def bubblesort(A):
    for i in range(1, len(A)):
        for j in range(0, len(A)-1):
            if A[j] > A[j+1]:
                A[j], A[j+1] = A[j+1], A[j]
    return A


# 병합
def mergesort(A):
    if len(A) == 1:
        return A
    mid = len(A) // 2
    left = mergesort(A[:mid])
    right = mergesort(A[mid:])
    merged = merge(left, right)
    return merged

def merge(list1, list2):
    merged = []
    while len(list1) > 0 and len(list2) > 0:
        if list1[0] < list2[0]:
            merged.append(list1.pop(0))
        else:
            merged.append(list2.pop(0))

    if len(list1) > 0:
        merged += list1
    if len(list2) > 0:
        merged += list2
    return merged


# 큌 #로무토파티션
def quicksort(A, lo, hi):
    def partition(lo, hi):
        pivot = A[hi]
        left = lo
        for right in range(lo, hi):
            if A[right] < pivot:
                A[left], A[right] = A[right], A[left]
                left+=1
        A[left], A[hi] = A[hi], A[left]
        return left
    if lo<hi:
        pivot = partition(lo, hi)
        quicksort(A, lo, pivot-1)
        quicksort(A, pivot+1, hi)
    return A

# 선택, 삽입, 큌, 계수
def selectionsort(A):
    for i in range(len(A)):
        min_index = i
        for j in range(i+1, len(A)):
            if A[min_index] > A[j]:
                min_index = j
        A[i], A[min_index] = A[min_index], A[i]
    return A

def insertionsort(A):
    for i in range(1, len(A)): #2번째부터
        for j in range(i, 0, -1): #아래로
            if A[j] < A[j-1]:
                A[j], A[j-1] = A[j-1], A[j]

            else: #자신보다 작은 데이터만나면 멈춤
                break
    return A

#호어파티션
def quick(A, start, end):
    if start >= end:
        return
    pivot = start
    left = start + 1
    right = end

    while left <= right:
        while left <= end and A[left] <= A[pivot]:
            left += 1 #
        while right > start and A[right] >= A[pivot]:
            right -= 1
        if left > right: #엇갈렸다면
            A[right], A[pivot] = A[pivot], A[right]
        else:
            A[left], A[right] = A[right], A[left]

    quick(A, start, right - 1)
    quick(A, right + 1, end)

    return A

def quick_sort(A):
    if len(A) <= 1:
        return A
    pivot = A[0]
    tail = A[1:]

    left_side = [x for x in tail if x<= pivot]
    right_side = [x for x in tail if x > pivot]

    return quick_sort(left_side) + [pivot] + quick_sort(right_side)

def countsort(A):
    result = []
    count = [0] * (max(A) + 1)
    for i in range(len(A)):
        count[A[i]] += 1

    for i in range(len(count)):
        for j in range(count[i]):
            result.append(i)
    return result

import math
def radixsort(A):
    # find MAX, digit
    max = -99999
    for i in range(len(A)):
        if A[i] > max:
            max = A[i]
    D = int(math.log10(max))

    for i in range(D + 1):
        buckets = []
        # 빈도
        for j in range(0, 10):
            buckets.append([])
        for j in range(len(A)):
            digit = int(A[j] / math.pow(10, i)) % 10
            buckets[digit].append(A[j])
        # printing
        cnt = 0
        for j in range(0, 10):
            for p in range(len(buckets[j])):
                A[cnt] = buckets[j][p]
                cnt = cnt + 1
    return A

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def sortList(head : ListNode):
    #nlogn
    #병합정렬
    if not (head.next and head):
        return head

    #분할  Runner
    half, slow, fast = None, head, head
    while fast and fast.next:
        half, slow, fast = slow, slow.next, fast.next.next
    half.next = None #분할

    left_side = sortList(head)
    right_side = sortList(slow)
    merged = mergelist(left_side, right_side)
    return merged
    p = head
    lst = []
    while p:
        lst.append(p.val)
        p = p.next
    lst.sort()
    #파이썬 리스트 -> 연결리스트

    p = head
    for l in range(len(lst)):
        p.val = lst[l]
        p = p.next
    return head


def mergelist(l1:ListNode, l2:ListNode):
    if l1 and l2:  #-1 5 / 0 3 4
        if l1.val > l2.val:
            l1, l2 = l2, l1 # 0 3 4 / 5
        l1.next = mergelist(l1.next, l2) #-1 0 3 4 5
    return l1 or l2

def mergeIntervals(lst):
       merged = []
       for i in sorted(lst, key = lambda x:x[0]):
           if merged and i[0] <= merged[-1][1]:
               merged[-1][1] = max(i[1], merged[-1][1])
           else:
               merged += i,
       return merged

# # bfs dfs 그래프
# ##문제은행 + 문제
# # DFS와 BFS	토마토
# # DFS와 BFS	유기농 배추
# # DFS와 BFS	DFS와 BFS
# # DFS와 BFS	바이러스
# # DFS와 BFS	단지번호붙이기
# # 백트래킹	N과 M(2)
# # ##programmers
# ##hackerrank (week11)
#def insertionsortlist()
def insertion(head):
    dummy = ListNode(0, head)
    #cur 정렬
    prev, cur = head, head.next

    while cur:
        if prev.val <= cur.val:
            prev, cur = cur, cur.next
            continue

        #처으부터?
        tmp = dummy
        while tmp.next.val <= cur.val:
            tmp = tmp.next #insertion 삽입 위치 확인


        prev.next = cur.next
        cur.next = tmp.next
        tmp.next = cur

        cur = prev.next
    return dummy.next

def largestNumber(nums):
    for i , n in enumerate(nums):
        nums[i] = str(n)
    #큰 수대로 정렬... nlogn?
    def to_swap(n1:str, n2:str):
        if n1 + n2 > n2 + n1:
            return -1
        else:
            return 1

    nums = sorted(nums, key=functools.cmp_to_key(to_swap))
    return "".join(nums)

def valid_anagram(s, t):
    return sorted(s) == sorted(t)

def sort_colors(colors): #quick partition
    #just modify colors
    l, r = 0, len(colors) - 1
    i = 0
    def swap(i, j):
        temp = colors[i]
        colors[i] = colors[j]
        colors[j] = temp

    while i <= r:
        print(i)
        if colors[i] == 0:
            swap(l, i)
            l += 1
        elif colors[i] == 2:
            swap(i, r)
            r -= 1
            i -= 1
        i += 1
    return colors

def k_closets_points_to_origin(points, K):
    Q = []
    for x, y in points:
        heapq.heappush(Q, (abs(x)**2+abs(y)**2, (x, y)))
    results = []
    for _ in range(K):
        d, (x, y) = heapq.heappop(Q)
        results.append([x, y])
    return results

#이코테
def sort_array(array):
    array.sort()
    return array

def grade_sort(array):
    array = sorted(array, key= lambda x:x[1])
    return array

def change_element(N, K, l1, l2):
    l1.sort()
    l2.sort(reverse=True)

    for i in range(K):
        if l1[i] < l2[i]:
            l1[i], l2[i] = l2[i], l1[i]
        else:
            break
    return sum(l1)

def subject_sort(students):
    students.sort(key=lambda x: ( -int(x[1]), int(x[2]), -int(x[3]), x[0]))
    return students

def anthena(n, alist):
    alist.sort()
    return alist[(n-1)//2]

def failure_rate(N, stages):
    answer = []
    length = len(stages)

    # 스테이지 번호를 1부터 N까지 증가시키며
    for i in range(1, N + 1):
        # 해당 스테이지에 머물러 있는 사람의 수 계산
        count = stages.count(i)

        # 실패율 계산
        if length == 0:
            fail = 0
        else:
            fail = count / length

        # 리스트에 (스테이지 번호, 실패율) 원소 삽입
        answer.append((i, fail))
        length -= count

    # 실패율을 기준으로 각 스테이지를 내림차순 정렬
    answer = sorted(answer, key=lambda t: t[1], reverse=True)

    # 정렬된 스테이지 번호 반환
    answer = [i[0] for i in answer]
    return answer

def card(n, data):
    import heapq

    #n = int(input())

    # 힙(Heap)에 초기 카드 묶음을 모두 삽입
    heap = []
    for i in range(n):
        #data = int(input())
        heapq.heappush(heap, data[i])

    result = 0

    # 힙(Heap)에 원소가 1개 남을 때까지
    while len(heap) != 1:
        # 가장 작은 2개의 카드 묶음 꺼내기
        one = heapq.heappop(heap)
        two = heapq.heappop(heap)
        # 카드 묶음을 합쳐서 다시 삽입
        sum_value = one + two
        print(sum_value, one, two)
        result += sum_value
        heapq.heappush(heap, sum_value)

    print(result)

#heapsort






#분할 정복
# # 비트
# # 그리디
# # 구현
# # 최대공약수
#
# 3부분 정보
# optimal substructure
class division_and_conquer:
    #bruteforce
    def majority_element(self, nums):
        for num in nums:
            if nums.count(num) > len(nums) // 2:
                return num
    #dp
    def ma(self, nums):
        dp = collections.defaultdict(int)
        for num in nums:
            if dp[num] == 0:
                dp[num] = nums.count(num)
            if dp[num] > len(nums)//2:
                return num

    def majorityElement(self, nums):

        #if 간단, 정복
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]

        #else 분할 조합
        half = len(nums) // 2
        a = self.majorityElement(nums[:len(nums)//2])
        b = self.majorityElement(nums[len(nums)//2:])
        return [b, a][nums.count(a) > len(nums)//2] #True 1 False 0
        #재귀의 특성상 dp보다 느림

    def mae(self, nums):
        return sorted(nums)[len(nums)//2]
#

    def diffWaysToCompute(self, input:str):
        def compute(left, right, value):
            results = []

            for l in left:
                for r in right:
                    results.append(eval(str(l) + value + str(r)))

            return results

        #분할
        if input.isdigit():
            return [int(input)]

        #정복
        result = []
        for index, value in enumerate(input):
            if value in "-+*":
                left = self.diffWaysToCompute(input[:index])
                right = self.diffWaysToCompute(input[index+1:])

                result.extend(compute(left, right, value))

        return result


#슬라이딩윈도우
class SlidingWindow:
    def sw(self,nums, k):
        if not nums:
            return nums
        r = []
        for i in range(len(nums)-k+1):
            r.append(max(nums[i:i+k]))
        return r

    def msw(self, nums, k):
        results = []
        window = collections.deque()
        current_max = float('-inf')
        for i, v in enumerate(nums):
            window.append(v)
            if i < k -1:
                continue

            if current_max == float('-inf'):
                current_max = max(window)
            elif v > current_max:
                current_max = v

            results.append(current_max)

            #최댓값이 윈도우에서 빠지면
            if current_max == window.popleft():
                current_max = float('-inf')

        return results

    def minWindow(self, s:str, t:str):
        def contains(s_substr, t_list):
            for t_elem in t_list:
                if t_elem in s_substr:
                    s_substr.remove(t_elem)
                else:
                    return False
            return True

        if not s or not t:
            return ""

        window_size = len(t)
        for size in range(window_size, len(s) + 1): #크기를 키우면서 확인
            for left in range(len(s)-size+1):
                s_substr = s[left:left+size]
                if contains(list(s_substr), list(t)):
                    return s_substr
        return ''
            #
            # results.append(current_max)
            # if current_max == window.popleft():

    #투포인터 슬라이딩윈도우
    def twopoint_slider(self, s, r):
        # 0 False 1 True
        need = collections.Counter(r)
        missing = len(r)

        left = start = end = 0
        for right, char in enumerate(s, 1): #점점 키우기 오른쪽 포인터 이동
            missing -= need[char] > 0
            need[char] -= 1
            #필요문자가 0 이면 ADOBEC
            if missing == 0: #왼쪽 포인터 이동
                while left<right and need[s[left]] < 0: #필요없는 문자
                    need[s[left]] += 1
                    left += 1

                if not end or right - left <= end - start: # start 0 end 6 left 0 right 6 left +1 missing =+1
                    start, end = left, right
                    need[s[left]] += 1
                    missing += 1
                    left += 1

        return s[start:end]


    def minCounter(self, s,r):
        t_count = collections.Counter(r) #필요
        current_count=  collections.Counter() #
        start = float('-inf')
        end = float('inf')

        left = 0
        for right, char in enumerate(s, 1): #오른쪽 포인터 이동
            current_count[char] += 1


            while current_count & t_count == t_count: #counter 의 and 연산자 사용 ?????
                if right - left < end - start: #비교

                    start, end = left, right
                current_count[s[left]] -= 1
                left += 1
        return s[start:end]

    def longestRepeatingReplace(self, s, k):
        #Two pointer, Sliding Window, Counter
        #오른쪽 이동 왼쪽 좁히기
        #윈도우 내 출현 빈도가 가장 높은 문자의 수를 뺀 값이 k 여야
        left = right = 0
        counts = collections.Counter()

        for right in range(1, len(s) + 1):
            counts[s[right-1]] += 1
            max_char_n = counts.most_common(1)[0][1] #
            if right - left - max_char_n > k:
                counts[s[left]] -= 1
                left += 1

        return right- left



class GreedyAlgorithm:
    def fractional_knapsack(self, cargo): #분할 가능
        capacity = 15
        pack = []
        for c in cargo:
            pack.append((c[0]/c[1], c[0], c[1]))
        pack.sort(reverse = True)

        total_value = 0
        for p in pack:
            if capacity - p[2] >= 0:
                capacity -= p[2]
                total_value += p[1]
            else:
                fraction = capacity/ p[2]
                total_value += p[1]*fraction
                break
        return total_value

    #change 거스름돈, 100 50 10 가능 100 80 50 10 불가능
    #가장 큰 합 불가능


    def bestTime(self, nums):
        #여러번 가능 탐욕
        result = 0
        for i in range(len(nums) - 1):
            if nums[i+1] > nums[i]: #
                result += nums[i+1] - nums[i]
        return result

    def maxProfit(self, prices):
        return sum(max(prices[i+1] - prices[i] , 0) for i in range(len(prices)-1))

    def queueReconstructByHeight(self, people):
        heap = []
        for person in people:
            heapq.heappush(heap, (-person[0], person[1]))

        result = []

        while heap:
            person = heapq.heappop(heap)
            result.insert(person[1], [-person[0], person[1]])
        return result

    def leastInterval(self, tasks, n):
        counter = collections.Counter(tasks) #할 일
        result = 0

        while True:
            sub_count = 0

            # for task, count in collections.Counter(tasks).item():
            #     heapq.heappush(heap, (-count, task))
            #     count, task = heapq.heappop(heap)
            #     heapq.heappush(heap, (-count + 1, task))

            for task, _ in counter.most_common(n+1): # 개수 순으로 추출
                sub_count += 1
                result += 1

                counter.subtract(task)
                #? 0 이하인 아이템을 목록에서 완전히 제거
                counter += collections.Counter()
            if not counter:
                break
            result += n - sub_count + 1  #idle
        return result

    def gasStation(self, gas, cost):
        if sum(gas) < sum(cost):
            return -1

        total = 0
        start = 0
        for i in range(len(gas)):
            total += gas[i] - cost[i]
            if total < 0:
                total = 0
                start = i + 1
        return start


    def cookie(self, g, s):
        g.sort()
        s.sort()
        chile_i = cookie_j = 0

        while chile_i < len(g) and cookie_j < len(s):
            if s[cookie_j] >= g[chile_i]:
                chile_i += 1
            cookie_j += 1

        return chile_i

    def cookiebs(self, g, s):
        g.sort()
        s.sort()

        result = 0
        for i in s:
            index = bisect.bisect_right(g, i)
            if index > result:
                result += 1
        return result

## 그리디 +구현

# 큰 수의 법칙
def bigest(nums, m, k):
    nums.sort()

    first = nums[-1]
    second = nums[-2]

    count = int(m / (k+1)) * k
    count += m % (k+1)

    result = 0
    result += count * first
    result += (m - count) * second
    return result

#숫자 카드 게임
def number_card(N, M, num): #가장 낮은수가 가장 큰 수인 행
    lea = 0
    for i in num:
        t = min(i) #가장 낮은 수
        if lea < t:
            lea = t #가장 큰 수
    return lea

#1이 될때까지
def finally1(n, k):
    result = 0
    #최소니까

    #매번 하면 안됨!
    # while n != 1:
    #     result += 1
    #     if (n % k) == 0:
    #         n = n % k
    #
    #     else:
    #         n -= 1
    while True:
        target = (n // k) *k
        result += n - target
        n = target
        if n < k:
            break
        result += 1
        n //= k
    result += (n - 1)
    return result

#상하좌우
def LRUD(N, travel):
    #LRUD
    dx = [-1,1,0,0]
    dy = [0,0,1,-1]
    pos = {
        'L':0,
        'R':1,
        'U':3,
        'D':2
    }

    x = y = 0
    for tr in travel:
        i = pos[tr]
        if 0 <= x + dx[i] < N and 0 <= y + dy[i] < N :
            x += dx[i]
            y += dy[i]
    return y+1, x+1

def include3inclock(n):
    #00시 00분 00초 ~ N시 00분 00초

    result = 0
    #완전탐색

    for i in range(n+1) :
        for j in range(60):
            for k in range(60):
                if '3' in str(i) + str(j) + str(k):
                    result += 1

    return result

def knight(pos):
    dx = [2, 2, -2 ,-2, 1, -1, 1,-1]
    dy = [1, -1, 1,-1,  2, 2, -2,-2]

    dic = {'a':0, 'b':1, 'c':2,'d':3,'e':4,'f':5,'g':6,'h':7}

    x = int(pos[1]) - 1
    y = dic[pos[0]]

    result = 0
    for i in range(8):
        nx = x + dx[i]
        ny = y + dy[i]
        if 0 <= nx < 8 and 0 <= ny < 8 :
            result += 1

    return result

def game_developer(N, M, A, B, d, grid):

    dx = [-1,0,1, 0] #0 1 2 3
    dy = [0, -1 ,0,1]

    x = A
    y = B
    d = d

    visited = [[False] * N for _ in range(M)]
    result = 1
    k = 0
    while True:
        k += 1
        d += 1
        d = d % 4
        nx = x + dx[d]
        ny = y + dy[d]

        if 0 <= nx < M and 0 <= ny < N and grid[nx][ny] != 1 and visited[nx][ny] == False:
            x = nx
            y = ny
            result += 1
            k = 0

        if k == 4: #뒤로 이동
            x = x + dx[(d+2)%4]
            y = y + dy[(d+2)%4]

            if grid[x][y] == 1 :

                break


    return result



#test2.py 계속

#https://programmers.co.kr/learn/courses/30/lessons/17683

import math

def replace_step(m):
    return m.replace('C#', 'c').replace('D#', 'd').replace('F#', 'f').replace('G#', 'g').replace('A#', 'a')

def solution(m, musicinfos):
    answer = None
    max_play_time = 0
    m = replace_step(m)

    for musicinfo in musicinfos:
        start_time, end_time, name, melody = musicinfo.split(",")
        play_time = (int(end_time[:2]) * 60 + int(end_time[3:])) - (int(start_time[:2]) * 60 + int(start_time[3:]))

        melody = replace_step(melody)
        melody_repeated_count = math.ceil(play_time / len(melody))
        melody_played = (melody * melody_repeated_count)[:play_time]
        if m in melody_played and play_time > max_play_time:
            answer = name
            max_play_time = play_time
    if not answer:
        return "(None)"
    return answer

#https://programmers.co.kr/learn/courses/30/lessons/72412 -> 효율성 탈락
class TreeNode:
    def __init__(self, value=None):
        self.value = value
        self.count = 0
        self.children = collections.defaultdict(TreeNode)
        self.points = list()

class Tree:
    def __init__(self):
        self.root = TreeNode(None)

    def insert(self, i):
        node = self.root
        for val in i:#.split(' '):
            if val.isdigit():
                node.points.append(int(val))
                break

            node = node.children[val]
            node.value = val
            node.count += 1

    def search(self, query):
        node = self.root
        q = re.split(' and | ', query)
        # count = 0
        # def dfs(q, node):
        #     # if q[0] == '-':
        #     #     for nodel in node.children:
        #     #         q2 = copy.deepcopy(q[1:])
        #     #         dfs(q2, node.children[nodel])
        #     if q[0].isdigit():
        #         node.points.sort()
        #         index = bisect.bisect_left(node.points, int(q[0]))
        #         nonlocal count
        #         count += len(node.points[index:])
        #         return
        #     if q[0] in node.children:
        #         node = node.children[q[0]]
        #         q = q[1:]
        #         dfs(q, node)

        # while q:
        #     if q[0] == '-':
        #         node_list = node.children #전부
        #         q = q[1:]
        #         continue
        # dfs(q, node)
        # return count
        while q:
            print(q)
            if q[0].isdigit():
                node.points.sort()
                print("node points", node.points)
                index = bisect.bisect_left(node.points, int(q[0]))
                return len(node.points[index:])
            if q[0] in node.children:
                print(q[0])
                node = node.children[q[0]]
                print("node", node)
                q = q[1:]

import pandas as pd
def kakao_solution(info, query):
    tree = Tree()

    def check(cecklist, j, score):
        if cecklist[0] not in [j[0], '-']:
            cecklist.insert(0, '-')
        if cecklist[1] not in [j[1], '-']:
            cecklist.insert(1, '-')
        if cecklist[2] not in [j[2], '-']:
            cecklist.insert(2, '-')
        if cecklist[3] not in [j[3], '-']:
            cecklist.insert(3, '-')
        return cecklist[:4] + [score]
    def make_all_cases(separate_info):
        cases = []
        for k in range(5):
            for condition in itertools.combinations([0, 1, 2, 3], k):
                case = []
                for idx in range(4):
                    if idx not in condition:
                        case.append(separate_info[idx])
                    else:
                        case.append('-')
                #cases.append(' '.join(case))
                case.append(separate_info[-1])
                cases.append(case)
        return cases


    for i in info: #insert 시
        j = i.split()#[:-1]
        cases = make_all_cases(j)
        # print(cases)
        # return
        # score = j[-1]
        # j = j[:-1]
        # j.extend(['-', '-','-', '-'])
        # c_list = list(itertools.combinations(j, 4))
        # c_l = list()
        # for c in c_list:
        #     c_l.append(c)
        # c_r = pd.DataFrame(c_l).drop_duplicates()#.reset_index()
        #
        # asdf = [check(list(row[1:]), j, score) for row in c_r.itertuples()]
        for p in cases:
            tree.insert(p)

    results = []

    for q in query:
        results.append(tree.search(q))

    return results


from bisect import bisect_left
from itertools import combinations

def make_all_cases(separate_info):
    cases = []
    for k in range(5):
        for condition in combinations([0, 1, 2, 3], k):
            case = []
            for idx in range(4):
                if idx not in condition:
                    case.append(separate_info[idx])
                else:
                    case.append('-')
            cases.append(''.join(case))
    return cases

def solution_kakao(info, query):
    answer = []
    all_people = {}
    for i in info:
        seperate_info = i.split()
        cases = make_all_cases(seperate_info)
        for case in cases:
            if case not in all_people.keys():
                all_people[case] = [int(seperate_info[4])]
            else:
                all_people[case].append(int(seperate_info[4]))

    for key in all_people.keys():
        all_people[key].sort()

    for q in query:
        seperate_q = q.split(' and ')
        seperate_q.extend(seperate_q.pop().split())
        target = ''
        for sq in seperate_q[:4]:
            target += sq
        print(target)
        print(all_people.keys())
        if target in all_people.keys():
            answer.append(len(all_people[target]) - bisect_left(all_people[target], int(seperate_q[4]), lo=0,
                                                                hi=len(all_people[target])))
        else:
            answer.append(0)
    return answer

#오늘의집
def solution1(path):
    answer = []

    #방향
    s = path[0]
    st = 0

    direct = collections.defaultdict(dict) #grph #NWES
    direct['E']['S'] = 'right'
    direct['E']['N'] = 'left'
    direct['W']['S'] = 'left'
    direct['W']['N'] = 'right'
    direct['N']['W'] = 'left'
    direct['N']['E'] = 'right'
    direct['S']['W'] = 'right'
    direct['S']['E'] = 'left'


    for i, p in enumerate(path[1:]):
        if i >= st +5:
            st += 1
        if s != p:
            result = f"Time {st}: Go straight {(i-st+1)*100}m and turn {direct[s][p]}"
            answer.append(result)
            s = p
            st = i+1
        #break
        #print(i,p)
    return answer

def solution2(tsring, variables):
    temp = collections.defaultdict(list)
    instead = dict()
    for i, t in enumerate(tsring):
        if t == "{":
            s = i
        if t == "}":
            e = i
            temp[tsring[s+1:e]].append([s+1, e])

    #result
    check = set()
    for i,  (s, trget) in enumerate(variables):
        instead[s] = trget
        if trget.startswith('{'):
            check.add(s)

    temps = copy.deepcopy(instead)
    cnt = 0
    while cnt <= len(variables):
        ls = set()
        for key in check:

            if instead[key][1:-1] in check:
                instead[key] = temps[instead[key][1:-1]]
            else:
                if instead[key][1:-1] not in instead:
                    instead[key] = instead[key]#[1:-1]
                else:
                    instead[key] = instead[instead[key][1:-1]]
                #check.remove(key)
                ls.add(key)
        for l in ls:
            check.remove(l)
        cnt += 1

    # if all(check):
    #     print("dddddd")
    #     for key in indsted:
    #         indsted[key] = temps[indsted[key][1:-1]]
    # else:
    #     for key in indsted:
    #         if indsted[key].startswith('{'):
    #             if indsted[key][1:-1] in indsted.keys():
    #                 # 있으ㅕㄴ
    #                 indsted[key] = indsted[indsted[key][1:-1]]
    #                 check[key] = False
    #             else:# 없으ㅕㄴ
    #                 indsted[key] = trget
    #                 print("----------", )

    start = 0
    end = 0
    answer = []
    for key in temp:
        for rnge in temp[key]:
            end = rnge[0]-1
            answer.append(tsring[start:end])
            if key not in instead.keys():
                instead[key] = "{" + key + "}"

            answer.append(instead[key])

            start = rnge[1]+1
            #tsring[rnge[0]-1:rnge[1]+1] = indsted[key]



    # print("-----------")
    # print(answer)
    # print("-----------")
    # print("".join(answer))
    return "".join(answer)

def mostCommonWord(call):
    words = [call for call in call.lower()]

    counts = collections.Counter(call.lower())
    val =  counts.most_common(1)[0][1]
    dele = []
    for letter, co in counts.most_common(len(call)):
        if co == val:
            dele.append(letter)

    temp = []
    for c in call:
        if c.lower() not in dele:
            temp.append(c)
    print(temp)
    # max_count = -1
    # for letter in counter:
    #     if counter[letter] > max_count:
    #         max_count = counter[letter]
    #         max_letter = letter

    # 가장 흔하게 등장하는 단어의 첫 번째 인덱스 리턴

    # print(counts.values())
    #
    # print(counts.most_common(1)[0][0])
    # counts.most_common(1)[0]#[0]
    return "".join(temp)#counts#.most_common(1)[0][0]

#지마켓
result = 0
def componentsafds(x, y, visited, grid):
    global result
    x_lst = [0, 1, 0, -1]
    y_lst = [1, 0, -1, 0]
    # print("check", visited[x][y], x, y)
    visited[x][y] = True

    if grid[x][y] == 1:
        result += 4

        for i in range(4):
            x_new = x + x_lst[i]
            y_new = y + y_lst[i]
            if 0 <= x_new < len(grid) and 0 <= y_new < len(grid[0]):
                if grid[x_new][y_new] == 1:
                    result -= 1
                    if visited[x_new][y_new] == False:
                #print("위치", x_new, y_new)
                #ODO: visited 확인
                        componentsafds(x + x_lst[i], y+y_lst[i], visited, grid)

    return visited

def numIslandslen(grid):
    visited = [[False] * len(grid[0]) for _ in range(len(grid))]
    count = 0

    answer = []
    for x in range(len(grid)):
        for y in range(len(grid[0])):
            if grid[x][y] == 1 and visited[x][y] == False:
                print("새로시작-------")
                #result = 0
                visited = componentsafds(x, y, visited, grid)
                print("-------", result)
                count += 1
    print(answer)
    return answer

count = 0
countd = 0
def shoppingkruskal(depar, hub, dest, roads): # n 노드 m link / 1 시작 p 거쳐 q 도착
    graph = collections.defaultdict(list)

    for s, v in roads:
        graph[s].append(v)

    # print(list(itertools.combinations(candidates, 2)))
    def dfs(depar, hub, graph):
        global count
        for neighbor in graph[depar]:
            if neighbor == hub:
                count += 1
            dfs(neighbor, hub, graph)

    def dfsdest(hub, dest, graph):
        global countd
        for neighbor in graph[hub]:
            if neighbor == dest:
                countd += 1
            dfsdest(neighbor, dest, graph)
        # for i in range(index, len(candidates)):
        #     dfs(csum - candidates[i], i, result + [candidates[i]])

    dfs(depar, hub, graph)
    dfsdest(hub, dest ,graph)
    print(count*countd)
    # return results
    #
    # cities = set()
    # for r, i in roads:
    #     cities.add(r)
    #     cities.add(i)
    # n = len(cities)
    # c = dict()
    # i = 1
    # for city in cities:
    #     c[city] = i
    #     i += 1
    #
    # INF = int(1e9)
    # # #최단경로 - 다익스트라 알고리즘, 플로이드워셜 알고리즘
    # graph = [[INF for _ in range(n + 1)] for _ in range(n + 1)]
    #
    # for x, y in roads:
    #     graph[c[x]][c[y]] = min(graph[c[x]][c[y]], 1)
    # for i in range(n + 1):
    #     graph[i][i] = 0
    #
    # #
    # # # k, x ? => 플로이드 워셜 문제
    #
    # for k in range(1, n + 1):
    #     for a in range(1, n + 1):
    #         for b in range(1, n + 1):
    #             if graph[a][b] == 1:
    #                 graph[a][b] = max(graph[a][b], graph[a][k] * graph[k][b]+1)
    #                 print(graph[:][1:])
    #             else:
    #                 graph[a][b] = min(graph[a][b], graph[a][k] * graph[k][b]+1)
    #
    # # print(graph)
    # # print(graph[c[depar]][c[hub]])
    # distance = graph[c[depar]][c[hub]] * graph[c[hub]][c[des]]
    # if distance >= INF:
    #     return 0
    # else:
    #     return distance % 10007
    #
    # answer = -1
    # return answer

def gmarket(nums):
    # 1. 정렬
    nums.sort()
    # 2. 홀수번째 값의 합(짝수 index)
    return sum(nums[::2])

#분할정복
def maximize_expression(expression):
    answer = 0
    # 1) permutation
    operation_list = []
    if '*' in expression:
        operation_list.append('*')
    if '+' in expression:
        operation_list.append('+')
    if '-' in expression:
        operation_list.append('-')
    operation_permutations = list(itertools.permutations(operation_list))
    print(operation_permutations)
    #2) regex
    expression = re.split('([^0-9])', expression)
    print(expression)

    for operation_permutation in operation_permutations:
        copied_expression = expression[:]
        for operator in operation_permutation:
            while operator in copied_expression:
                op_idx = copied_expression.index(operator)

                cal = str(eval(
                    copied_expression[op_idx-1] + copied_expression[op_idx] + copied_expression[op_idx+1]
                ))
                copied_expression[op_idx - 1] = cal
                copied_expression = copied_expression[:op_idx] + copied_expression[op_idx+2:]

                # if copied_expression[op_idx] == '*':
                #     cal = int(copied_expression[op_idx-1]) * int(copied_expression[op_idx+1])
                # elif copied_expression[op_idx] == '+':
                #     cal = int(copied_expression[op_idx-1]) + int(copied_expression[op_idx+1])
                # else:
                #     cal = int(copied_expression[op_idx-1]) - int(copied_expression[op_idx+1])

        answer = max(answer, abs((int(copied_expression[0]))))
    return answer
if __name__ == '__main__':
    # print(numIslandslen([[0,0,1,0,0],[0,1,1,0,1],[0,0,1,0,1],[1,1,1,0,1]]))
    # print(numIslandslen([[1,0,1,1],[0,0,1,1],[1,1,0,1],[1,1,0,0]]	))
    # print(shoppingkruskal("SEOUL", "DAEGU", "YEOSU", [["ULSAN","BUSAN"],["DAEJEON","ULSAN"],["DAEJEON","GWANGJU"],["SEOUL","DAEJEON"],["SEOUL","ULSAN"],["DAEJEON","DAEGU"],["GWANGJU","BUSAN"],["DAEGU","GWANGJU"],["DAEGU","BUSAN"],["ULSAN","DAEGU"],["GWANGJU","YEOSU"],["BUSAN","YEOSU"]]))
    # print(shoppingkruskal("ULSAN", "SEOUL", "BUSAN", [["SEOUL","DAEJEON"],["ULSAN","BUSAN"],["DAEJEON","ULSAN"],["DAEJEON","GWANGJU"],["SEOUL","ULSAN"],["DAEJEON","BUSAN"],["GWANGJU","BUSAN"]]))

    # print(mostCommonWord("abcabcdefabc"))
    # print(mostCommonWord("abxdeydeabz"))
    # print(mostCommonWord("abcabca"))
    # print(mostCommonWord("ABCabcA"))

    # print(solution("EEESEEEEEENNNN"))
    # print(solution("SSSSSSWWWNNNNNN"))
    # print(solution("this is {template} {template} is {state}", [["template", "string"], ["state", "changed"]]))
    # print(solution("this is {template} {template} is {state}", [["template", "string"], ["state", "{template}"]]))
    # print(solution("this is {template} {template} is {state}", [["template", "{state}"], ["state", "{template}"]]))
    # # print(solution("this is {template} {template} is {state}", [["template", "{state}"], ["state", "{templates}"]]))
    # print(solution("{a} {b} {c} {d} {i}", [["b", "{c}"], ["a", "{b}"], ["e", "{f}"], ["h", "i"], ["d", "{e}"], ["f", "{d}"], ["c", "d"]]))

    data = """
    park 주민등록번호
    """
    #pat = re.compile("(\d{6})[-]\d{7}")
    #print(pat.sub("\g<1>-*******", data))
    #print(maximize_expression("100-200*300-500+20"))
    print()

   # print(exit())
   # print(soldier())
   # print(uglyNumber())
   # print(editDistance("abc", "ab"))
   # print(editDistance("ca", "abc"))
   # print(editDistance("abababababa", "aaaaaaaaaaa"))
    # print(shortPalindrome("ghhggh"))
    # print(search_in_rotated_array([2,5,6,7,0,1,2],0))
    # print(intersection([4,9,5], [9,4,9, 8, 4]))
    # print(intersection([1,2,2,1], [2, 2]))
    # print(two_sum([2, 7, 11, 15], 9))
    # print(two_sum([3, 2, 4], 6))
    # print(two_sum([3, 3], 6))
    # print(two_sum([0, 4, 3, 0], 0))
    # print(two_sum([-3,4,3,90], 0))
    # print(search2D([[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], 20))
    # print(find_item([8, 3, 7, 9, 2], [5,7,9]))
    # print(teokbokki([19,15,10,17],6))
    # print(count_target([1,1,2,2,2,2,3],2))
    # print(fixed_point([-15,-4,2,8,13]))
    #print(wifi([1,2,8,4,9], 3))
    #print(immigration(6, [7,10]))
    #print(stones(25, [2,14,11, 21, 17], 2))
    #print(kth_number(3, 7))
    #print(cut_tree([20, 15, 10, 17], 7))
    #print(cut_tree([4, 42, 40, 26, 46], 20))
    #print(cut_lan([802,743,457,539], 11)) #200
    #print(budget([120,110,140,150],485))
    #print(budget([70, 80,30, 40, 100], 450))
    # print(search_lyrics(["frodo", "front", "frost", "frozen", "frame", "kakao"], ["fro??", "????o", "fr???", "fro???", "pro?"]))
    # print(search_lyrics_trie(["frodo", "front", "frost", "frozen", "frame", "kakao"], ["fro??", "????o", "fr???", "fro???", "pro?", "?????"]))
    # print(palindromeParis(['bat', 'tab', 'cat']))
    # print(palindromeParis(['a', '']))
    # print(palindromeParis(['abcd', 'dcba', 'lls', 's', 'sssll']))
    # print(numIslands([
    #                   ["1","1","1","1","0"],
    #                   ["1","1","0","1","0"],
    #                   ["1","1","0","0","0"],
    #                   ["0","0","0","0","0"]
    #                 ]))
    # print(numIslands([
    #                   ["1","1","0","0","0"],
    #                   ["1","1","0","0","0"],
    #                   ["0","0","1","0","0"],
    #                   ["0","0","0","1","1"]
    #                 ]))
    # print(numIslands([["1","1","1"],["0","1","0"],["1","1","1"]]))
    # print(numIslands([["1","0","1","1","1"],["1","0","1","0","1"],["1","1","1","0","1"]]))

    # print(permutation([1,2,3]))
    # print(combination(4,2))
    #print(combinationSum([2,3,6,7], 7))
    # print(subsets([1,2,3]))

    #print(findItinerary([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))
    #print(findItinerary([["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]))
    # print(networkDelayTime([[2,1,1],[2,3,1],[3,4,1]],4,2))
    # print(findCheapestPrice(4, [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], 0,3,1))
    # print(juiceComponent([[0,0,1,1,0],[0,0,0,1,1],[1,1,1,1,1],[0,0,0,0,0]]))

    # print(maze(5,6,[[1,0,1,0,1,0], [1,1,1,1,1,1], [0,0,0,0,0,1], [1,1,1,1,1,1], [1,1,1,1,1,1]]))


    # print(Floyd(4, 7, [[1,2,5], [2,1,7], [1,4,8], [3,1,2], [2, 3, 9], [4,3,3], [3,4,4]]))
    # print(future(5,7,[[1,2],[1,3], [1,4], [2,4], [3,4], [3,5], [4,5]], 4,5)) #1 시작, 4 거쳐 5 도착
    # print(future(4,2,[[1,3],[2,4]], 3,4))
    # print(telegraph(3,2,1, [[1,2,4], [1,3,2]])) #2 4
    # print(topology_sort(7, 8, [[1,2],[1,5],[2,3],[2,6],[3,4],[4,7],[5,6],[6,4]]))
    # print(team(7,8,[[0,1,3],[1,1,7],[0,7,6],[1,7,1],[0,3,7],[0,4,2],[0,1,1],[1,1,1]]))
    # print(city_plan(7,12,[[1,2,3],[1,3,2],[3,2,1],[2,5,2],[3,4,4],[7,3,6],[5,1,5],[1,6,2],[6,4,1],[6,5,3],[4,5,3],[6,7,4]]))
    #print(curriculum(5, [[10], [10, 1], [4, 1], [4, 3, 1], [3, 3]]))

    # print(find_shortest(4, 4, 2, 1, [[1,2], [1,3], [2,3], [2,4]]))
    # print(find_shortest(4, 4, 1, 1, [[1, 2], [1, 3], [2, 3], [2, 4]]))
    # print(find_shortest(4, 3, 2, 1, [[1, 2], [1, 3], [1, 4]]))

    # print(laboratory(7,7,[[2,0,0,0,1,1,0],[0,0,1,0,1,2,0],[0,1,1,0,1,0,0],[0,1,0,0,0,0,0],[0,0,0,0,0,1,1],[0,1,0,0,0,0,0],[0,1,0,0,0,0,0]]))
    # print(laboratory(4,6,[[0,0,0,0,0,0],[1,0,0,0,0,2],[1,1,1,0,0,2],[0,0,0,0,0,2]]))
    # print(laboratory(8,8,[[2,0,0,0,0,0,0,2],[2,0,0,0,0,0,0,2],[2,0,0,0,0,0,0,2],[2,0,0,0,0,0,0,2],[2,0,0,0,0,0,0,2],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]))
    # print(contagion(3,3,[[1,0,2], [0,0,0],[3,0,0]],2,3,2))
    # print(contagion(3,3,[[1,0,2], [0,0,0],[3,0,0]],1,2,2))
    # print(bracket("()))((()"))
    # print(bracket(")("))
    # print(bracket("(()())()"))

    # print(pushdfs([1,2,3,4,5,6], 2,1,1,1))
    # print(pushdfs([3, 4, 5], 1,0,1,0))
    # print(pushdfs([5,6],0,0,1,0))

    #X, S, T = 'X', 'S','T'
    # print(student_teacher(5, [[X, S, X, X, T],
    #                             [T, X, S, X, X],
    #                             [X, X, X, X, X],
    #                             [X,T, X, X, X],
    #                             [X, X, T, X, X]]))
    #
    # print(student_teacher(4, [[S, S, S, T],
    #                          [T, X, X, X],
    #                          [X, X, X, X],
    #                          [T, T, T, X]]))

    # print(population(2, 20, 50, [[50, 30], [20,40]]))
    # print(population(2, 40, 50, [[50, 30], [20, 40]]))
    # print(population(2, 20, 50, [[50, 30], [30, 40]]))
    # print(population(3, 5, 10, [[10,15 ,0], [20,30,25],[40, 22,10]]))
    # print(population(4, 10, 50, [[10, 100, 20, 90], [80, 100,60,70],[70,20,30,40],[50,20,100,10]]))

    # print(robot([[0, 0, 0, 1, 1],[0, 0, 0, 1, 0],[0, 1, 0, 1, 1],[1, 1, 0, 0, 1],[0, 0, 0, 0, 0]]))

    # print(floydvo(5, [[1, 2, 2],
    #                 [1, 3, 3],
    #                 [1, 4, 1],
    #                 [1, 5, 10],
    #                 [2, 4 ,2],
    #                 [3, 4, 1],
    #                 [3, 5, 1],
    #                 [4, 5, 3],
    #                 [3, 5, 10],
    #                 [3 ,1 ,8],
    #                 [1 ,4, 2],
    #                 [5, 1, 7],
    #                 [3, 4, 2],
    #                 [5, 2, 4]]))

    # print(ranking(6, [[1,5], [3,4], [4,2], [4,6], [5,2], [5,4]]))
    # print(mars_exploration(3, [[5,5,4], [3,9,1], [3,2,7]]))
    # print(mars_exploration(5, [[3,7,2,0,1], [2,8,0,9,1], [1,2,1,8,1],[9,8,9,2,0],[3,6,5,1,5]]))
    # print(mars_exploration(7, [[9,0,5,1,1,5,3], [4,1,2,1,6,5,3],[0,7,6,1,6,8,5],[1,1,7,8,3,2,3],[9,4,0,7,6,4,1],[5,8,3,2,4,8,3], [7,4,8,4,8,3,4]]))
    # print(hideandseek(6, [[3,6],[4,3],[3,2],[1,3],[1,2],[2,4],[5,2]]))
    # print(travel_plan(5, [[0, 1, 0, 1, 1], [1, 0, 1,1,0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 0, 0, 0, 0]], [2,3,5,4]))
    # print(travel_plan(5, [[0, 1, 0, 1, 0], [1, 0, 1,1,0], [0, 1, 0, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], [2,3,5,4]))
    # print(gate(4, 3, [4, 1, 1]))
    # print(gate(4, 6, [2, 2, 3, 3, 4, 4]))
    # print(dark(7, [[0,1,7],[0,3,5],[1,2,8],[1,3,9],[1,4,7],[2,4,5],[3,4,15],[3,5,6],[4,5,8],[4,6,9],[5,6,11]]))
    # print(planet_turnel(5, [[11, -15, -15], [14, -5, -15], [-1, -1, -5], [10, -4, -1], [19, -4, 19]]))
    # print(final_ranking(5, [5,4,3,2,1], [[2,4], [3,4]]))
    # print(final_ranking(3, [2,3,1],[]))
    # print(final_ranking(4, [1,2,3,4], [[1,2],[3,4],[2,3]]))
    # print(baby_shark([[0,0,0],[0,0,0],[0,9,0]]))
    # print(baby_shark([[0,0,1],[0,0,0],[0,9,0]]))
    # print(baby_shark([[4,3,2,1],[0,0,0,0],[0,0,9,0],[1,2,3,4]]))
    # print(adolescent_shark([[7,6,2,3,15,6,9,8],[3,1,1,8,14,7,10,1],[6,1,13,6,4,3,11,4],[16,1,8,7,5,2,12,2]]))
    # print(adolescent_shark([[16,7,1,4,4,3,12,8], [14,7,7,6,3,4,10,2], [5,2,15,2,8,3,6,4],
    #                         [11,8,2,4,13,5,9,4]]))
    # print(adolescent_shark([[12, 6, 14, 5, 4,5,6,7], [15,1,11,7,3,7,7,5], [10,3,8,3,16,6,1,1],
    #                         [5,8,2,7,13,6,9,2]]))
    #
    # print(adult_shark(5, 4, 4, [[0, 0, 0, 0, 3],
    #                             [0,2 ,0 ,0 ,0],
    #                             [1 ,0, 0, 0,4],
    #                            [0 ,0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0]],
    #                             [4, 4, 3, 1],
    #                            [[[2, 3, 1, 4],
    #                             [4 ,1, 2, 3],
    #                             [3, 4 ,2, 1],
    #                             [4 ,3, 1, 2]],
    #                             [[2 ,4, 3, 1],
    #                            [ 2, 1 ,3, 4],
    #                            [3, 4, 1 ,2],
    #                             [4, 1, 2, 3]],
    #                             [[4, 3 ,2 ,1],
    #                            [ 1, 4 ,3, 2],
    #                             [1 ,3 ,2, 4],
    #                            [3, 2, 1, 4]],
    #                            [[ 3 ,4,1 ,2],
    #                             [3, 2 ,4, 1],
    #                             [1 ,4 ,2 ,3],
    #                             [1, 4 ,2 ,3]]]))
    # print(adult_shark2(5, 4, 4, [[0, 0, 0, 0, 3],
    #                             [0, 2, 0, 0, 0],
    #                             [1, 0, 0, 0, 4],
    #                             [0, 0, 0, 0, 0],
    #                             [0, 0, 0, 0, 0]],
    #                   [4, 4, 3, 1],
    #                   [[[2, 3, 1, 4],
    #                     [4, 1, 2, 3],
    #                     [3, 4, 2, 1],
    #                     [4, 3, 1, 2]],
    #                    [[2, 4, 3, 1],
    #                     [2, 1, 3, 4],
    #                     [3, 4, 1, 2],
    #                     [4, 1, 2, 3]],
    #                    [[4, 3, 2, 1],
    #                     [1, 4, 3, 2],
    #                     [1, 3, 2, 4],
    #                     [3, 2, 1, 4]],
    #                    [[3, 4, 1, 2],
    #                     [3, 2, 4, 1],
    #                     [1, 4, 2, 3],
    #                     [1, 4, 2, 3]]]))
    # print(adult_shark(4, 2, 6, [[1, 0, 0, 0],
    #                             [0, 0 ,0, 0],
    #                             [0 ,0, 0, 0],
    #                             [0 ,0 ,0 ,2]],
    #                   [4,3],
    #                   [[[1, 2, 3, 4],
    #                     [2, 3 ,4, 1],
    #                     [3, 4,1 ,2],
    #                     [4, 1, 2, 3]],
    #                     [[1, 2,3,4],
    #                     [2, 3, 4, 1],
    #                     [3, 4, 1, 2],
    #                     [4 ,1,2 ,3]]]))
    # print(adult_shark2(4, 2, 6, [[1, 0, 0, 0],
    #                             [0, 0 ,0, 0],
    #                             [0 ,0, 0, 0],
    #                             [0 ,0 ,0 ,2]],
    #                   [4,3],
    #                   [[[1, 2, 3, 4],
    #                     [2, 3 ,4, 1],
    #                     [3, 4,1 ,2],
    #                     [4, 1, 2, 3]],
    #                     [[1, 2,3,4],
    #                     [2, 3, 4, 1],
    #                     [3, 4, 1, 2],
    #                     [4 ,1,2 ,3]]]))
    #
    #
    # print(adult_shark(5, 4, 1,
    #                     [[0, 0, 0, 0, 3],
    #                     [0, 2, 0, 0, 0],
    #                     [1, 0, 0, 0, 4],
    #                     [0, 0, 0, 0, 0],
    #                     [0 ,0, 0, 0, 0]],
    #                     [4, 4, 3, 1],
    #                     [[[2, 3, 1, 4],
    #                     [4,1,2 ,3],
    #                     [3, 4, 2, 1],
    #                     [4, 3, 1, 2]],
    #                     [[2, 4, 3, 1],
    #                     [2, 1, 3, 4],
    #                     [3, 4, 1, 2],
    #                     [4, 1, 2, 3]],
    #                     [[4, 3, 2, 1],
    #                     [1, 4, 3, 2],
    #                     [1, 3, 2, 4],
    #                     [3, 2, 1, 4]],
    #                     [[3, 4, 1, 2],
    #                     [3, 2, 4, 1],
    #                     [1, 4, 2, 3],
    #                     [1, 4, 2, 3]]]))
    #
    # print(adult_shark(5, 4, 10,
    #                     [[0, 0, 0, 0, 3],
    #                     [0, 0, 0, 0, 0],
    #                     [1, 2, 0, 0, 0],
    #                     [0, 0, 0 ,0, 4],
    #                     [0, 0, 0, 0, 0]],
    #                     [4, 4, 3, 1],
    #                     [[[2, 3, 1, 4],
    #                     [4, 1, 2, 3],
    #                     [3, 4, 2, 1],
    #                     [4, 3, 1, 2]],
    #                     [[2, 4, 3, 1],
    #                     [2, 1, 3, 4],
    #                     [3, 4, 1, 2],
    #                     [4, 1, 2, 3]],
    #                     [[4, 3, 2, 1],
    #                     [1, 4, 3, 2],
    #                     [1, 3, 2, 4],
    #                     [3, 2, 1, 4]],
    #                     [[3 ,4, 1, 2],
    #                     [3, 2, 4, 1],
    #                     [1, 4, 2, 3],
    #                     [1, 4, 2, 3]]]))
    # print(target_number([1,1,1,1,1], 3))
    # print(target_number([4,1,2,1], 4))
    # print(network(3, [[1, 1, 0], [1, 1, 0], [0, 0, 1]]))
    # print(network(3, [[1, 1, 0], [1, 1, 1], [0, 1, 1]]))
    # print(word_transition("hit", "cog", ['hot', 'dot', 'dog', 'lot', 'log', 'cog']))
    #print(findItinerary([["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]))
    # print(findItinerary([["ICN", "JFK"], ["HND", "IAD"], ["JFK", "HND"]]))
    # print(findItinerary([["ICN", "SFO"], ["ICN", "ATL"], ["SFO", "ATL"], ["ATL", "ICN"], ["ATL","SFO"]]))

    # print(tomato(6,4, [[0, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0],
    #                     [0 ,0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 1]]))
    # print(tomato(6, 4, [[1, -1, 0, 0, 0, 0],
    #                     [0, -1, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, -1, 0],
    #                     [0 ,0, 0, 0, -1, 1]]))
    # print(tomato(6, 4, [[0, -1, 0, 0, 0, 0],
    #                     [-1, 0, 0, 0, 0, 0],
    #                     [0, 0, 0, 0, 0, 0],
    #                     [0 ,0, 0, 0, 0, 1]]))
    # print(organic_(10, 8, 17, [[0, 0],
    #                             [1, 0],
    #                             [1, 1],
    #                             [4, 2],
    #                             [4, 3],
    #                             [4, 5],
    #                             [2, 4],
    #                            [3, 4],
    #                             [7, 4],
    #                             [8, 4],
    #                             [9, 4],
    #                             [7, 5],
    #                             [8, 5],
    #                             [9, 5],
    #                             [7, 6],
    #                             [8 ,6],
    #                             [9 ,6]]))
    #
    # print(organic_(10, 10, 1, [[5, 5]]))
    # print(virus(7, 6, [[1 ,2],
    #                     [2 ,3],
    #                     [1 ,5],
    #                     [5 ,2],
    #                     [5 ,6],
    #                     [4, 7]]))
    # print(makeNumber(7, [[0,1,1,0,1,0,0],
    #                     [0,1,1,0,1,0,1],
    #                     [1,1,1,0,1,0,1],
    #                     [0,0,0,0,1,1,1],
    #                     [0,1,0,0,0,0,0],
    #                     [0,1,1,1,1,1,0],
    #                     [0,1,1,1,0,0,0]]))
    # print(dfsbfs(4, 5, 1, [[1,2], [1,3], [1,4], [2,4], [3,4]]))
    # print(dfsbfs(1000, 1, 1000, [[999, 1000]]))
    # print(hanoi(3, 1,3))

    # print(sort_cordinates([[0,4],[1,2],[1,-1],[2,2],[3,3]]))



    # print(bubblesort([38,27,43,3,98,82,10]))
    # print(mergesort([38,27,43,3,98,82,10]))
    # print(quicksort([38,27,43,3,98,82,10], 0, 6))
    # print(selectionsort([38,27,43,3,98,82,10]))
    # print(insertionsort([38,27,43,3,98,82,10]))
    # print(quick([38,27,43,3,98,82,10], 0, 6))
    # print(quick_sort([38,27,43,3,98,82,10]))
    # print(countsort([38,27,43,3,98,82,10]))
    # print(radixsort([38,27,43,3,98,82,10]))

    # print(sortList(ListNode(val= 4, next= ListNode(val= 2, next= ListNode(val= 1, next= ListNode(val= 3, next= None))))))
    # print(mergeIntervals([[1,3], [2,6], [8, 10], [15, 18]]))
    # print(insertion(ListNode(val=4, next= ListNode(val= 2, next= ListNode(val= 1, next= ListNode(val= 3, next= None))))))
    # print(largestNumber([10,2]))
    # print(valid_anagram("anagram", "nagaram"))
    # print(sort_colors([2,0,2,1,1,0]))
    # print(k_closets_points_to_origin([[1,3], [-2, 2]], 1))
    # print(k_closets_points_to_origin([[3,3],[5,-1], [-2, 4]], 2))
    # print(sort_array([3, 15, 27, 12]))
    # print(grade_sort([["홍길동", 95], ["이순신", 77]]))
    # print(change_element(5, 3, [1,2,5,4,3], [5,5,6,6,5]))
    # print(failure_rate(5, [2, 1, 2, 6, 2, 4, 3, 3]))
    # print(failure_rate(4, [4,4,4,4,4]))
    # print(subject_sort([["Junkyu", 50, 60, 100],
    #                     ["Sangkeun", 80, 60, 50],
    #                     ["Sunyoung", 80, 70, 100],
    #                     ["Soong", 50, 60, 90],
    #                     ["Haebin", 50, 60, 100],
    #                     ["Kangsoo", 60, 80, 100],
    #                      ["Donghyuk" ,80, 60, 100],
    #                     ["Sei", 70, 70, 70],
    #                     ["Wonseob", 70, 70, 90],
    #                     ["Sanghyun", 70, 70, 80],
    #                     ["nsj", 80, 80, 80],
    #                     ["Taewhan", 50, 60, 90]]))
    # print(anthena(4, [5,1,7,9]))
    # print(card(3, [10, 20, 40]))

    # print(kakao_solution(["java backend junior pizza 150","python frontend senior chicken 210","python frontend senior chicken 150","cpp backend senior pizza 260","java backend junior chicken 80","python backend senior chicken 50"],
    #                      #["- and - and - and - 150"]))
    #                      ["java and backend and junior and pizza 100","python and frontend and senior and chicken 200","cpp and - and senior and pizza 250","- and backend and senior and - 150","- and - and - and chicken 100","- and - and - and - 150"]))


    # print(solution("ABCDEFG",["12:00,12:14,HELLO,CDEFGAB", "13:00,13:05,WORLD,ABCDEF"]))
    # print(solution("CC#BCC#BCC#BCC#B",["03:00,03:30,FOO,CC#B", "04:00,04:08,BAR,CC#BCC#BCC#B"]	))
    # print(solution("ABC",["12:00,12:14,HELLO,C#DEFGAB", "13:00,13:05,WORLD,ABCDEF"]))

    # DC = division_and_conquer()
    # print(DC.majority_element([2,2,1,1,1,2,2]))
    # print(DC.ma([2,2,1,1,1,2,2]))
    # print(DC.majorityElement([2,2,1,1,1,2,2]))
    # print(DC.mae([2,2,1,1,1,2,2]))
    # print(DC.diffWaysToCompute("2*3-4*5"))
    # SW = SlidingWindow()
    # print(SW.sw([1,3,-1,-3,5,3,6,7],3))
    # print(SW.msw([1,3,-1,-3,5,3,6,7],3))
    # print(SW.minWindow("ADOBECODEBANC", "ABC"))
    # print(SW.twopoint_slider("ADOBECODEBANC", "ABC"))
    # print(SW.minCounter("ADOBECODEBANC", "ABC"))
    # print(SW.longestRepeatingReplace("AAABBC", 2))
    # GA = GreedyAlgorithm()
    # print(GA.fractional_knapsack([(4,12),(2,1),(10,4),(1,1), (2,2),]))
    # print(GA.bestTime([7,1,5,3,6,4]))
    # print(GA.maxProfit([7,1,5,3,6,4]))
    # print(GA.queueReconstructByHeight([[7,0],[4,4], [7,1], [5,0],[6,1],[5,2]]))
    # print(GA.leastInterval(["A", "A", "A", "B", "C", "D"], n = 2))
    # print(GA.gasStation([1,2,3,4,5],[3,4,5,1,2]))
    # print(GA.cookie([1,2,3],[1,1]))
    # print(GA.cookie([1,2],[1,2,3]))
    # print(GA.cookiebs([1,2,3],[1,1]))
    # print(GA.cookiebs([1,2],[1,2,3]))

    print(bigest([2,4,5,4,6], 8, 3))
    print(number_card(3, 3, [[3,1,2], [4,1,4],[2,2,2]]))
    print(finally1(25,5))
    print(LRUD(5, ["R", "R", "R", "U", "D","D"]))
    print(include3inclock(5))
    print(knight('a1'))
    print(game_developer(4,4,1,1,0,[[1,1,1,1],[1,0,0,1],[1,1,0,1],[1,1,1,1]]))