#counting sort
import bisect
import collections
import copy
import heapq
import math
import re
from typing import Optional
from xmlrpc.client import boolean


def counting_sort(lst):
    mina=min(lst)
    maxa = max(lst)

    counting = [0] * len(lst)
    for i in range(len(lst)):
        counting[lst[i]-mina]+=1 #counting

    #printing
    result = []
    for i in range(len(counting)):
        for j in range(counting[i]):
            result.append(i+mina)

    return result

def radix_sort(lst):
    D = int(math.log10(max(lst)))

    for i in range(D+1):
        bucket = []

        for j in range(0, 10):
            bucket.append([])
        for j in range(len(lst)):
            digit = int(lst[j] // math.pow(10,i)) % 10
            bucket[digit].append(lst[j])
        #printing
        cnt = 0
        for j in range(0, 10):
            for i in bucket[j]:
                lst[cnt] = i
                cnt += 1

    return lst


def binarySearch(lst, target):
    left, right = 0, len(lst)

    while left<= right:
        mid = (left+right)//2
        if lst[mid] == target:
            return mid
        if lst[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def merge_sort(lst):
    if len(lst) == 1:
        return lst
    #분할 devide
    left = merge_sort(lst[len(lst)//2:])
    right = merge_sort(lst[:len(lst)//2])

    #conquer
    i = j = 0
    result = []
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    if i == len(left):
        result.extend(right[j:])
    if j == len(right):
        result.extend(left[i:])
    return result

def longestPalindrome(s):
    def expand(left, right):
        while left > 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return s[left+1:right]

    result = ""
    for i in range(len(s)):
        result = max(result,
            expand(i, i+1),
            key=len)
    return result

#두수의합
def twoSum(nums, target):
    left, right = 0, len(nums) - 1

    while left < right:
        g = nums[left] + nums[right]
        if g == target:
            return [left, right]
        elif g < target:
            left += 1
        else:
            right -= 1
    return -1

#빗물트래핑
def trapping(nums):
    if not nums:
        return 0

    left, right = 0, len(nums) - 1
    left_max = nums[left]
    right_max = nums[right]
    result = 0
    while left < right :
        left_max, right_max =  max(left_max, nums[left]), max(right_max, nums[right])
        if left_max <= right_max:
            result += left_max - nums[left]
            left += 1
        else:
            result += right_max - nums[right]
            right -= 1

    return result


def trappingStack(nums):
    print(nums)
    #변곡점에서
    result = 0
    stack = []
    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:

            top = stack.pop()
            if not len(stack):
                break

            distance = i - stack[-1] - 1
            waters = min(nums[i], nums[stack[-1]]) - nums[top]
            result += distance * waters

        stack.append(i)
    return result

#세수의합
def threesum(nums):
    result = []
    nums.sort()

    for i in range(len(nums)-2):
        if i > 0 and nums[i] == nums[i-1]:
            continue

        left, right = i + 1, len(nums) - 1
        while left < right:
            sum = nums[i] + nums[left] + nums[right]
            if sum < 0:
                left += 1
            elif sum > 0 :
                right -= 1
            else:
                result.append([nums[i], nums[left], nums[right]])

                while left<right and nums[left] == nums[left+1]:
                    left +=1
                while left<right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -=1




    return result

#자신을 제외한 곱
def productexcept(nums):
    p = 1
    result = [p]

    for i in range(len(nums) - 1):
        result.append(p * nums[i])
        p *= nums[i]

    q = 1
    for j in range(len(nums) - 1, -1, -1):
        result[j] *=  q
        q *= nums[j]

    return result

def rainTrapping(lst): #빗물트래핑

    left, right = 0, len(lst) - 1

    left_max = lst[left]
    right_max = lst[right]

    answer = 0
    while left < right:
        if left_max <= right_max :
            left_max = max(left_max, lst[left])
            answer += left_max - lst[left]
            left += 1
        else:

            right_max = max(right_max, lst[right])
            answer += right_max - lst[right]
            right -= 1

    return answer

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListProblem:
    # def isPalindrome(self, head):
    #     #runner
    #     slow = fast = head
    #     rev = None
    #     while fast and fast.next:
    #         fast = fast.next.next
    #         rev, rev.next , slow = slow, rev, slow.next
    #
    #     while slow and rev.val == slow.val:
    #         rev, slow = rev.next, slow.next
    #     return not rev # not None = True

    def isPalindrome(self, head):
        #runner
        slow = fast = head
        rev = None
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev , slow.next

        while slow and slow.val == rev.val:
            slow, rev= slow.next, rev.next
        return not rev

    # def mergeTwoLists(self, l1:ListNode, l2:ListNode) -> ListNode:
    #     if (not l1) or (l2 and l1.val > l2.val):
    #         l1, l2 = l2, l1
    #
    #     if l1:
    #         l1.next = self.mergeTwoLists(l1.next, l2)
    #
    #     return l1

    def mergeTwoLists(self, l1, l2):
        if (not l1) or (l2 and l1.val > l2.val):
            l1, l2 = l2, l1

        if l1:
            l1.next = self.mergeTwoLists(l1.next, l2)

        return l1

    # def reverseList(self, head):
    #     if head is None:
    #         return head
    #     prev, cur = None, head
    #
    #     while cur:
    #         tmpNode = cur.next
    #         cur.next = prev
    #         prev, cur, = cur, tmpNode
    #
    #     return prev

    def reverseList(self, head):

        prev, cur = ListNode(None, head), head

        while cur and cur.next:
            nxt = cur.next
            cur.next = prev

            prev, cur = cur, nxt

        return cur

    # def reverseLinkedList(self, head):
    #     def reverse(node, prev = None):
    #         if not node:
    #             return prev
    #         next, node.next = node.next, prev
    #         return reverse(next, node)
    #
    #     return reverse(head)

    def reverseLinkedList(self, head):

        def reverse(node, prev=None):
            if not node:
                return prev

            next, node.next = node.next, prev
            return reverse(next, node)

        return reverse(head)

    def reverseBetween(self, head, left, right):

        leftMax, cur = ListNode(None, head), head
        for _ in range(left - 1):
            leftMax, cur = leftMax.next, cur.next

        prev = None
        for _ in range(right -  left + 1):
            tmpNode = cur.next
            cur.next = prev
            prev, cur = cur, tmpNode

        leftMax.next.next = cur
        leftMax.next = prev
        #tmp.next = cur

        return head

    def addTwo(self, l1, l2):

        # root = head = ListNode(0)
        # carry = 0
        # while l1 or l2 or carry:
        #     sum = 0
        #     if l1:
        #         sum += l1.val
        #         l1 = l1.next
        #     if l2:
        #         sum += l2.val
        #         l2 = l2.next
        #
        #     carry, val = divmod(carry + sum, 10)
        #     head.next = ListNode(val)
        #     head = head.next
        # return root.next

        carry = 0
        root = head = ListNode(0)
        while l1 or l2 or carry:
            sum = 0
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next

            carry, val = divmod(carry + sum, 10)

            head.next = ListNode(val)
            head = head.next
        return root.next


    # def swapReverse(self, head):
    #     if head and head.next:
    #         p = head.next
    #         head.next = self.swapPairs(p.next)
    #         p.next = head
    #         return p
    #     return head

    def swapReverse(self, head):
        if head and head.next:
            p = head.next
            head.next = self.swapPairs(p.next)
            p.next = head
            return p
        return head


    def swapPairs(self, head):
        root = prev = ListNode(None, head)
        cur = head
        while cur and cur.next: ###
            b = cur.next

            cur.next = b.next
            b.next = cur

            prev.next = b

            prev, cur = prev.next.next, cur.next

        return root.next

    def swapPairs(self, head):
        root = prev = ListNode(None, head)
        cur = head

        while cur and cur.next:
            b = cur.next

            cur.next = b.next
            b.next = cur
            prev.next = b

            prev, cur = prev.next.next, cur.next

        return root.next

    # def evenOddList(self, head):
    #     if head is None:
    #         return head
    #     odd = head
    #     even = head.next
    #     even_head = head.next
    #
    #     while even and even.next:
    #         odd.next, even.next = odd.next.next, even.next.next
    #         odd, even = odd.next, even.next
    #
    #     odd.next = even_head
    #     return head

    def evenOddList(self, head):
        odd = head
        even = head.next
        even_head = head.next

        if even and even.next:
            odd.next = odd.next.next
            even.next = even.next.next
            odd, even = odd.next, even.next

        odd.next = even_head
        return head

#2021 카카오 코테 인턴십
#시험장 나누기
#Q k 개의 그룹으로 나누어(k-1 번 분리), 가장 큰 그룹의 인원을 최소화
# -> 그룹의 인원이 최대 x 일때, 최소 그룹 몇 개? k
#이진 탐색, bfs - recursion


answer = 0
def dfs2(node, x):
        global answer
        print(node)

        left = 0
        if l[node] != -1:
            left = dfs2(l[node], x)
        right = 0
        if r[node] != -1:
            right= dfs2(r[node], x)

        if l[node] == -1 and r[node] == -1:
            print("값, node", node, num[node])
            return num[node]
        if left + right + num[node] <= x:
            return left+ right + num[node]
        # if num[node] + min(left, right) <= x:
        #     cnt += 1  # 그룹
        #     return x[cur] + min(lv, rv)
        elif left >= right and right + num[node] <= x:
            answer += 1
            print("추가 right 이커", right, left, "answer", answer)
            return num[node] + right
        elif right > left and left + num[node] <= x:
            answer += 1
            print("추가 left 이커", right, left)

            return num[node] + left
        elif left + num[node] > x and right+ num[node]>x:
            answer += 2
            print("둘다 잘라", right, left)

            return num[node]

def solve2(root, x):
    global answer
    answer = 0
    answer += 1
    dfs2(root, x)
    print("-----------------", answer)
    return answer

num = [0] * 20
l = [-1] * 20
r = [-1] * 20

def divide_test(k, nump, links):

    for i in range(len(nump)):
        num[i] = nump[i]

    set_notroot = set()
    set_root = set()
    for i in range(len(num)):
        set_root.add(i)
    for i in range(len(links)):
        #if links[i][0] != -1:
            l[i] = links[i][0]
        #if links[i][1] != -1:
            r[i] = links[i][1]
            set_notroot.add(links[i][0])
            set_notroot.add(links[i][1])
    root = set_root - set_notroot
    root = root.pop()

    start = max(nump)
    end = sum(nump)

    while start <= end:
        mid = (start + end) // 2
        print("mid, end, start", mid, end, start)
        p = solve2(root, mid)
        print("p 그룹 몇개 ", p)
        if p > k:
            start = mid + 1
        else:
            end = mid - 1
    return start


import sys
sys.setrecursionlimit(10**6)
# # #시험장 나누기
# l = [0] * 10005 # 왼쪽 자식 노드 번호
# r = [0] * 10005 # 오른쪽 자식 노드 번호
# x = [0] * 10005 # 시험장의 응시 인원
# p = [-1] * 10005 # 부모 노드 번호
# root = 0 # 루트
# cnt = 0

# cur : 현재 보는 노드 번호, lim : 그룹의 최대 인원 수
def dfs(cur, lim):
    global cnt
    lv = 0
    if l[cur] != -1:
        dfs(l[cur], lim)
    rv = 0 # 오른쪽 자식 트리에서 넘어오는 인원 수
    if r[cur] != -1:
        dfs(r[cur], lim)
    # 1. 왼쪽 자식 트리와 오른쪽 자식 트리에서 넘어오는 인원을 모두 합해도 lim 이하일 경우
    if x[cur] + lv + rv <= lim:
        return x[cur] + lv + rv
    # 2. 왼쪽 자식 트리와 오른쪽 자식 트리에서 넘어오는 인원 중 작은 것을 합해도 lim 이하일 경우
    if x[cur] + min(lv, rv) <= lim:
        cnt += 1 #그룹
        return x[cur] + min(lv, rv)

def solve(lim):
    global cnt
    cnt = 0
    dfs(root, lim)
    cnt += 1
    return cnt

def solution(k, num, links):
    global root
    n = len(num)
    for i in range(n):
        l[i], r[i] = links[i]
        x[i] = num[i]
        if l[i] != -1: p[l[i]] = i
        if r[i] != -1: p[r[i]] = i

    for i in range(n): #시험장의 수
        if p[i] == -1:
            root = i
            break
    st = max(x)
    en = 10 ** 8
    while st < en:
        mid = (st + en) // 2
        if solve(mid) <= k:
            en = mid
        else:
            st = mid + 1
    return st


#미로탈출
def escape_maze(self, n, start, end, roads, traps):
    INF = int(1e9)
    visited = [False] * (n + 1)
    graph = [[INF] * (n+1) for _ in range(n+1)]
    for s, v, d in roads:
        graph[s][v] = min(d, graph[s][v])
    def swap_in_trap(trap):
        for i in range(n+1):
            if graph[i][trap] != INF:
                graph[trap][i] = graph[i][trap]
                graph[i][trap] = INF
            elif graph[trap][i] != INF:
                graph[i][trap] = graph[trap][i]
                graph[trap][i] = INF

    #최소시간
    Q = []
    heapq.heappush(Q, [0, start]) #시작 거리
    while Q:#?
        dist, v = heapq.heappop(Q)
        if v == end:
            return dist
        if v in traps:
            swap_in_trap(v)

        for neighbor in range(n+1):
            if graph[v][neighbor] != INF:
                if graph[v][neighbor] < dist+graph[v][neighbor]:
                #if not visited[neighbor] or neighbor in traps:
                    visited[neighbor] = True

                    heapq.heappush(Q, [dist+graph[v][neighbor] , neighbor])

    return dist




    # 성격유형
    # 더 높은 점수, 사전 순

#계속 바꾸지 말고, bitmask 로 역방향인지, 정방향인지


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        # if not root:
        #     return 0
        # return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        # level = 0
        # q = [root]
        # while q:
        #
        #     for _ in range(len(q)):
        #         node = q.popleft()
        #         if node.left:
        #             q.append(node.left)
        #         if node.right:
        #             q.append(node.right)
        #     level += 1
        # return level
        # stack = [[root, 1]]
        # answer = 0
        # while stack:
        #     node, depth = stack.pop()
        #     answer = max(depth, answer)
        #     if node:
        #         stack.append([node.left, depth+1])
        #         stack.append([node.right, depth + 1])
        # return answer

def maximumSubarray(num): #O(n**2) ->  O(n)
    maxSum = num[0]
    sumPrefix = 0
    for n in num:
        if sumPrefix + n < 0:
            sumPrefix = 0
        sumPrefix += n
        maxSum = max(maxSum, sumPrefix)

    return maxSum


def personality(survey, choices):
     dic = collections.defaultdict(int)
     for i, c in enumerate(choices):
         if c < 4:
            dic[survey[i][0]] += 4 - c

         else:

            dic[survey[i][1]] += c % 4
     print(dic)

     answer = []
     if dic['R'] >= dic['T']:
         answer.append('R')
     else:
         answer.append('T')

     if dic['C'] >= dic['F']:
         answer.append('C')
     else:
         answer.append('F')
     if dic['J'] >= dic['M']:
         answer.append('J')
     else:
         answer.append('M')
     if dic['A'] >= dic['N']:
         answer.append('A')
     else:
         answer.append('N')
     return ''.join(answer)

def queue_sum(queue1, queue2):
    q1 = sum(queue1)
    q2 = sum(queue2)
    if (q1 + q2) % 2 != 0:
        return -1
    else:
        target = (q1 + q2) // 2
    answer = 0
    while q1 != q2 :
        if answer > int(1e7):
            return -1

        while q2 > target:
            x = queue2.popleft()
            queue1.append(x)
            q2 -= x
            q1 += x
            answer += 1
            if len(queue2) == 1 and q2 > target:
                return -1
        while q1 > target:
            x = queue1.popleft()
            queue2.append(x)
            q1 -= x
            q2 += x
            answer += 1

            if len(queue1) == 1 and q1 > target:
                return -1

    #print(queue1, queue2)

    return answer

def algocoding(alp,cop, problems):

    problems.sort(key=lambda x:[x[0], x[1], x[2], x[3]])
    print(problems)
    alpcop = [alp, cop]
    answer = 0
    for i in range(len(problems)-1):

       if i == 0 and problems[0][0] > alp:
           answer += problems[0][0] - alp
           alp = problems[0][0]
       if i == 0 and problems[0][1] > cop:
           answer+= problems[0][1] - cop
           cop = problems[0][1]

       if problems[i][2] == 0 or problems[i][4] / problems[i][2] > 1:
           answer += (problems[i+1][0] - alp)
           alp += (problems[i+1][0] - alp)
           print("1",answer)
       else:
           x = math.ceil((problems[i+1][0] - alp) / problems[i][2])
           answer += problems[i][4] * x
           alp += problems[i][2] * x
           cop += problems[i][3] * x
           print("2,",answer, alp, cop)



       if problems[i][3] == 0 or problems[i][4] / problems[i][3] > 1:

           answer = answer + problems[i+1][1] - cop
           cop += problems[i+1][1] - cop
           print("3", answer, alp, cop)
       else:
           y = math.ceil((problems[i+1][1] - cop) / problems[i][3])
           answer += problems[i][4] * y
           alp += problems[i][2] * y
           cop += problems[i][3] * y
           print("4", answer)



    return answer

def moutain(n, paths, gates, summits):
    summits.sort()
    # graph = [[0] * (n+1) for _ in range(n+1)]
    graph = collections.defaultdict(dict)
    for i, j, w in paths:
        graph[i][j] = w
        graph[j][i] = w

    Q = []


    for gate in gates:
        isV = [False] * (n + 1)
        isV[gate] = True
        heapq.heappush(Q, [0, [gate], False, isV])

    i = 0

    minnnn = int(1e9)
    answer = []
    while Q:
        #i += 1
        intensity, visited, check_summit, isV = heapq.heappop(Q)

        if intensity > minnnn:
            # print("ASdfasdfasdfasdf", min, intensity)
            break
        # if minnnn !=
        #     break

        if visited[-1] in summits:
            minnnn = intensity
            answer.append([check_summit, intensity])

        elif   minnnn != int(1e9) :
            break

        else:

            for neighbor in graph[visited[-1]].keys():

                if neighbor == visited[0] and check_summit is not False:
                    minnnn = intensity
                    answer.append([check_summit, intensity])
                    #answer.sort()
                    #return answer[0]
                if neighbor == visited[0] and check_summit is False:
                    continue
                if neighbor in gates:
                    continue
                if len(visited) > 4 and neighbor == visited[-4] and neighbor == visited[-2]:
                   continue #반복 X

                if neighbor in summits and check_summit is False:
                   # print(neighbor, "asdfasdfasdfqwerqwerqwerw")
                   # minnnn = intensity
                   # answer.append([neighbor, intensity])
                   # continue
                    isV = [False] * (n + 1)
                    b = copy.deepcopy(visited)
                    b.append(neighbor)
                    heapq.heappush(Q, [max(intensity, graph[b[-2]][neighbor]),[b[0],b[-1]] , neighbor, isV])
                if neighbor not in summits and neighbor not in visited:
                    #asd = copy.deepcopy(isV)
                    #asd[neighbor] = True
                    #print("---", neighbor, visited, isV)
                    b = copy.deepcopy(visited)
                    b.append(neighbor)
                    heapq.heappush(Q, [max(intensity, graph[b[-2]][neighbor]), b, check_summit, isV])
                # print(Q)
        # if i >15 :
        #     break

    #print("answer", answer)
    answer.sort()
    return answer[0]
    return intensity




    #intensity 최소 - 휴식없이 쉬는 시간
    #출입구 한번씩, 반드시 돌아와야 , 산봉우리 한번만 포함 반드시

def rotate(rc, operations):
    n = len(rc)
    m = len(rc[0])

    def shift(k, p):
        p %= n
        rc = copy.deepcopy(k)
        for i in range(n):
            for j in range(m):
                # if i == n - 1:
                #     rc[0][j] = k[i][j]
                # else:
                    rc[i + p // n][j] = k[i][j]
        return rc


    def rotate(rc, q):
        k = copy.deepcopy(rc)
        # for i in range(n):
        #     for j in range(m):
        i = 0
        for j in range(m):
            # if i == 0:
            if j + q > m - 1:
                k[i +  q][j] = rc[i][j]
            else:
                k[i][j + q] = rc[i][j]
        j = m - 1
        for i in range(1, n):
            if i +q > n - 1:
                k[i][j - q] = rc[i][j]
            else:
                k[i + q][j] = rc[i][j]
        i = n - 1
        for j in range(m - 1):
            if j  - q <= 0:
                k[i - q][j] = rc[i][j]
            else:
                k[i][j - q] = rc[i][j]
        j = 0
        for i in range(1, n - 1):
            if i - q >= 0:
                k[i - q][j] = rc[i][j]
            else:
                k[0][j+q] = rc[i][j]
        return k

    k = copy.deepcopy(rc)
    x = 0
    y = 0
    for i, o in  enumerate(operations):
        if i != len(operations)-1 and o =='ShiftRow' and operations[i+1] == o:
            x += 1
        if  i != len(operations)-1 and o == 'ShiftRow' and operations[i + 1] != o:
            x += 1
            k = shift(k,x)
            x = 0
            #print(k)
        if i != len(operations) - 1 and o == 'Rotate' and operations[i + 1] != o:
            y += 1
        if i != len(operations)-1 and o == 'Rotate' and operations[i + 1] != o:
            y += 1
            k = rotate(k,y)
            y = 0

        if i == len(operations):
            if x > 0:
                x += 1
                k = shift(k, x)
            if y > 0:
                y += 1
                k = rotate(k, y)
            if x == 0 and y == 0 and o == 'Rotate':
                k = rotate(k,1)
            elif x == 0 and y == 0 and o == 'ShiftRow':
                k = shift(k,1)

    return k

#def kakao():#정확도 효율성

def turing1(digits, num):
    dic = collections.defaultdict(int)
    for i, d in enumerate(digits):
        dic[d] = i

    k = 0
    answer = 0
    for n in num:
        answer += abs(k - dic[n])
        k = dic[n]

    return answer

def turing2(cards):
    answer = []
    for card in cards:
        k = collections.Counter(card)
        p = k.most_common(len(k.keys()))
        p.sort(key=lambda x:[x[1],-x[0]])
        answer.append(p[0])
    answer.sort(key=lambda x:[x[1], -x[0]])
    if answer[0][1] != 1:
        return -1
    else:
        return answer[0][0]

if __name__ == '__main__':

    # print(turing1("0123456789", "210"))
    # print(turing1("8459761203", "5439"))

    # print(turing2([[5,7,3,9,4,9,8,3,1],[1,2,2,4,4,1],[1,2,3]]))
    # print(turing2([[5,5],[1,1]]))

    #카카오 2021
    print(1 ^ 1)

    # print(escape_maze(3	1	3	[[1, 2, 2], [3, 2, 3]]	[2])) #5
    # print(escape_maze(4	1	4	[[1, 2, 1], [3, 2, 1], [2, 4, 1]]	[2, 3])) #5

    # print(divide_test(3, [12, 30, 1, 8, 8, 6, 20, 7, 5, 10, 4, 1], [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [8, 5], [2, 10], [3, 0], [6, 1], [11, -1], [7, 4], [-1, -1], [-1, -1]]))
    # print(divide_test(2,[6, 9, 7, 5], [[-1, -1], [-1, -1], [-1, 0], [2, 1]]))
    # print(divide_test(1,[6, 9, 7, 5], [[-1, -1], [-1, -1], [-1, 0], [2, 1]]))
    # print(divide_test(4,[6, 9, 7, 5], [[-1, -1], [-1, -1], [-1, 0], [2, 1]]))

    # print(rotate([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ["Rotate","ShiftRow"]))
    # print(rotate([[8, 6, 3], [3, 3, 7], [8, 4, 9]], ["Rotate", "ShiftRow", "ShiftRow"]))
    # print(rotate([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], ["ShiftRow", "Rotate", "ShiftRow", "Rotate"]))

    # print(queue_sum(collections.deque([3, 2, 7, 2]),collections.deque([4, 6, 5, 1])))
    # print(queue_sum(collections.deque([1,2,1,2]),collections.deque([1,10,1,2])))
    # print(queue_sum(collections.deque([1,1]),collections.deque([1,5])))

    # print(algocoding(10, 10, 	[[10,15,2,1,2],[20,20,3,3,4]]))
    # print(algocoding(0,0, [[0,0,2,1,2],[4,5,3,1,2],[4,11,4,0,2],[10,4,0,4,2]]))

    # print(moutain(6,[[1, 2, 3], [2, 3, 5], [2, 4, 2], [2, 5, 4], [3, 4, 4], [4, 5, 3], [4, 6, 1], [5, 6, 1]],[1, 3]	,[5]  ))
    # print(moutain(7, 	[[1, 4, 4], [1, 6, 1], [1, 7, 3], [2, 5, 2], [3, 7, 4], [5, 6, 6]], [1],[2,3,4]))
    # print(moutain(7, 	[[1, 2, 5], [1, 4, 1], [2, 3, 1], [2, 6, 7], [4, 5, 1], [5, 6, 1], [6, 7, 1]], [3,7],[1,5]))
    # print(moutain(5, 	[[1, 3, 10], [1, 4, 20], [2, 3, 4], [2, 4, 6], [3, 5, 20], [4, 5, 6]], [1,2], [5]))

    # print(personality(["AN", "CF", "MJ", "RT", "NA"],[5, 3, 2, 7, 5] ))
    # print(personality(["TR", "RT", "TR"],[7, 1, 3] ))

    # print(room(10, [1,3,4,1,3,1]))
    # print(cave(9,[[0,1],[0,3],[0,7],[8,1],[3,6],[1,2],[4,7],[7,5]],[[4,1],[8,7],[6,5]] ))

    lst = [2,5,0,3,3,3,1,5,4,2, 7]
    # print(counting_sort(lst))

    lst = [802, 95, 10, 3, 13, 3, 11, 503, 4, 2, 0]
    #print(radix_sort(lst))

    #print( 1239 // math.pow(10,2) % 10)

    lst = [12, 25, 31, 48, 54, 66, 70, 83, 95, 108]
    #print(binarySearch(lst, 83))

    import random
    lstRandom = []
    for itr in range(0,10):
        lstRandom.append(random.randrange(0,100))
    # print(lstRandom)
    # print(merge_sort(lstRandom))

    # print(longestPalindrome("cdbbd"))

    # print(twoSum([2,7,11,15],9))
    # print(trapping([0,1,0,2,1,0,1,3,2,1,2,1]))
    # print(trappingStack([0,1,0,2,1,0,1,3,2,1,2,1]))
    # print(threesum([-1, 0, 1, 2, -1, -4]))
    # print(productexcept([1,2,3,4]))
    # print(rainTrapping([0,1,0,2,1,0,1,3,2,1,2,1]))
    LP = LinkedListProblem()
    # print(print(divmod(19, 10)))
    # print(LP.isPalindrome(ListNode(1, ListNode(2, ListNode(2, ListNode(1)))))) #[1,2,2,1]
    # print(LP.isPalindrome(ListNode(1, ListNode(2, ListNode(2, ListNode(3)))))) #[1,2,2,1]
    # print(LP.mergeTwoLists(ListNode(1, ListNode(2, ListNode(4))), ListNode(1, ListNode(3, ListNode(4)))))
    # print(LP.reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))
    # print(LP.reverseLinkedList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))
    # print(LP.reverseBetween(ListNode(1,ListNode(2, ListNode(3, ListNode(4, ListNode(5))))), 2, 4))
    # print(LP.reverseBetween(ListNode(5), 1,1))
    # print(LP.addTwo(ListNode(2, ListNode(4, ListNode(3))), ListNode(5, ListNode(6, ListNode(4)))))
    # print(LP.swapPairs(ListNode(1, ListNode(2, ListNode(3, ListNode( 4))))))
    # print(LP.swapReverse(ListNode(1, ListNode(2, ListNode(3, ListNode( 4))))))
    # print(LP.evenOddList(ListNode(1,  ListNode( 2, ListNode(3, ListNode(4, ListNode(5)))))))
    # print(LP.evenOddList(ListNode(1,  ListNode( 2, ListNode(3, ListNode(4, ListNode(5, ListNode(6, ListNode(7)))))))))
    # K = Kakao()
    # print(K.escape_maze(3,1,3,[[1, 2, 2], [3, 2, 3]], [2]))
    # print(K.escape_maze(4,1,4,[[1, 2, 1], [3, 2, 1], [2, 4, 1]], [2, 3]	))