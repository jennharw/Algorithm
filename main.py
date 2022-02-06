# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Algorithm
# https://teamsparta.notion.site/Chapter-2-W2-W5-740098f7060548d0a9c40703f915875c

import collections
import heapq
import math
import re
import sys


#6장 문자열 조작
class stringProblem:
    # Use a breakpoint in the code line below to debug your script.
    # Press Ctrl+F8 to toggle the breakpoint.

    #4장, p.99, bigO, 자료구조

    #분할상환분석(Amortized Analysis) - 알고리즘 복잡도 계산시, 최악보다는 분할 상한방법, ex 동적배열의 doubliing ; https://www.notion.so/5a6d0c9ca51844fbb709b83b619dd465 (유진환님 자료 참고)

    #병렬화 (numpy, pandas)

    # 자료형 - float, int, bool, set, dict, str, tuple, list 등 = Object(python)
    # list 가변 point하기 (복사하려면 deep copy)
    # str,tuple  불변, 참조를 다시

    # 0114-15 20pm 강의; ch5, 6 문자열, 7 배열
    #개념 String, String manipuliation - 조작, 처리, 전처리
    s: str = "hello" #type hint
    type(s)
    s[-1] #index, slicing
    s[1:len(s)]

    "s" in "str"
    s.index("e")

    li = ['1','2','3','4']
    s1 = "".join(li)
    #split

    #char[] - {a, b, c} , string - abc 배열이 아님 - 불변성 메모리 관리 때문

    #1번 문제 https://leetcode.com/problems/valid-palindrome/
    # Palindrome p.138
    #https://colab.research.google.com/drive/16ilffjqI481o7yG2gctxZbwMH3WT3HX_?usp=sharing

    #1 Palindrome
    #1) list
    def isPalindrome(self, words):
        strs = []
        for char in words:
            if char.isalnum():  # alphabet + number
                strs.append(char.lower())
        while len(strs) > 1:
            if strs.pop(0) != strs.pop():
                return False
        return True
    #O(n**2) O(n) pop(0)인경우, deque 사용하자
    #2) deque pop(0)가 O(n)인데 반해 popleft() O(1) 이므로
    def isPalindromeDeque(self, s):
        strs = collections.deque()

        # preprocess the input string (preprocessing)
        for char in s:
            if char.isalnum():
                strs.append(char.lower())

        # decide whether the input string is palindrome or not (main algorithms)
        while len(strs) > 1:
            # 데크에서 strs.popleft()은 O(1)
            # Deque: Doubly-ended Queue
            if strs.popleft() != strs.pop():
                return False
            return True

    #3)
    def isPalindromeRe(self, s):
        # preprocess the input string (preprocessing)
        s = s.lower()
        s = re.sub('[^a-z0-9]', '', s)

        # decide whether the input string is palindrome or not (main algorithms)
        #print(s[::-1])
        return s == s[::-1]  # 슬라이싱



    def isPalindromeWhile(self, s): #    continue break보다
        s = s.lower()  # return the string which all characters consisted of s are lower letters
        s = re.sub('[^a-z0-9]', '', s)  # non-alphanumeric characters remove in the input string

        # decide whether the input string is palindrome or not (main algorithms)
        i = 0
        j = - i - 1
        while True:
            # exit point could be change because of the non-alphanumeric characters
            # so, we will use the preprocessed string with regex (regular expression)
            # before deciding whether the input string is palindrome or not
            if i == len(s) // 2 or j == len(s) // 2:
                break

            if not s[i].isalnum():
                i += 1
                continue

            if not s[j].isalnum():
                j -= 1
                continue

            if s[i] != s[j]:
                return False

            i += 1
            j -= 1

        return True

    #2번 문제 Reverse String
    #two pointer
    def reverseString(self, s):
        #for 문으로 swap 이 아니라 two pointer 사용
        left, right = 0, len(s) - 1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return s
    #2)
    def reverseStringPython(self, s):
        return s.reverse()

    #3번 문제 로그파일 재정렬 문자로 구성된 로그가 앞, 문자가 동일한 경우 식별자 순, 숫자로그는 그대로
    #숫자인지, 문자인지 확인 / 문자 식별자, 숫자로그 그대로
    def reorderLogFiles(self, logs):
        letters, digits = [], []
        for log in logs:
            if log.split()[1].isdigit():
                digits.append(log)
            else:
                letters.append(log)
        letters.sort(key = lambda x: (x.split()[1:], x.split()[0])) #글자로, 글자가 같다면 식별자로
        return letters + digits

    #4번 문제 가장흔한 단어
    def mostCommonWord(self, sentence, banned):
        #sentence split 해서 count 보다 -> collections.Counter사용하기
        #dict에서 count하기 dict[word] += 1
        words = [word for word in re.sub(r'[^\w]', ' ', sentence).lower().split() if word not in banned]
        counts = collections.Counter(words)
        return counts.most_common(1)[0][0]

    #5번 문제 Group Anagrams
    #dict 에 list 넣어 group 화
    def groupAnagrams(self, wordlist):
        anagrams = collections.defaultdict(list) #dict에 list넣을 수 있음
        for word in wordlist:
            anagrams[''.join(sorted(word))].append(word)
        return list(anagrams.values())

    #6번 문제 가장 긴 팰린드롬 문자열
    def longestPalindrome(self, s):
        def expand(left, right):
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -= 1
                right += 1 #palindrome
            return s[left+1:right]
        if len(s) < 2 or s == s[::-1]:
             return s

        result = ""

        for i in range(len(s)-1):
            result = max(result,
                         expand(i, i+1),
                         key=len)

        return result

    #지금까지 dict, for(bruteForce) 만 써왔는데
    # defaultDict(default값 이 있고, list도 넣을 수 있는 걸 잘 활용하자), orderedDict, Counter
    # 또한 투포인터 left, right를 적극 활용
    # def 안에 def


#7장 배열
class arrangementProblem: #배열
    #07) 두수의 합
    def twoSum(self, nums, target):
        #bruteForce O(n**2)
        #dict O(n)
        num_maps = {}
        for i, num in enumerate(nums):
            num_maps[num] = i
        #키와 값을 바꿔서dict에 저장
        for i, num in enumerate(nums):
            if target - num in num_maps and i != num_maps[target-num]:
                return [i, num_maps[target-num]]

    def twoSumTwoPointer(self, nums, target):
        # sort list, nums.sort ; 그러나 이경우, index를 잃어버려 다시 찾아야 (정렬 값이 아님)
        # two pointer란 왼쪽 포인터와 오른쪽 포인터의 합이 타겟보다 크면 right-1, 작으면 left+1
        left, right = 0, len(nums)-1
        while not left == right:
            if nums[left] + nums[right] < target:
                left += 1
            elif nums[left] + nums[right] > target:
                right -=1
            else:
                return [left, right] # O(n)

    #08)빗물 트래핑
    def trap(self, height): #빗물트래핑 -> 단순 brute force O(n**2)
        #two pointer, left, right O(n)
        if not height:
            return 0
        volume = 0
        left, right = 0, len(height)-1
        left_max, right_max = height[left], height[right]

        while left<right:
            left_max, right_max = max(height[left], left_max), max(height[right], right_max)
            if left_max <= right_max:
                volume += left_max - height[left]
                left += 1
            else:
                volume += right_max - height[right]
                right-=1
        return volume

    def trapStackPractice(self, height):
        stack = []
        volume = 0

        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()
                if not len(stack):
                    break
                distance = i - stack[-1] - 1
                waters = min(height[i], height[stack[-1]]) - height[top]
                volume += distance * waters
            stack.append(i)
        return volume

    def trapStack(self, height): #O(n) LIFO
        stack = []
        volume = 0

        for i in range(len(height)): #0, 1
            while stack and height[i] > height[stack[-1]]: #
                top = stack.pop()

                if not len(stack):
                    break
                distance = i - stack[-1] - 1
                waters = min(height[i], height[stack[-1]]) - height[top]
                volume += distance * waters
            stack.append(i) # 0
        return volume

    #09) 세수의 합
    def threeSum(self, nums): #O(n**3) brute force
        #투 포인터로 합계산 O(n**2)
        result = []
        nums.sort()
        for i in range(len(nums) - 2):
            if i > 0 and nums[i] == nums[i -1]:
                continue
            left, right = i + 1, len(nums) -1
            while left < right:
                sum = nums[i] + nums[left] + nums[right]
                if sum < 0 :
                    left += 1
                elif sum>0:
                    right -=1
                else:
                    result.append([nums[i], nums[left], nums[right]])
                    while left<right and nums[left] == nums[left+1]: #중복값있으면 스킵처리
                        left+=1
                    while left < right and nums[right] == nums[right -1]:
                        right-=1
                    left +=1
                    right -=1
        return result
#
    #10) 배열 파티션
    def arrayPairSum(self, nums):
        # sum = 0
        # pair = []
        # nums.sort
        # #
        # for n in nums:
        #     pair.append(n)
        #     if len(pair) ==2:
        #         sum += min(pair)
        #         pair = []
        # for i, n in enumerate(nums):
        #     if  i % 2 ==0:
        #         sum += n
        # return sum

        return sum(sorted(nums)[::2])

    #11) 자신을 제외한 배열의 곱
    def productExceptSelf(self, nums): #O(n)
        out = []
        p = 1
        for i in range(len(nums)):
            out.append(p)
            p = p * nums[i]

        p = 1
        for i in range(len(nums)-1, -1, -1):
            out[i] = out[i] * p
            p = p * nums[i]
        return out

    #12) 주식
    def maxProfit(self, prices):
        #bruteforce -> o(n**2) -> O(n)
        #최댓값 갱신해서 O(n)
        profit = 0
        min_price = sys.maxsize
        for price in prices:
            min_price = min(min_price, price)
            profit = max(profit, price-min_price)
        return profit

#8장) 연결리스트

    #Definition for singly - linked list. 삽입 쉬움
# /**
#  * Definition for singly-linked list.
#  * struct ListNode {
#  *     int val;
#  *     ListNode *next;
#  *     ListNode() : val(0), next(nullptr) {}
#  *     ListNode(int x) : val(x), next(nullptr) {}
#  *     ListNode(int x, ListNode *next) : val(x), next(next) {}
#  * };
# #  */
#
# class Node:
#     nodeNext = None
#     objValue = ''
#     def __init__(self, objValue = '', nodeNext = None):
#         self.next = nodeNext
#         self.val = objValue
#
# class SinglyLinkedList:
#     nodeHead = ""
#     nodeTail = ""
#     size = 0
#
#     def __init__(self):
#         self.nodeTail = Node(binTail=True)
#         self.nodeHead = Node(binHead=True, nodeNext=self.nodeTail)

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedListProblem:
    #13)
    def isPalindrome(self, head):
        rev = None
        slow = fast = head
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev, slow.next #역순 , runner활용
        if fast:
            slow = slow.next

        while rev and rev.val == slow.val:
            slow, rev = slow.next, rev.next
        return not rev

    #14)
    def mergeTwoLists(self, l1 : ListNode, l2:ListNode) -> ListNode: #재귀
        #default조건
        if (not l1) or(l2 and l1.val > l2.val):
            l1, l2 = l2, l1
        if l1:
            l1.next = self.mergeTwoLists(l1.next, l2)

        return l1

    #)15
    def reverseList(self, head):
        def reverse(node, prev = None):
            if not node:
                return prev
            next, node.next = node.next , prev
            return reverse(next, node)
        return reverse(head)

    #반복이 공간복잡도, 나고, 실행속도 빠르다
    def reverseListWhile(self, head):
        node, prev = head, None
        while node:
            next, node.next = node.next, prev
            prev, node = node, next
        return prev

    #)16
    def addTwoNumbers(self):
        #연결리스트 뒤집고, 리스트, 연결리스트, 더하고, reverse
        def reverseList(head):
            node, prev = head, None
            while node:
                next, node.next = node.next, prev
                prev, node = node, next
            return prev

        def toList(node):
            list = list()
            while node:
                list.append(node.val)
                node = node.next
            return list

        def toReverseList(result):
            prev  = None
            for r in result:
                node = ListNode(r)
                node.next = prev
                prev = node
            return node
        def addTwoNumbers(l1, l2):
            a = toList(l1)
            b = toList(l2)
            result = int(''.join(str(e)for e in a)) + int(''.join(str(e) for e in b))
            return toReverseList(result)

    def addTwoNumbersFullAdder(self, li, l2):
        root = head = ListNode(0)
        carry = 0
        while l1 or l2 or carry:
            sum = 0
            if l1:
                sum += l1.val
                l1 = l1.next
            if l2:
                sum += l2.val
                l2 = l2.next
            #몫(자리올림수)와 나머지 (값) 계산
            carry, val = divmod(sum + carry, 10)
            head.next = ListNode(val)
            head = head.next
        return root.next

    #)17
    def swapPairsRecursive(self ,head):
        #pair단위로 swap
        #재귀
        if head and head.next:
            p = head.next
            head.next = self.swapPairs(p.next)
            p.next = head
            return p
        return head

    def swapPairsWhile(self, head):
        #값을 swap, 연결도 바꿔야
        root = prev = ListNode(None)
        prev.next= head
        while head and head.next:
            b = head.next
            head.next = b.next

            #가리키기
            prev.next = b

            head = head.next
            prev = prev.next.next
        return root.next

    def swapPairs(self, head): #값만 변경
        cur = head
        while cur and cur.next:
            cur.val, cur.next.val = cur.next.val, cur.val
            cur= cur.next.next
        return head

    #) 18
    def oddEvenList(self, head):
        if head is None:
            return None
        odd = head
        even = head.next
        even_head = head.next

        while even and even.next:
            odd.next, even.next = odd.next.next,  even.next.next
            odd, even = odd.next, even.next
        odd.next = even_head
        return head

    #) 19
    def reverseBetween(self, head, m, n):
        if not head or m == n:
            return head

        root = start = ListNode(None)
        root.next = head
        #start, end 지정
        for _ in range(m - 1):
            start = start.next
        end = start.next

        for _ in range(n, m):
            tmp, start.next, end.next = start.next, end.next, end.next.next
            start.next.next = tmp
        return root.next

# tuple
# swap

#9장 Stack
class Stack:
    def __init__(self):
        self.top = None
    def push(self, value):
        self.top = ListNode(value, self.top)

    def pop(self, value):
        if self.top is None:
            return None
        topNode = self.top
        self.top = self.top.next
        return topNode.val
    def is_empty(self):
        return self.top is None

class StackProblem:

    #)20
    def validParentheses(self, s):
        stack = []
        table = {
            ')':'(',
            '}':'{',
            ']':'[',
        }
        for char in s:
            if char not in table:
                stack.append(char)
            elif not stack or table[char] != stack.pop():
                return False

        return len(stack) == 0

    #) 21
    def remove_duplicate_letters_recursion(self, s):
        # ? return sorted(set(s))
        for char in sorted(set(s)):
            suffix = s[s.index(char):]
            if set(s) == set(suffix):
                return char + self.remove_duplicate_letters_recursion(suffix.replace(char, ''))
        return ''


    def remove_duplicate_letters_stack(self, s):
        counter, seen, stack = collections.Counter(s), set(), []

        for char in s:
            counter[char] -= 1
            if char in seen:
                continue
            while stack and char < stack[-1] and counter[stack[-1]] >0 :
                seen.remove(stack.pop())
            stack.append(char)
            seen.add(char)
        return ''.join(stack)

    # ) 22
    def dailyTemparature(self, l):
        stack = []
        daily = [0] * len(l)
        for i, n in enumerate(l):

            while stack and n > l[stack[-1]]:
                lst = stack.pop()
                daily[lst] = i - lst

            stack.append(i)
        return daily


    #https://www.acmicpc.net/problem/2164 시간, 공간 제한
    def card2(self, cardN):
        cardQueue = collections.deque()
        for i in range(1,cardN+1):
            cardQueue.append(i)
        i = 0
        while len(cardQueue) >1:
            if i % 2 ==0:
                cardQueue.popleft()
            else:
                cardQueue.append(cardQueue.popleft())
            i += 1

        return cardQueue.popleft()

#23) implmentStackUsingQueue
class Mystack:
    def __init__(self):
        self.q = collections.deque()
    def push(self, x):
        #가장 왼쪽으로
        self.q.append(x)
        #재정렬
        for _ in range(len(self.q)-1):
            self.q.append(self.q.popleft())
    def pop(self):

        #마지막
        return self.q.popleft()
    def top(self):
        return self.q[0]
    def empty(self):
        return len(self.q) == 0

#24) implementQueueUsingStack
class MyQueue:
    def __init__(self):
        self.input = []
        self.output = []
    def push(self, x):
        self.input.append(x)
        #
    def pop(self):
        self.peek()
        return self.output.pop()
    def peek(self):
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())
        return self.output[-1]
    def empty(self):
        return self.input == [] and self.output == []

#25)
class MyCircularQueue:
    def __init__(self, size):
        self.front = 0
        self.rear = 0
        self.size = size
        self.queue = [None] * size #collections.deque()

    #enqueue
    def enQueue(self, x):
        if self.queue[self.rear] is None:
            self.queue[self.rear] = x
            self.rear = (self.rear + 1) % self.size
            return True
        else:
            return False

    def deQueue(self):
        if self.queue[self.front] is None:
            return False
        else:
            self.queue[self.front] = None
            self.front = (self.front + 1) % self.size
            return True
    def Front(self):
        return -1 if self.queue[self.front] is None else self.queue[self.front]
    def Rear(self):
        return -1 if self.queue[self.rear-1] is None else self.queue[self.rear-1]
    def isEmpty(self):
        return self.front == self.rear and self.queue[self.front] is None
    def isFull(self):
        return self.front == self.rear and self.queue[self.front] is not None



# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, left=None, right = None):
        self.val = val
        self.left = left
        self.right = right

#10장 우선순위 Queue
#)26
class MyCircularQueue:
    def __init__(self, size):
        self.head , self.tail = ListNode(None), ListNode(None)
        self.k, self.len = size, 0
        self.head.right , self.tail.left = self.tail, self.head

    def _add(self, node : ListNode, new: ListNode):
        n = node.right
        node.right = new
        new.right, new.left  = n, node
        n.left = new

    def insertFront(self, value):
        if self.len == self.k :
            return False
        self.len += 1
        self._add(self.head, ListNode(value))
        return True

    def insertLast(self, value):
        if self.len == self.k :
            return False
        self.len += 1
        self._add(self.tail.left, ListNode(value))
        return True


    def _del(self, node: ListNode):
        n = node.right.right
        node.right = n
        n.left = node

    def deleteFront(self):
        if self.len == 0:
            return False
        self.len -= 1
        self._del(self.head)
        return True

    def deleteLast(self, value):
        if self.len == 0:
            return False
        self.len -= 1
        self._del(self.tail.left.left)
        return True

    def getFront(self):
        return self.head.right.val if self.len else -1
    def getRear(self):
        return self.tail.left.val if self.len else -1

    def isFull(self):
        return self.len == self.k

    def isEmpty(self):
        return self.len == 0



class PriorityQueueProblem: #k개

    #)27
    def mergeKSortedList(self, l):
        root= result = ListNode(None)
        heap = []
        #heap 에 저장
        for i in range(len(l)):
            if l[i]:
                heapq.heappush(heap, (l[i].val, i, l[i]))
        #heap 추출해서 wjwkd
        while heap:
            node = heapq.heappop(heap)
            idx = node[1]
            result.next = node[2]
            result = result.next
            if result.next:
                heapq.heappush(heap, (result.next.val, idx, result.next))
        return root.next


class ListNode:
    def __init__(self, key = None, value = None):
        self.key = key
        self.value = value
        self.next = None

#11장 해시 테이블
#design hashmap - chaining
#28)
class MyHashMap:
    def __init__(self):
        self.size = 100
        self.table = collections.defaultdict(ListNode)

    def put(self, key, value):
        index = key % self.size
        if self.table[index].value is None:  #defaultDict 자칫 True 가 안될 수도
            self.table[index] = ListNode(key, value)
            return

        p = self.table[index]
        while p:
            if p.key ==  key:
                p.value = value
                return
            if p.next is None:
                break
            p = p.next
        p.next= ListNode(key,value)

    def get(self, key):
        index = key % self.size
        if self.table[index].value is None:
            return -1
        p = self.table[index]
        while p:
            if p.key == key:
                return p.value
            p = p.next
        return -1

    def remove(self, key):
        index = key % self.size
        if self.table[index].value is None:
            return -1
        p = self.table[index]
        if p.key == key:
            self.table[key] = ListNode() if p.next is None else p.next

        prev = p #연결리스트 삭제
        while p:
            if p.key == key:
                prev.next = p.next
                return
            prev, p = p, p.next


def test_hashtable():
    ht = MyHashMap()

    ht.put(1, 1)
    ht.put(2, 2)
    assert ht.get(1) == 1
    assert ht.get(3) == -1

    ht.put(2, 1)
    assert ht.get(2) == 1

    ht.remove(2)
    assert ht.get(2) == -1


def test_birthday_problem():
    import random
    TRIALS = 100000
    same_birthdays = 0

    for _ in range(TRIALS):
        birthdays = []
        for i in range(23):
            birthday = random.randint(1, 365)
            if birthday in birthdays:
                same_birthdays += 1
                break
            birthdays.append(birthday)

    print(f"{same_birthdays / TRIALS * 100}%")

class HashTableProblems:
    #29)
    def jewelsAndStones(self, jewels:str, stones:str): #여러 방법, hash table, defaultdict, Counter, ~
        freqs = {}
        for char in stones:
            if char not in freqs:
                freqs[char] = 1
            else:
                freqs[char] += 1

        count = 0
        for char in jewels:
            if char in freqs[char]:
                count += freqs[char]
        return count

    def jewelsAndStonesDefaultDict(self, j, s):
        freqs = collections.defaultdict(int)
        count = 0
        for char in s:
            freqs[char] += 1
        for char in j:
            count += freqs[char]
        return count
    def jewelsAndStonesCounter(self, j, s:str):
        freqs = collections.Counter(s)
        count = 0
        for char in j:
            count += freqs[char]
        return count
    def jewelsAndStones(self, J, S):
        return sum(s in J for s in S)
    #30 )
    def longestsubstring(self, s:str):
        #sliding, two pointer
        long = {}
        max_length = count = 0
        for index, char in enumerate(s): #right
            if char not in long and count <= long[char]:

                count =  long[char] +  1

            else:#최대문자길이
                max_length = max(max_length, index-count + 1)

            long[char] = index
        return max_length

    #)31
    def topKfrequestHeapq(self, nums, k):
        #priorityQueue heapq
        freqs = collections.Counter(nums)
        freqs_heap = []
        for f in freqs:
            heapq.heappush(freqs_heap, (-freqs[f], f))
        topk = list()
        for _ in range(k): #k번이상 등장
            topk.append(heapq.heappop(freqs_heap)[1]) #가장 많은 것부터
        return topk


    def topKfrequent(self, nums, k):
        return list(zip(*collections.Counter(nums).most_common(k)))[0]

        #counter


#1주차 TEST
#https://programmers.co.kr/learn/courses/30/lessons/42583?language=python3
def solution(bridge_length, weight, truck_weights):
    truck = collections.deque(truck_weights)
    truck_queing= collections.deque()
    answer = 1
    while truck:
        p = truck.popleft()
        truck_queing.append(p)
        answer += bridge_length * 1
        while truck and truck[0] <= weight-sum(truck_queing):
            truck_queing.append(truck.popleft())
            answer += 1
        truck_queing.popleft()
    return answer

#https://programmers.co.kr/learn/courses/30/lessons/42586
def solution1(progresses, speeds):
    answer = []
    pro = collections.defaultdict(int) #orderedDict
    for k, i in enumerate(list(zip(progresses, speeds))):
       pro[k] = math.ceil((100-i[0]) / i[1])

    count = collections.defaultdict(int)
    start = 0
    #two pointer, sliding
    for i in range(len(pro)):
        if pro[i] > start:
            start = pro[i]
            count[start] += 1
        else:
            count[start] += 1
    for i in count:
        answer.append(count[i])

    return answer

# 1/29 - 문자열조작, 배열, 연결리스트
#https://programmers.co.kr/learn/courses/30/lessons/60057?language=python3
def solution2(s):

    #default dict, two pointer (start ,...)

    strt_temp = dict()

    for j in range(1,len(s)//2+1):
            answer = len(s)
            strt = 0
            count = 1
            while strt < len(s):
                if s[strt:strt+j] == s[strt+j:strt+j+j]:

                    answer = answer-j
                    strt = strt+j
                    count += 1

                else:
                    if count != 1:
                        answer += 1
                    count = 1
                    strt = strt + j

            strt_temp[j] = answer

    return min(strt_temp.values())

# 1/31 - 스택, 큐, 해시테이블
#https://leetcode.com/problems/largest-rectangle-in-histogram/
def solution3(heights): #n**2
    answer = 0
    # for i in range(len(heights)):
    #     width = 1
    #     while width < len(heights):
    #         if (i+width< len(heights)-1) and (heights[i] < heights[i+width]):
    #             # 오른 쪽 체크
    #             width += 1
    #         if (i-width> 0) and (heights[i] < heights[i-width]):
    #             width += 1
    #             # 왼쪽 체크
    #         else:
    #             answer = max(heights[i] * width, answer)
    #             break

    # 스택 ->
    stck = []
    for i in range(len(heights)):
        stck.append(heights[i])
        width = 1
        while stck:
            if (i+width< len(heights)-1) and stck[-1] < heights[i+width]:
                    # 오른 쪽 체크
                    width += 1
            else:
                    answer = max(heights[i] * width, answer)
                    stck.pop()
    return answer

#2주차 TEST
#https://programmers.co.kr/learn/courses/30/lessons/49994
def solution4(dirs): #bfs(recursion, 스택), dfs (queue)
    visited = []
    dirs_dict = {
        "U":(1, 0),
        "D":(-1, 0),
        "R":(0, 1),
        "L":(0,-1)
    }
    x, y = 0, 0
    for dir in dirs:
        if -5 <= x + dirs_dict[dir][0] <= 5 and -5 <= y + dirs_dict[dir][1] <= 5:
            x, y = x + dirs_dict[dir][0], y + dirs_dict[dir][1]
            if (x,y,dir) not in visited:
                visited.append((x,y, dir))
    return len(visited)



#https://programmers.co.kr/learn/courses/30/lessons/43165
def solution5(numbers, target): #조합? dfs, backtracking?
    count = []
    result = []

    def dfs_m(i, pth):
        if i == len(numbers)-1:

            pth.append(-numbers[i])

            if sum(pth)==target:
                count.append(pth)

            result.append(pth[:])
            return
        else:
            pth.append(-numbers[i])
            dfs_p(i + 1, pth[:])
            dfs_m(i+1, pth[:])

    def dfs_p(i, pth):
        if i == len(numbers)-1:

            pth.append(numbers[i])

            if sum(pth)==target:
                count.append(pth)

            result.append(pth[:])
            return
        else:

            pth.append(numbers[i])
            dfs_p(i+1, pth[:])
            dfs_m(i+1, pth[:])


    dfs_p(0, [])
    dfs_m(0, [])

    # print(result)
    # for i in result:
    #     if sum(i) == target:
    #         count+=1
    # print(count)


    return len(count)


#3주차 TEST
#https://programmers.co.kr/learn/courses/30/lessons/42626?language=python3
def solution6(scoville, K):
    answer = 0
    #heapq

    heapq.heapify(scoville)
    #print(heapq.heappop(scoville))
    p = heapq.heappop(scoville)
    while p < K:
        try:
            answer += 1
            t = p + 2 * heapq.heappop(scoville)
            heapq.heappush(scoville, t)

            p = heapq.heappop(scoville)
        except IndexError:
            return -1

    return answer
#https://programmers.co.kr/learn/courses/30/lessons/17686?language=python3
import re

def solution7(files):
    answer = []
    c = re.compile('[a-zA-Z]+')
    n = re.compile('[0-9]+')
    new_files = []

    # for index, file in enumerate(files):
    #     new_files.append((c.findall(file.lower())[0], n.findall(file)[0]), file)

    # new_files.sort(key = lambda x:(x[0], int(x[1]))) #차선 정렬
    # for file in new_files :
    #     answer.append(file[2])


    indexes = []
    new_files_n = []
    stck = collections.deque()
    for index, file in enumerate(files):
        heapq.heappush(new_files,(c.findall(file.lower())[0], index))

    while new_files:
        (char, i) = heapq.heappop(new_files)
        if (stck and char == stck[-1][0]) or len(stck) == 0:
            heapq.heappush(new_files_n, (int(n.findall(files[i])[0]), i))
        else:
            heapq.heappush(new_files_n, (int(n.findall(files[i])[0]), i))
            while new_files_n:
                (char, i) = heapq.heappop(new_files_n)
                indexes.append(i)
        stck.append((char, i))

    while new_files_n:
        (char, i) = heapq.heappop(new_files_n)
        indexes.append(i)

    for i in indexes:
        answer.append(files[i])


    #차선정렬
    #     result = []
    #     for file in files:
    #         headIdx = 0
    #         head = ''
    #         number = ''
    #         tail = ''
    #         for i in range(len(file)):

    #             if not headIdx and file[i].isdigit():
    #                 head = file[:i]
    #                 headIdx = i

    #             if headIdx and not file[i].isdigit():
    #                 number = file[headIdx:i]
    #                 tail = file[i:]
    #                 result.append([head, number, tail])
    #                 break

    #             if headIdx and i==len(file)-1 and number=='':
    #                 number = file[headIdx:]
    #                 result.append([head, number])

    #     print(f'{result}')

    #     result = sorted(result, key = lambda x:(x[0].lower(), int(x[1])))

    #     print(f'{result}')

    #     return [''.join(file) for file in result]
    return answer



##BackJoon + 이코테
# stack
# - [https://www.acmicpc.net/problem/9012](https://www.acmicpc.net/problem/9012)
# - [https://www.acmicpc.net/problem/1874](https://www.acmicpc.net/problem/1874)
#
# queue
# - [https://www.acmicpc.net/problem/2164](https://www.acmicpc.net/problem/2164)
# - [https://www.acmicpc.net/problem/1966](https://www.acmicpc.net/problem/1966)
#
# hash
# - [https://www.acmicpc.net/problem/1920](https://www.acmicpc.net/problem/1920)
# - [https://www.acmicpc.net/problem/17219](https://www.acmicpc.net/problem/17219)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(solution6([1, 2, 3, 9, 10, 12], 7))
    print(solution7(["img12.png", "img10.png", "img02.png", "img1.png", "IMG01.GIF", "img2.JPG"]))
    print(solution7(["F-5 Freedom Fighter", "B-50 Superfortress", "A-10 Thunderbolt II", "F-14 Tomcat"]))
    print(solution7(["muzi1.txt", "MUZI1.txt", "muzi000.txt", " muzi1.TXT"]))


   # print(solution4("ULURRDLLU"))
   # print(solution4("LULLLLLLU"))
    print(solution5([1,1,1,1,1],3))
    print(solution5([4, 1, 2, 1], 4))

    # print(solution(2, 10, [7,4,5,6]))
    # print(solution(100, 100, [10]))
    # print(solution(100, 100, [10,10,10,10,10,10,10,10,10,10]))

    # print(solution1([93, 30, 55], [1, 30, 5]))
    # print(solution1([95, 90, 99, 99, 80, 99], [1, 1,1,1,1, 1]))


    # 1/31 월
    # print(solution3([2,1,5,6,2,3]))
    # print(solution3([2,4]))


    # #1/29 토
    # print(solution2("aabbaccc"))
    # print(solution2("ababcdcdababcdcd")) #2ab2cd2ab2cd, 2ababcdcd
    # print(solution2("abcabcdede"))
    # print(solution2("abcabcabcabcdededededede"))
    # print(solution2("xababcdcdababcdcd"))
    #


    #stringProblem = stringProblem()
    # print(stringProblem.isPalindromeRe("A man, a plan, a canal: Panama"))
    # print(stringProblem.reverseString(["H","a","n","n","a","h"]))
    # print(stringProblem.reorderLogFiles(["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]))
    # print(stringProblem.mostCommonWord("Bob hit a ball, the hit BALL flew far after it was hit.", ["hit"]))
    # print(stringProblem.groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))
    # print(stringProblem.longestPalindrome("cbbd"))

    # arr = arrangementProblem()
    # # print(arr.twoSum([2,7, 11,15], 9))
    # # print(arr.twoSum([2,7, 11,15], 9))
    # # print(arr.trap([0,1,0,2,1,0,1,3,2,1,2,1]))
    # print(arr.trapStackPractice([0,1,0,2,1,0,1,3,2,1,2,1]))
    # # print(arr.threeSum([-1, 0, 1, 2, -1, -4]))
    # # print(arr.arrayPairSum([1,4,3,2]))
    # # print(arr.productExceptSelf([1,2,3,4]))
    # # print(arr.maxProfit([7, 1,5,3,6,4]))
    #
    # llp = LinkedListProblem()
    # print(llp.isPalindrome(ListNode(1, ListNode(2, ListNode(2, ListNode(1)))))) #[1,2,2,1]
    # print(llp.mergeTwoLists(ListNode(1, ListNode(2, ListNode(4))), ListNode(1, ListNode(3, ListNode(4)))))
    # print(llp.reverseList(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))
    # print(llp.reverseListWhile(ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))))

    # sp = StackProblem()
    # print(sp.validParentheses("()[]{}"))
    # print(sp.remove_duplicate_letters_recursion("cbacdcbc"))
    # print(sp.remove_duplicate_letters_stack("cbacdcbc"))
    # print(sp.card2(6))
    # print(sp.dailyTemparature([73,74,75,71,69,72,76,73]))
    # pqp = PriorityQueueProblem()
    # print(pqp.mergeKSortedList([ListNode(1, ListNode(4, ListNode(5))), ListNode(1, ListNode(3, ListNode(4))), ListNode(2, ListNode(6))]))




# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
