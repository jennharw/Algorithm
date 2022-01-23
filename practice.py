import collections
import heapq




class ListNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

class myCircularDeque:
    def __init__(self, size):
        self.size = size
        self.front = ListNode(None) #0 , 계속 None
        self.rear = ListNode(None) #0 , 계속 None
        #self.table
        self.front.right , self.rear.left = self.rear, self.head
        self.len = 0

    def __add(self, node, new):
        right = node.right
        node.right = new
        new.left , new.right = node, right
        right.left= new

    def insertLast(self, val):
        if self.len == self.size:
            return False

        self.len += 1

        # left = self.rear.left
        # new =  ListNode(val=val)
        # self.rear.left = new
        # new.right, new.left = self.rear, left
        self.__add(self.rear.left, ListNode(val=val))
        return True


    def insertFront(self, val):
        if self.len == self.size:
            return False

        self.len += 1

        self.__add(self.front, ListNode(val=val))

        # right = self.front.right
        # new = ListNode(val=val)
        # self.front.right = new
        # new.left , new.right = self.front, right

        return True


    # def getRear(self):
    #
    # def deleteLast(self):
    #
    # def getFront(self):
    #
    #
    # myCircularDeque.insertLast(1); // return True
    # myCircularDeque.insertLast(2); // return True
    # myCircularDeque.insertFront(3); // return True
    # myCircularDeque.insertFront(4); // return False, the
    # queue is full.
    # myCircularDeque.getRear(); // return 2
    # myCircularDeque.isFull(); // return True
    # myCircularDeque.deleteLast(); // return True
    # myCircularDeque.insertFront(4); // return True
    # myCircularDeque.getFront(); /
#class MyCircularDeque

class ListNode:
    def __init__(self, val, next):
        self.val = val
        self.next = next

class PriorityQueueProblem:
    #27
    def mergeKsortedlist(self, lists):
        #merge sort -> ListNode , Priority Queue heapq

        heap = []
        for i in range(len(lists)):
            for j in lists[i]:
                heapq.heappush(heap, (j,i,lists[i])) #중복제거
        answer = []
        while heap:
            answer.append(heapq.heappop(heap)[0])
        return answer

    # for i in range(len(lists)):
    #     for j in lists[i]:
    #         heapq.heappush(heap, (j, i, lists[i]))  # 중복제거
    # answer = []
    # while heap:
    #     answer.append(heapq.heappop(heap)[0])




class ListNode:
    def __init__(self, key= None,val=None):
        self.key = key
        self.val = val
        self.next = Noneheap = []

class HashTableChaining:
    def __init__(self):
        self.table = collections.defaultdict(ListNode) #
        self.size = 5
    def put(self, key, value):
        #index 정하기heap = []

        index = key % self.size
        if self.table[index].val is None:
            self.table[index] = ListNode(key, value)
            return
        # 존재하면
        p = self.table[index]
        while p:
            if p.key == key: #중복 X
                p.val = value
                return
            if p.key is None:
                break
            p = p.next
        p.next = ListNode(key, value)

    def push(self, key):
        index = key % self.size

        if self.table[index] is None:
            return -1
        else: #존재
            p = self.table[index]
            while p:
                if p.key == key:
                    return p.val
                p = p.next
            return -1

    def remove(self, key):
        index = key % self.size
        if self.table[index].val is None: #없을 때
            return -1

        prev = self.table[index]
        if prev.key == key:
            self.table[index] = ListNode() if prev.next is None else prev.next
            return
        # if prev.next is None:
        #     if prev.key == key:
        #         return self.table[index] is None
        #
        p = prev.next
        while p:
            if p.key == key:
                prev.next = p.next
                return
                # if p.next is not None:
                #     return prev.next = p.next
                # else:
                #     perv.next is None
            prev, p = p, p.next

        return -1 #key 없을 때


class HashTableProblems:
    #29 jewels and stones
    def jewelsAndStonesHashTable(self, J, W): # hashTable - dict (python 에서 chaining 없음), 선형 탐사 lf < 0.8
        freqs = dict()
        for char in W:
            if char in freqs:
                freqs[char] += 1
            else:
                freqs[char] = 1
        count = 0
        for char in J:
            count += freqs[char]
        return count

    def JASDefaultDict(self, J, W):
        freqs = collections.defaultdict(int)   #defaultDict
        for char in W:
            freqs[char] += 1
        count = 0
        for char in J:
            count += freqs[char]
        return count


    def JASCounter(self, J, W):
        freqs = collections.Counter(W)
        count = 0
        for char in J:
            count += freqs[char]
        return count

    def JAS(self, J, W):
        return (sum(char in J for char in W))

    #30 longest substring without repeating characters
    def lswc(self, s:str):
        #sliding and two pointer
        max_length = 0
        count = 0
        seen = []
        for i, char in enumerate(s):
            if char in seen:
                count = 1
                seen = []
                seen.append(char)
            else:
                seen.append(char)
                count += 1
            max_length = max(max_length, count)
        return max_length

    def lswc2(self, s:str): #abcabcdbb, start
        max_length = 0
        start = 0
        seen = {}
        for i, char in enumerate(s):
            if char in seen:
                start = seen[char] + 1
            else:
                max_length = max(max_length, i - start + 1)
            seen[char] = i
        return max_length

    #31 top k frequent
    def topKfrequent(self, l, k):
        #heapq
        l_c = collections.Counter(l)
        heap = []
        topk = []
        for p in l_c:
            heapq.heappush(heap, (-l_c[p], p)) #
        #상위 2번째
        for _ in range(k):
            topk.append(heapq.heappop(heap)[1])
        return topk

    def topKfrequentPython(self, l, k):
        l_c = collections.Counter(l)
        l_c.most_common(k)
        return list(zip(*collections.Counter(l).most_common(k)))[0]

if __name__ == '__main__':
    htp = HashTableProblems()
    # print(htp.jewelsAndStonesHashTable(J = "aA", W = "aAAbbbb"))
    # print(htp.JASDefaultDict(J = "aA", W = "aAAbbbb"))
    # print(htp.JASCounter(J = "aA", W = "aAAbbbb"))
    # print(htp.JAS(J = "aA", W = "aAAbbbb"))

    # print(htp.lswc2("abcabcbb"))
    # print(htp.lswc2("bbbbb"))
    # print(htp.lswc2("pwwkew"))

    # print(htp.topKfrequent([1,1,1,2,2,3], k = 2))
    # print(htp.topKfrequentPython([1,1,1,2,2,3], k = 2))

    # ht = HashTableChaining()
    #
    # ht.put(1, 1)
    # ht.put(2, 2)
    # assert ht.push(1) == 1
    # assert ht.push(3) == -1
    #
    # ht.put(2, 1)
    # assert ht.push(2) == 1
    #
    # ht.remove(2)
    # assert ht.push(2) == -1

    pqp = PriorityQueueProblem()
    # print(pqp.mergeKsortedlist([[1,4,5], [1,3,4], [2,6]]))
    # MyCircularDeque
    # myCircularDeque = new
    # MyCircularDeque(3);
    # myCircularDeque.insertLast(1); // return True
    # myCircularDeque.insertLast(2); // return True
    # myCircularDeque.insertFront(3); // return True
    # myCircularDeque.insertFront(4); // return False, the
    # queue is full.
    # myCircularDeque.getRear(); // return 2
    # myCircularDeque.isFull(); // return True
    # myCircularDeque.deleteLast(); // return True
    # myCircularDeque.insertFront(4); // return True
    # myCircularDeque.getFront(); // return 4