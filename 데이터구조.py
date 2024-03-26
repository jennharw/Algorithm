import collections


class Node:
    def __init__(self, value, next):
        self.value = value
        self.next = next

class Stack:
    def __init__(self):
        self.last = None

    def push(self, item):
        self.last = Node(item, self.last)

    def pop(self):
        value = self.last.value
        self.last = self.last.next
        return value

stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
stack.push(4)
stack.push(5)

# for _ in range(5):
#     print(stack.pop())

graph = {
    1: [2, 3, 4],
    2: [5],
    3: [5],
    4: [],
    5: [6, 7],
    6: [],
    7: [3],
}

#그래프 순회

def recursive_dfs(v, discovered = []):
    discovered.append(v)
    for neighbor in graph[v]:
        if neighbor not in discovered:
            discovered = recursive_dfs(neighbor, discovered)
    return discovered

def iterative_dfs(start):
    stack = []
    stack.append(start)
    discovered = []

    while stack:
        next = stack.pop()
        if next not in discovered:
            discovered.append(next)
            for neighbor in graph[next]:
                stack.append(neighbor)
    return discovered


def iterative_bfs(start):
    q = collections.deque()
    q.append(start)
    discovered = [start]
    while q:
        next = q.popleft()
        for neighbor in graph[next]:
            if neighbor not in discovered:
                q.append(neighbor)
                discovered.append(neighbor)
    return discovered

# print(f'recursive dfs: {recursive_dfs(1)}')
# print(f'iterative dfs: {iterative_dfs(1)}')
# print(f'iterative bfs: {iterative_bfs(1)}')


#tree traverse
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
root = Node('F',
            Node('B',
                 Node('A'),
                 Node('D',
                      Node('C'), Node('E'))),
            Node('G',
                 None,
                 Node('I', Node('H'))
                 )
            )


#preorder
def preorder(node):
    if node is None:
        return
    print(node.val)
    preorder(node.left)
    preorder(node.right)

#inorder
def inorder(node):
    if node is None:
        return
    inorder(node.left)
    print(node.val)
    inorder(node.right)

#postorder
def postorder(node):
    if node is None:
        return
    postorder(node.left)
    postorder(node.right)
    print(node.val)

# preorder(root)
# inorder(root)
# postorder(root)


class Node:
    def __init__(self, data, left_node, right_node):
        self.data = data
        self.left_node = left_node
        self.right_node = right_node

# 전위 순회(Preorder Traversal)
def pre_order(node):
    print(node.data, end=' ')
    if node.left_node != None:
        pre_order(tree[node.left_node])
    if node.right_node != None:
        pre_order(tree[node.right_node])

# 중위 순회(Inorder Traversal)
def in_order(node):
    if node.left_node != None:
        in_order(tree[node.left_node])
    print(node.data, end=' ')
    if node.right_node != None:
        in_order(tree[node.right_node])

# 후위 순회(Postorder Traversal)
def post_order(node):
    if node.left_node != None:
        post_order(tree[node.left_node])
    if node.right_node != None:
        post_order(tree[node.right_node])
    print(node.data, end=' ')
#
# n = int(input())
tree = {}
#
# for i in range(n):
#     data, left_node, right_node = input().split()
#     if left_node == "None":
#         left_node = None
#     if right_node == "None":
#         right_node = None
#     tree[data] = Node(data, left_node, right_node)
#
# pre_order(tree['A'])
# print()
# in_order(tree['A'])
# print()
# post_order(tree['A'])

class BinaryHeap:
    def __init__(self):
        self.arrPriority = [0] * 99
        self.arrValue = [0] * 99
        self.size = 0

    #insert
    def enqueueWithPrioirty(self, value, priority):
        self.arrPriority[self.size] = priority
        self.arrValue[self.size] = value
        self.size = self.size + 1
        self.percolateUp(self.size - 1)


    #삽입
    def percolateUp(self, idxPercolate):
        if idxPercolate == 0:
            return
        parent = int((idxPercolate-1) / 2)
        if self.arrPriority[parent] < self.arrPriority[idxPercolate]:
            self.arrPriority[parent] , self.arrPriority[idxPercolate] = self.arrPriority[idxPercolate], self.arrPriority[parent]
            self.arrValue[parent], self.arrValue[idxPercolate] = self.arrValue[idxPercolate], self.arrValue[parent]

            self.percolateUp(parent)

    def dequeueWithPrioirty(self):
        if self.size == 0:
            return 0
        retPriority = self.arrPriority[0]
        retValue = self.arrValue[0]

        self.arrPriority[0] = self.arrPriority[self.size-1]
        self.arrValue[0] = self.arrValue[self.size - 1]
        self.size -= 1
        self.percolateDown(0)
        return retValue

    def percolateDown(self, idxPercolate):
        if 2 * idxPercolate + 1 > self.size:
            return
        else:
            left = idxPercolate * 2 + 1
            leftPriority = self.arrPriority[left]

        if 2 * idxPercolate + 2 > self.size:
            return
        else:
            right = idxPercolate * 2 + 2
            rightPriority = self.arrPriority[right]

        if leftPriority > rightPriority:
            biggestChild = left
        else:
            biggestChild = right
        if self.arrPriority[biggestChild] > self.arrPriority[idxPercolate]:
            self.arrPriority[idxPercolate], self.arrPriority[biggestChild] = self.arrPriority[idxPercolate], self.arrPriority[biggestChild]
            self.arrValue[idxPercolate], self.arrValue[biggestChild] = self.arrValue[biggestChild], self.arrValue[idxPercolate]
            self.percolateDown(biggestChild)

bh = BinaryHeap()
bh.enqueueWithPrioirty('Tommy',1)
bh.enqueueWithPrioirty('Lee', 2)
bh.enqueueWithPrioirty('James',3)
bh.enqueueWithPrioirty('Peter',99)

print(bh.dequeueWithPrioirty())
print(bh.dequeueWithPrioirty())
print(bh.dequeueWithPrioirty())
print(bh.dequeueWithPrioirty())