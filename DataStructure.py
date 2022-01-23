class Node:
    nodeNext = None
    objValue = ''
    binHead = False
    binTail = False
    def __init__(self, objValue = '', nodeNext = None, binHead = False,binTail = False):
        self.nodeNext = nodeNext
        self.objValue = objValue
        self.binHead = binHead
        self.binTail = binTail
    def getValue(self):
        return self.objValue
    def setValue(self, objValue):
        self.objValue = objValue
    def getNext(self):
        return self.nodeNext
    def setNext(self, nodeNext):
        self.nodeNext = nodeNext
    def isHead(self):
        return self.binHead
    def isTail(self):
        return self.binTail

#연결리스트
class SinglyLinkedList:
    nodeHead = ""
    nodeTail = ""
    size = 0

    def __init__(self):
        self.nodeTail = Node(binTail=True)
        self.nodeHead = Node(binHead=True, nodeNext=self.nodeTail)

    def insertAt(self, objInsert, idxInsert):
        nodeNew = Node(objValue=objInsert)
        nodePrev = self.get(idxInsert - 1)
        nodeNext = nodePrev.getNext()

        nodePrev.setNext(nodeNew)
        nodeNew.setNext(nodeNext)

        self.size = self.size + 1

    def removeAt(self, idxInsert):
        nodePrev = self.get(idxInsert - 1)
        nodeRemove = nodePrev.getNext()
        nodeNext = nodeRemove.getNext()

        nodePrev.setNext(nodeNext)

        self.size = self.size - 1
        return nodeRemove.getValue()

    def get(self, idx):
        nodeReturn = self.nodeHead
        for i in range(idx + 1):
            nodeReturn = nodeReturn.getNext()
        return nodeReturn

    def getSize(self):
        return self.size

    def printStatus(self):
        nodeCurrent = self.nodeHead
        while nodeCurrent.getNext().isTail() == False:
            nodeCurrent = nodeCurrent.getNext()
            print(nodeCurrent.getValue())
        print(" ")


# Data Structure
# 1. ArrayList
x = ['a', 'b', 'd', 'e', 'f']

#1_1. insert
idxInsert= 2
valInsert = 'c'
#x[idxInsert] = valInsert

#for i in range(idxInsert, len(x)):
#    temp = x[idxInsert],    x[idxInsert+1] = temp,    idxInsert += 1
y = list(range(6))
for i in range(0,idxInsert):
    y[i] = x[i]
y[idxInsert] = valInsert
for i in range(idxInsert, len(x)):
    y[i+1] = x[i]

#1_2. delete
idxDelete = 3

z = list(range(5))
for i in range(0,idxDelete):
    z[i] = y[i]
for i in range(idxDelete+1, len(y)):
    z[i-1] = y[i]

# 2. LinkedList
list1 = SinglyLinkedList()
list1.insertAt("a",0)
list1.insertAt("b",1)
list1.insertAt("c",2)
list1.insertAt("d",3)
list1.insertAt("e",4)
#list1.printStatus()
list1.insertAt("f",2)
#list1.printStatus()
list1.removeAt(3)
#list1.printStatus()

class Stack(object):
    lstInstance = SinglyLinkedList()

    def pop(self):
        return self.lstInstance.removeAt(0)
    def push(self, value):
        self.lstInstance.insertAt(value,0)

#3. Stack
stack = Stack()
stack.push("c")
stack.push("e")

#print(stack.pop())

#4 Queue


class Queue:
    lstInstance = SinglyLinkedList()

    def dequeue(self):
        return self.lstInstance.removeAt(0)

    def enqueue(self, value):
        self.lstInstance.insertAt(value, self.lstInstance.getSize())

    def isEmpty(self):
        return self.lstInstance.getSize() == 0



queue = Queue()
queue.enqueue("e")
queue.enqueue("f")
queue.enqueue("g")

#print(queue.dequeue())

class PriorityNode:
    value = ""
    priority = -1
    def __init__(self, value, priority):
        self.value = value
        self.priority = priority

    def getValue(self):
        return self.value
    def getPriority(self):
        return self.priority


from PriorityNode import PriorityNode
from SinglyLinkedList import SinglyLinkedList


class PrioirtyQueue:
    list = ''

    def __init__(self):
        self.list = SinglyLinkedList()

    # sorted implementation
    def enqueue(self, value, prioirty):

        idxInsert = 0

        for i in range(self.list.getSize()):
            node = self.list.get(i)
            if self.list.getSize() == 0:
                idxInsert = i
                break
            ##LinkedListNode <- PrioirtyNode

            if node.getValue().getPriority() > prioirty:
                idxInsert = i + 1
            else:
                idxInsert = i
                break

        self.list.insertAt(PriorityNode(value, prioirty), idxInsert)

    def dequeue(self):
        return self.list.removeAt(0).getValue()


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


class TreeNode:
    nodeLHS = None
    nodeRHS = None
    nodeParent = None
    value = None

    def __init__(self, value, nodeParent):
        self.value = value
        self.nodeParent = nodeParent

    def getLHS(self):
        return self.nodeLHS

    def getRHS(self):
        return self.nodeRHS

    def getValue(self):
        return self.value

    def getParent(self):
        return self.nodeParent

    def setLHS(self, LHS):
        self.nodeLHS = LHS

    def setRHS(self, RHS):
        self.nodeRHS = RHS

    def setValue(self, value):
        self.value = value

    def setParent(self, nodeParent):
        self.nodeParent = nodeParent


class BinarySearchTree:
    root = None

    def __init__(self):
        pass

    def insert(self, value, node=None):
        if node is None:
            node = self.root

        if self.root is None:
            self.root = TreeNode(value, None)
            return

        if value == node.getValue():
            return
        if value > node.getValue():
            if node.getRHS() is None:
                node.setRHS(TreeNode(value, node))
            else:
                self.insert(value, node.getRHS())
        if value < node.getValue():
            if node.getLHS() is None:
                node.setLHS(TreeNode(value, node))
            else:
                self.insert(value, node.getLHS())
        return

    def search(self, value, node=None):
        if node is None:
            node = self.root
        if value == node.getValue():
            return True
        if value < node.getValue():
            if node.getLHS() is None:
                return False
            else:
                self.search(value, node.getLHS())
        if value > node.getValue():
            if node.getRHS() is None:
                return False
            else:
                self.search(value, node.getRHS())

    def delete(self, value, node=None):
        if node is None:
            node = self.root
        if node.getValue() < value:
            return self.delete(value, node.getRHS())

        if node.getValue() > value:
            return self.delete(value, node.getLHS())

        if node.getValue() == value:
            # childeren 2
            if node.getLHS() is not None and node.getRHS() is not None:
                minNode = self.findMin(node.getRHS())
                node.setValue(minNode.getValue())
                self.delete(minNode.getValue(), node.getRHS())
                return
            parent = node.getParent()
            if node.getLHS() is not None:
                if node == self.root:
                    self.root == node.getLHS()
                elif parent.getLHS() == node:
                    parent.setLHS(node.getLHS())
                    node.getLHS().setParent(parent)
                    return
                else:
                    parent.setRHS(node.getLHS())
                    node.getLHS().setParent(parent)
                return
            if node.getRHS() is not None:
                if node == self.root:
                    self.root == node.getRHS()
                elif parent.getLHS() == node:
                    parent.setLHS(node.getRHS())
                    node.getRHS().setParent(parent)
                else:
                    parent.setRHS(node.getRHS())
                    node.getRHS().setParent(parent)
                return
            if node == self.root:
                self.root = None
            elif parent.getLHS() == node:
                parent.setLHS(None)
            else:
                parent.setRHS(None)
            return

    def findMax(self, node=None):
        if node is None:
            node = self.root
        if node.getRHS() is None:
            return node
        return self.findMax(node.getRHS())

    def findMin(self, node=None):
        if node is None:
            node = self.root
        if node.getLHS() is None:
            return node
        return self.findMin(node.getLHS())

    # BFS queue
    def traverseLevelOrder(self):
        ret = []
        Q = Queue()
        Q.enqueue(self.root)
        while not Q.isEmpty():
            node = Q.dequeue()
            if node is None:
                continue
            ret.append(node.getValue())
            if node.getLHS() is not None:
                Q.enqueue(node.getLHS())
            if node.getRHS() is not None:
                Q.enqueue(node.getRHS())
        return ret

    # DFS stack
    def traverseInOrder(self, node=None):
        ret = []
        if node is None:
            node = self.root

        if node.getLHS() is not None:
            ret = ret + self.traverseInOrder(node.getLHS())
        ret.append(node.getValue())
        if node.getRHS() is not None:
            ret = ret + self.traverseInOrder(node.getRHS())
        return ret

    def traversePreOrder(self, node=None):
        if node is None:
            node = self.root
        ret = []
        ret.append(node.getValue())
        if node.getLHS() is not None:
            ret = ret + self.traversePreOrder(node.getLHS())
        if node.getRHS() is not None:
            ret = ret + self.traversePreOrder(node.getRHS())
        return ret

    def traversePostOrder(self, node=None):
        if node is None:
            node = self.root
        ret = []
        if node.getLHS() is not None:
            ret = ret + self.traversePostOrder(node.getLHS())
        if node.getRHS() is not None:
            ret = ret + self.traversePostOrder(node.getRHS())
        ret.append(node.getValue())
        return ret

# priority queue