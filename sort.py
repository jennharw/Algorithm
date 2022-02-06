import collections
import heapq


def binary_search(arr, key, start, end):
    if start == end:
        if arr[start] > key:
            return start
        else:
            return start + 1
    if start > end:
        return start

    #binary search
    mid = (start + end ) /2
    if arr[mid] < key:
        return binary_search(arr, key, mid + 1, end)
    elif arr[mid] > key:
        return binary_search(arr, key, start, mid - 1)
    else:
        return key


def insertion_sort(l, left=0, right=None):
    if right is None:
        right = len(l)-1
    for i in range(left+1, right+1):
        key = l[i]

        j = binary_search(l, key, left+1, right+1)
        # j = i -1
        # while j >= left and l[j] > key:
        #     l[j+1] = l[j]
        #     j -= 1 #binary 아님

        l[j+1] = key
    return l


def merge(left, right):
    a = b = 0

    cArr = []

    while a < len(left) and b < len(right):
        if left[a] < right[b]:
            cArr.append(left[a])
            a += 1
        elif left[a] > right[b]:
            cArr.append(right[b])
            b += 1
        else:
            cArr.append(left[a])
            #cArr.append(right[b])
            a += 1
            b += 1

    #if a < len(left):
    cArr.append(left[a:])

    #if b < len(left):
    cArr.append(right[b:])
    return cArr

def tim_sort(l):
    min_run = 32
    n = len(l)
    for i in range(0, n, min_run):
        insertion_sort(l, i, min((i+min_run-1), (n-1)))
    #min_run단위로 sort

    size = min_run

    while size<n:
        for s in range(0, n, size*2):
            mid = s + size - 1
            end = min((s + size*2 -1), (n - 1))

            merged = merge(left = l[s:mid+1], right= l[mid+1, end+1])
            l[s:s+len(merged)] = merged
        size *= 2
    return 1

#mergesort
#quick sort
#insertionSort



#if __name__ == '__main__':
