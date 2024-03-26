#!/bin/python3
import bisect
import collections
import heapq
import math
import os
import random
import re
import sys

#
# Complete the 'gridChallenge' function below.
#
# The function is expected to return a STRING.
# The function accepts STRING_ARRAY grid as parameter.
#

#HackerRank
def gridChallenge(grid):
    # Write your code here
    #1) alphabet order
    grid = [sorted(x) for x in grid]

    result = True
    i = 0
    j = 0
    while result:
        if grid[i][j] < grid[i+1][j] and i < len(grid) - 2:
            i += 1
        elif i == len(grid) - 2 and j == len(grid[i]) - 1:
            break
        elif i == len(grid) - 2:
            j += 1
        else:
            print(i)
            result = False

    return "YES" if result else "NO"


    # for i in range(x):
    #     for j in range(grid[i]):
    #         grid[i][j]
    # for i in range(len(grid)):
    #      print(sorted(grid[i]))

def superDigit(n, k):
    def super_digit(n):
        if len(n) == 1:
            return n

        sum = 0
        for i in n:
            sum += int(i)
        return str(sum)

    p = n * k
    while len(p) != 1:
        p = super_digit(p)

    return p
    # # return int(n) * k
    # #return str(n)[0]

def minimumBribes(q):
    #...stack?
    pos = range(len(q)+1)
    org = range(len(q) + 1)
    p = [0] * len(q)
    for i in range(len(q)-1, -1, -1):
        oldp = pos[q[i]]
        newp = i + 1
        while oldp != newp:
            p[oldp-1] += 1
            if p[oldp-1] > 2:
                return "Too chaotic"
                break
            print(org[2])
            temp = org[oldp]
            print(org[oldp+1])
            org[oldp] = org[oldp+1]
            org[oldp+1]  = temp
            #org[oldp], org[oldp+1] = org[oldp + 1], org[oldp]

            print(org)
            #pos[q[oldp]] = oldp
            print(oldp)
            print(p)
            print(q)

    for x in p:
        if x > 2:
            return "Too chaotic"
    return sum(p)

def solution(board, moves): #https://programmers.co.kr/learn/courses/30/lessons/64061 2주차
    stack = []
    dp = collections.defaultdict(collections.deque)

    for b in board:
        for a in range(len(b)):
            if b[a] != 0:
                dp[a].append(b[a])
    print(dp)
    answer = 0
    for move in moves:
        if dp[move-1]:
          t =  dp[move-1].popleft()
          print("t", t)
          if stack and stack[-1] == t:
                answer   += 2
                stack.pop()
          else:
                stack.append(t)
    return answer

if __name__ == '__main__':

    grid = ['ebacd', 'fghij', 'olmkn', 'trpqs', 'xywuv']
    # result = gridChallenge(grid)
    # #print(result)
    #
    # #recursion , dp ...?
    # result = superDigit("9875", 4)
    # result = superDigit("148", 3)
    #print(result)

    #print(minimumBribes([2, 1, 5, 3, 4]))
    #print(minimumBribes([2, 5, 1, 3, 4]))

    #print(minimumBribes([1, 2, 5, 3, 4, 7, 8, 6]))
    #print(minimumBribes([5, 1, 2, 3, 7, 8, 6, 4]))
    #print(minimumBribes([1, 2, 5, 3, 7, 8, 6, 4]))
