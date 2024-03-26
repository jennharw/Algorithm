# import collections
# import re
# b = input()
# b = re.findall('\d+|\+|\-', b)
# # greedy 더하기 빼기
#
# result = []
# i = 0
# while i < len(b):
#     if b[i] == '+':
#         result[-1] = str(int(result[-1]) + int(b[i + 1]))
#         i += 2
#     else:
#
#         result.append(b[i])
#         i += 1
#
# x = eval(''.join(token.lstrip('0') for token in result))
# print(x)


# N, K = map(int, input().split())
#
# coinlist = list()
# for i in range(N):
#     coinlist.append(int(input()))
# coinlist.sort(reverse=True)
#
# result = 0
# for coin in coinlist:
#     result += K //coin
#     K = K % coin
#     if K == 0:
#         print(result)
#         break

# n = int(input())
# plist = list(map(int, input().split()))
# plist.sort()
#
# result = [plist[0]]
# for p in plist[1:]:
#     result.append(result[-1] + p)
# print(sum(result))

# N = int(input())
# count = 0
#
# movie = 666
#
# while True:
#     if '666' in str(movie):
#         count += 1
#     if count == N:
#         print(movie)
#         break
#     movie += 1


# import itertools
# N, T =  map(int, input().split())
# cards = list(map(int, input().split()))
#
# maximum = 0
# for x in list(itertools.combinations(cards, 3)):
#     if maximum < sum(x) <= T:
#         maximum = sum(x)
# print(maximum)


# n = int(input())
# p = n // 5
# while p >= 0:
#     if (n - (p * 5))  % 3 == 0:
#         print(p +  ((n - (p * 5))  // 3))
#         break
#     p -= 1
# if p == -1:
#     print(-1)

# result=[]
# import math
#
# def vertran(n):
#
#     prime = [True] * (2*n+1)
#     for i in range(2, int(math.sqrt(2*n))):
#         j = 1
#         while i * j < 2*n+1:
#             prime[i*j] = False
#             j += 1
#     count = 0
#     for i in range(n+1, 2*n+1):
#         if prime[i]:
#             count += 1
#     return count
#
# while True:
#     a = int(input())
#     if a == 0:
#         break
#     result.append(vertran(a))
#
# for i in result:
#     print(i)
# #매번 하는게 아니라



# A,B,V=  map(int, input().split())
# v_ = V - A
# count = v_ // (A - B)
# if v_ % (A-B) > 0 :
#     count +=1
# print(count+1)
# t = int(input())

# for i in range(t):
#     h, w, n = map(int, input().split())
#     num = n//h + 1
#     floor = n % h
#     if n % h == 0:  # h의 배수이면,
#         num = n//h
#         floor = h
#     print(f'{floor*100+num}')
# t = int(input())
#
# for i in range(t):
#     H, W, N = map(int, input().split())
#     Y = N % H  #
#     X = N // H + 1
#
#     if Y == 0:
#         Y = H
#
#     print(str(Y) + str(X).zfill(2))
# import math
# M, N=  map(int, input().split())
#
# prime = [True] * (N + 1)
#
# for i in range(2, int(math.sqrt(N)) + 1):
#     j = 2
#     while i * j <= N+1:
#         prime[i * j] = False
#         j += 1
#
# for i in range(M, N + 1):
#     if prime[i]:
#         print(i)

N = int(input())
current = str(N).zfill(2)  # 26
count = 1

while True:
    next = 0
    for i in current:
        next += int(i)  # 8

    next = current[-1] + str(next)[-1]

    # print(current, next)
    if int(next) == N:
        print(count)
        break

    current = next
    count += 1
