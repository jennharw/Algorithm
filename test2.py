
###실전문제

# 모험가길드
import collections
import functools
import heapq
import itertools
import math
import re
from datetime import datetime


def exploreguild(people):
    people.sort() #1 2 2 2 3

    result = 0  # 총 그룹의 수
    count = 0  # 현재 그룹에 포함된 모험가의 수

    for p in people:  # 공포도를 낮은 것부터 하나씩 확인하며o
        count += 1
        if count >= p:
            result += 1
            count = 0
    return result

#곱하기 혹은 더하기
def timeoradd(snum:str):
    result = int(snum[0])
    for s in snum[1:]:
        s = int(s)
        if result <= 1 or s <= 1:
            result += s
        else:
            result *= s

    return result


#문자열뒤집기
def flipnum(snum:str):
    #연속된
    a = 1
    b = 0
    for i in range(len(snum)-1):
        if snum[i] != snum[i+1]:
            if snum[i+1] == snum[0]:
                a += 1
            else:
                b += 1
    return min(a, b)

#만들수 없는 금액
def cannotmake(clist):
    clist.sort()
    target = 1 #11 인데 다음 값 13 이면 12

    for i in range(len(clist)):

        if target < clist[i]:
            return target
        target += clist[i]

#볼링공고르기
def bowling(blist): #
    count = collections.Counter(blist)
    count_keys = list(itertools.combinations(list(count.keys()), 2)) #* 2 # 두 사람
    print(count_keys)
    result = 0
    # m = count(countt)
    for x,y in count_keys:
        n = count[x] #
        m = count[y]

        result += n * m# * 2

    return result

#무지의 먹방라이브
def muzimukbang(food_times, k): #몇번음식부터 먹어야 하는지
    if sum(food_times) <= k:
        return -1

    #우선순위 q
    q = []
    for i in range(len(food_times)):
         heapq.heappush(q, (food_times[i], i+1)) #시간, 번호

    sum_value = 0
    previous = 0
    length = len(food_times)

    while sum_value + (q[0][0] - previous) * length <= k:
        now = heapq.heappop(q)[0]
        sum_value += sum_value + (now - previous) * length
        length -= 1
        previous = now

    #남은 음식 중 몇번 째
    result = sorted(q, key = lambda x:x[1])
    return result[(k - sum_value) % length][1]




#럭키스트레이트
def luckystraing(N):
    l = len(N) // 2

    summary = 0
    for x in N[:l]:
        print(x)
        summary += int(x)

    for y in N[l:]:
        print(y)
        summary -= int(y)

    if summary == 0:
        print("LUCKY")
    else:
        print("READY")



def recostructstr(strt : str):
    ss = []
    rr = 0
    for s in strt:
        if s.isdigit():
            rr += int(s)
        else:
            ss.append(s)

    return ''.join(sorted(ss)) + str(rr)



def compressstr(s): #aabbaccc
    answer = l = len(s)
    for step in range(1, len(s) // 2 + 1):
        compressed = ""
        prev = s[0:step]
        count = 1

        for j in range(step, len(s), step):

            if prev == s[j:j+step]:
                count += 1
            else:
                compressed += str(count) + prev if count >= 2 else prev
                prev = s[j:j+step]
                count = 1

        compressed += str(count)  + prev if count >= 2 else prev
        answer = min(answer, len(compressed))

    return answer


def key(key, lock):
    def rotate_key(key):
        m = len(key)

        r_key = [[0] * (m) for _ in range(m)]
        for i in range(m):
            for j in range(m):
                r_key[j][m - i - 1] = key[i][j]
        return r_key

    def check(new_lock):
        for i in range(m-1, m+n-1):
            for j in range(m-1, m+n-1):
                if new_lock[i][j] != 1:
                    return False
        return True

    m = len(key)
    #열쇠
    n = len(lock)
    #lock

    grid = [[0] * (n + 2*(m-1)) for _ in range(n + 2*(m-1))]

    for i in range(len(lock)):
        for j in range(len(lock[0])):
            grid[i+m-1][j+m-1] = lock[i][j]

    #회전 하기
    #빈칸 없는지 확인

    #겹치기
    new_key = rotate_key(key)
    print(new_key)

    #좌물쇠에 끼우기
    for x in range(m + n - 1):
        for y in range(m + n - 1):
            for i in range(m):
                for j in range(m):
                    grid[x + i][y + j] += new_key[i][j]
            print("xy", x, y)
            print(grid)
            if check(grid) == True:
                return True

            #좌물쇠에서 열쇠 빼기
            for i in range(m):
                for j in range(m):
                    grid[x + i][y + j] -= new_key[i][j]
    return False

def snake_game(N, K, apple_list, direction_list):
    grid = [[0] * N for _ in range(N)]

    dx = [0,0, 1,-1 ]
    dy = [1,-1,  0,0]
    #방향 오 D 왼 L
    # 오 왼 아래 위 방향 0 1 2 3

    dic_head = collections.defaultdict(dict)

    dic_head[0]['D'] = 2 #아래
    dic_head[0]['L'] = 3
    dic_head[1]['D'] = 3
    dic_head[1]['L'] = 2
    dic_head[2]['D'] = 1
    dic_head[2]['L'] = 0
    dic_head[3]['D'] = 0
    dic_head[3]['L'] = 1

    for ax, ay in apple_list:
        grid[ax-1][ay-1] = 1


    x = y = d = 0
    traced = [(x,y)]

    result = 0
    while True:
        traced.pop(0)
        #1칸 이동
        nx = x + dx[d]
        ny = y + dy[d]

        if (nx, ny) in traced or nx < 0 or nx >= N or ny < 0 or ny >= N:
            return result + 1
        else:
            #사과가 있을 시
            if grid[nx][ny] == 1:
                grid[nx][ny] = 0
                traced.append((x, y))

            traced.append((x,y))

            x = nx
            y = ny
            traced.append((x,y))

            result += 1


            if direction_list and  result == direction_list[0][0]:

                d = dic_head[d][direction_list[0][1]] #오 D -> 아래 방향
                print(direction_list[0][0], d)
                direction_list.pop(0)
                #방향 바꿔야 할 시

# #기둥과보설치
def possible(answer):
    for x, y, stuff in answer:

        if stuff == 0:  # 설치된 것이 '기둥'인 경우
            # '바닥 위' 혹은 '보의 한쪽 끝 부분 위' 혹은 '다른 기둥 위'라면 정상
            if y == 0 or [x - 1, y, 1] in answer or [x + 1, y, 1] in answer or [x, y - 1, 0] in answer:
                continue
            return False  # 아니라면 거짓(False) 반환
        elif stuff == 1:  # 설치된 것이 '보'인 경우
            # '한쪽 끝부분이 기둥 위' 혹은 '양쪽 끝부분이 다른 보와 동시에 연결'이라면 정상
            if [x, y - 1, 0] in answer or [x + 1, y - 1, 0] in answer or (
                    [x - 1, y, 1] in answer and [x + 1, y, 1] in answer):
                continue
            return False  # 아니라면 거짓(False) 반환
    return True



def columns_and_beams(build_frame):
    traced = []

    for build in build_frame:
        x, y, stuff, operate = build
        if operate == 0:  # 삭제하는 경우
            traced.remove([x, y, stuff])  # 일단 삭제를 해본 뒤에
            if not possible(traced):  # 가능한 구조물인지 확인
                traced.append([x, y, stuff])
        if operate == 1:  # 설치하는 경우
            traced.append([x, y, stuff])  # 일단 설치를 해본 뒤에
            if not possible(traced):  # 가능한 구조물인지 확인
                traced.remove([x, y, stuff])
    return sorted(traced)
                #     print(build)
    #     print(traced)
    #
    #     if build[2] == 1: #보 - 한쪽 끝 부분이 기둥 위에 있거나 양쪽 끝 부분이 다른 보와 동시에 연결되어
    #         if ([build[0]+1, build[1]-1, 0] in traced) or ([build[0], build[1]-1, 0] in traced) or ([build[0]-1, build[1], 1] in traced and [build[0]+1, build[1], 1] in traced) :
    #             if build[3] == 1:
    #                 traced.append([build[0], build[1] ,1])
    #             # else:
    #             #     traced.remove([build[0], build[1] ,1])
    #
    #     else: # 기둥 - 바닥 위에 있거나  다른 기둥 위에 있어야 보의 한쪽 끝 부분 위에 있거나
    #         if build[1] == 0  or ([build[0]-1, build[1], 1] in traced) or ([build[0], build[1]-1, 0] in traced) or  ([build[0]+1, build[1], 1] in traced) :
    #             if build[3] == 1:
    #                 traced.append([build[0] , build[1] ,0])
    #             # else:
    #             #
    #             #     traced.remove([build[0] , build[1] ,0])
    #
    # return sorted(traced, key= lambda x:(x[0],x[1],x[2]))


def chicken(N, M, grid):

    def distance(chick):
        result = 0
        for hx, hy in houses:
            res = []
            for cx, cy in chick:
                res.append(abs(cx-hx) + abs(cy-hy))
            result += min(res)
        return result


    chickens = []
    houses = []
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 2:
                chickens.append([i, j])
                grid[i][j] = 0
            elif grid[i][j] == 1:
                houses.append([i, j])
    minimum = int(1e9)
    for chick in list(itertools.combinations(chickens, M)):
        dist = distance(chick)
        if dist < minimum:
            minimum = dist
    return minimum

def inspection(n, weak, dist):
    # dist.sort()
    #
    # diff = []
    # for i, w in enumerate(weak):
    #     if i == 0:
    #         diff.append(weak[i] - weak[i-1] + n)#  1 - 10 + 12   3
    #         diff.append(weak[i+1] - weak[i])
    #     elif i == len(weak) -1 :
    #         diff.append(weak[i] - weak[i - 1])  # 1 - 10 + 12   3
    #         diff.append(weak[0] - weak[i] + n)
    #     else:
    #         diff.append(weak[i] - weak[i - 1])
    #         diff.append(weak[0] - weak[i])
    #
    # BruteForce 완전 탐색 원형 배열

    #list(itertools.permutations(dist))
    weakSize = len(weak)
    weak = weak + [w + n for w in weak]
    minCnt = math.inf
    for start in range(weakSize):
        for d in itertools.permutations(dist):
            cnt = 1
            pos = start
            for i in range(1, weakSize):
                nextPos = start + i
                diff = weak[nextPos] - weak[pos]
                if diff > d[cnt - 0]:
                    pos = nextPos
                    cnt += 1
                    if cnt > len(dist):
                        break
            if cnt <= len(dist):
                minCnt = min(minCnt, cnt)

    if minCnt == math.inf:
        return  -1

    result = 0
    return minCnt




# 모험가 길드
# 곱하기 혹은 더하기
# 문자열 뒤집기
# 만들 수 없는 금액
# 볼링공 고르기
# 무지의 먹방라이브
#
# #럭키스트레이트
# #문자열재정렬
# #문자열압축
# #자물쇠와열쇠
# #뱀
# #기둥과보설치
# #치킨배달
# #외벽점검

#비밀 지도 ★
def secret_map(n, arr1, arr2):
    maps = []
    for i in range(n):
        maps.append(bin(arr1[i] | arr2[i])[2:]
                    .zfill(n)
                    .replace('1','#')
                    .replace('0',' '))

    return maps

    arr_1 = []


    for i in range(n):
            b = bin(arr1[i])
            c = bin(arr2[i])
            b0 = '0' * (n - len(b)+2) + b[2:]
            c0 = '0' * (n - len(c)+2) + c[2:]

            k = ""
            for j in range(n):
                if b0[j] == '0' and  c0[j] == '0':
                    k += ' '
                else:
                    k += "#"
            arr_1.append(k)
    return arr_1

#다트 게임	★
def dart_game(dartResult):
    result = []
    r = 0
    i = 0

    while i < len(dartResult):
        r = int(dartResult[i])
        if dartResult[i+1] == 'S':
            r = r ** 1
        elif dartResult[i+1] == 'D':
            r = r ** 2
        else:
            r = r ** 3


        if i + 2 < len(dartResult) and dartResult[i+2] == '*':
            if result:
                result[-1] = result[-1]*2
            result.append(r*2)
            i += 3
        elif i + 2 < len(dartResult) and dartResult[i+2] == '#':

            result.append(r*-1)
            i += 3
        else:
            result.append(r)
            i += 2
    return sum(result)

def cache_cities(cacheSize, cities):
    cache = collections.deque(maxlen=cacheSize)


    result = 0
    for city in cities:
        city = city.lower()
        if city.lower() in cache:
            cache.append(city)
            cache.remove(city)
            result += 1
        else:
            result += 5
            cache.append(city)


    return result


def shuttle(n, t, m, timetable):
    #셔틀은 09:00부터 총 n회 t분 간격으로 역에 도착하며, 하나의 셔틀에는 최대 m명의 승객
    #콘이 셔틀을 타고 사무실로 갈 수 있는 도착 시각 중 제일 늦은 시각

    timetable.sort()
    timetable = [
        int(time[:2]) * 60 + int(time[3:])
        for time in timetable
    ]
    print(timetable)
    current = 60 * 9 #'09:00'

    for _ in range(n): #셔틀 수
        for _ in range(m): #셔틀에 타는 사람 수
            if timetable and timetable[0] <= current: ## 대기가 있는 경우 1분 전 도착
                candidate = timetable.pop(0) - 1
            else:  # 대기가 없는 경우 정시 도착
                candidate = current

        current += t # t 분 후

    # 시, 분으로 다시 변경
    h, m = divmod(candidate, 60)
    return str(h).zfill(2) + ':' + str(m).zfill(2)

#뉴스 클러스터링
def news_clustering(str1, str2):

    # 두 글자씩 끊어서 다중집합 구성
    str1s = [
        str1[i:i + 2].lower()
        for i in range(len(str1) - 1)
        if re.findall('[a-z]{2}', str1[i:i + 2].lower())
    ]
    str2s = [
        str2[i:i + 2].lower()
        for i in range(len(str2) - 1)
        if re.findall('[a-z]{2}', str2[i:i + 2].lower())
    ]

    #교집합
    intersection = sum((collections.Counter(str1s) &
                        collections.Counter(str2s)).values())
    # 합집합 계산
    union = sum((collections.Counter(str1s) |
                 collections.Counter(str2s)).values())

    jaccard_sim = 1 if union == 0 else intersection / union
    return int(jaccard_sim * 65536)



def friends4block(m, n, board): #반복 보다 그냥 sliding 아니라
    board = [list(x) for x in board]

    #일치 판별
    matched = True
    while matched:
        matched = []
        for i in range(m - 1):
            for j in range(n - 1):
                if board[i][j] == board[i][j+1] == board[i+1][j+1] == board[i+1][j] !='#':
                    matched.append([i, j])
        #삭제
        for i, j in matched:
            board[i][j] = board[i][j + 1] = board[i + 1][j + 1] = board[i + 1][j] = '#'

        # 빈공간
        for _ in range(m):
            for i in range(m - 1):
                for j in range(n - 1):
                    if board[i + 1][j] == '#':
                        board[i+1][j], board[i][j] = board[i][j] , '#'

    return sum(x.count('#') for x in board)

#추석 트래픽
import datetime

def traffic(times):#sliding?
    combined_logs = []
    for log in times:
        logs = log.split(' ')
        timestamp = datetime.datetime.strptime(logs[0] + ' ' + logs[1], "%Y-%m-%d %H:%M:%S.%f").timestamp()

        combined_logs.append((timestamp, -1))
        combined_logs.append((timestamp - float(logs[2][:-1]) + 0.001, 1))

    accumulated = 0
    max_requests = 1
    combined_logs.sort(key=lambda x: x[0])
    for i, elem1 in enumerate(combined_logs):
        current = accumulated
        for elem2 in combined_logs[i:]:
            if elem2[0] - elem1[0] > 0.999:
                break
            if elem2[1] > 0:
                current += elem2[1]

        max_requests = max(max_requests, current)
        accumulated += elem1[1]

    return max_requests
#문제은행 greedy bruteforce
# 잃어버린 괄호 1541

def lostb(b):
    # b=input()
    b = re.findall('\d+|\+|\-',b)
    #greedy 더하기 빼기

    result = []
    i = 0
    while i < len(b):
        if b[i] == '+':
            result[-1] =str(int(result[-1]) + int(b[i+1]))
            i += 2
        else:

            result.append(b[i])
            i += 1

    x = eval(''.join(token.lstrip('0') for token in result))
    print(x)

# 동전 0 11047


def coin(N, K, coinlist):#약수
    coinlist.sort(reverse=True)

    result = 0
    for coin in coinlist:
        result += K //coin
        K = K % coin
        if K == 0:
            return result


# ATM 11399
def ATM(plist):
    n = int(input())
    plist = list(map(int, input().split()))
    plist.sort()

    result = [plist[0]]
    for p in plist[1:]:
        result.append(result[-1] + p)
    print(sum(result))

# 영화감독 숌 1436


def movie666(N):
    count = 0

    movie = 666

    while True:
        if '666' in str(movie):
            count += 1
        if count == N:
            print(movie)
            break
        movie += 1


# 블랙잭 2798
def blackjak(T, cards):
    maximum = 0
    for x in list(itertools.combinations(cards, 3)):
        if maximum < sum(x) <= T:
            maximum = sum(x)
    print(maximum)


# 분해합 2231
def divideSum(N):
    result = 1

    while True:
        tgt = result
        for i in str(result):
            tgt += int(i)
        if tgt == N:
            print(result)
            break

        result += 1



#프로그래머스

def mock_exam(answers):

    x = [1,2,3,4,5] #5

    y = [2,1,2,3,2,4,2,5] #8

    z = [3,3,1,1,2,2,4,4,5,5] #10

    x_ = 0
    y_ = 0
    z_ = 0
    for i, wer  in enumerate(answers):

        if x[i%5] == wer:
            x_ += 1

        if y[i % 8] == wer:
            y_ += 1

        if z[i % 10] == wer:
            z_ += 1
    list1 = [x_, y_, z_]
    mx = max(list1)
    result = []
    for i in range(len(list1)):
        if list1[i] == mx:
            result.append(i+1)

    return result


def find_prime(numbers):
    numbers = re.findall('\d',numbers)
    numbers.sort(reverse=True)
    mx = int(''.join(numbers))
    is_p = [True] * (mx+1)
    for i in range(2, int(math.sqrt(mx))+1):
        j = 2
        while i * j < mx+1:
            is_p[i * j] = False
            j += 1

    # #여기서
    exe = set()
    l = len(numbers)
    for i in range(1, l + 1):
        for x in itertools.permutations(numbers, i):
            exe.add(x)

    result = set()
    for k in exe:
        k = int(''.join(k))
        if is_p[k]:
            result.add(k)

    if 1 in result:
        result.remove(1)
    if 0 in result:
        result.remove(0)
    count = 0
    for _ in result:
        count += 1
    return count
    # print(exe)
    #
    # for i, p in enumerate(is_p[2:]):
    #     if p and all([i in numbers for i in str(i+2)]):
    #         result.append(i+2)
    # return result


def carpet(brown, yellow):

    #
    for i in range(1, yellow+1):
        if yellow % i == 0:
            #세로가 1일때  yellow / 1   2
            if (i  + (yellow/i)) * 2 + 4 == brown:
                return [int(yellow/i) + 2 , i +2]



def training(n, lost, reserve): #
    reserve_uniq = set(reserve) - set(lost)
    lost_uniq = set(lost) - set(reserve)

    for i in reserve_uniq:
        if i - 1 in lost_uniq:
            lost_uniq.remove(i - 1)
        elif i + 1 in lost_uniq:
            lost_uniq.remove(i + 1)

    return n - len(lost_uniq)


def joystick(name):

    min_move = len(name) - 1

    result = 0

    for i, n in enumerate(name):
        result += min(ord(n) - ord("A"), ord("Z") - ord(n) + 1)

        # 해당 알파벳 다음부터 연속된 A 문자열 찾기
        next = i + 1
        while next < len(name) and name[next] == 'A':
            next += 1

        # 기존, 연속된 A의 왼쪽시작 방식, 연속된 A의 오른쪽시작 방식 비교 및 갱신
        min_move = min(min_move, 2 *  i + len(name) - next, i + 2 * (len(name) - next ))
    print(min_move)
    return result + min_move



def makebigest(number, k): #itertools 효율성X
    stck = []

    for num in number:
        while stck and int(stck[-1]) < int(num) and k > 0:
            stck.pop()
            k -= 1
        stck.append(num)
    ''.join(stck[:len(stck) - k])
    return ''.join(stck)



def life_boat(people, limit): #최대 2명 remove -> 효율성 x
    people.sort(reverse=True)
    i = 0
    j = len(people) - 1
    count = 0
    while i <= j:
        count += 1
        if people[i] + people[j] <= limit:
            j -= 1
        i += 1
    return count


#섬 연결하기
def find_parent(parent, a):
    if parent[a] != a:
        parent[a] = find_parent(parent, parent[a])
    return parent[a]

def union_parent(parent, a, b):
    a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a <= b:
        parent[b] = a
    else:
        parent[a] = b

def link_island(n, costs):

    parents = [0] * (n + 1)

    for i in range(n+1):
        parents[i] = i

    costs.sort(key = lambda x:x[2])
    dist = 0

    for x, y, d in costs:
        if find_parent(parents, x) != find_parent(parents, y):
            dist += d
            union_parent(parents, x, y)

    return dist




#단속카메라
def surveillance(routes):
    routes.sort(key=lambda x: x[1])
    result = [routes[0][1]]
    for route in routes:
        if route[0] > result[-1]:
            result.append(route[1])
    count = 0
    for _ in result:
        count+=1

    return count



def kth(array, commands):
    result = []
    for x, y, z in commands:
        result.append(sorted(array[x-1:y])[z-1])
    return result

def biggest(numbers):
    numbers = [str(num) for num in numbers]
    def compare(x, y):
        if x + y > y + x:
            return -1
        return 1

    numbers =  sorted(numbers, key = functools.cmp_to_key(compare))
    return ''.join(numbers)


def h_index(citations):
    citations.sort(reverse=True)
    mx = 0
    for i in range(len(citations)):
        if citations[i] <= i:
            return i
    return len(citations)

def longest(n, vertex):
    graph = collections.defaultdict(list)
    for x, y in vertex:
        graph[x].append(y)
        graph[y].append(x)

    visited = [-1] * (n + 1)
    visited[1] = 0

    #1번
    q = collections.deque()
    q.append((1, 0))
    while q:
        stt, c = q.popleft()
        for neighbor in graph[stt]:
            if visited[neighbor] == -1:
                visited[neighbor] = c + 1
                q.append((neighbor, c+1))

    return visited[1:].count(max(visited))

def ranking(n, results):
    #floyd
    INF = int(1e9)
    graph = [[INF] * (n+1) for _ in range(n+1)]

    for x in range(1, n+1):
        graph[x][x] = 0

    for x, y in results:
        graph[x][y] = 1

    for k in range(1, n+1):
        for p in range(1, n+1):
            for q in range(1, n+1):
                graph[p][q] = min(graph[p][q], graph[p][k] + graph[k][q])

    result = 0
    for i in range(1, n+1):
        count = 0
        for j in range(1, n+1):
            if graph[i][j] != INF or graph[j][i] != INF:
                count += 1
        if count == n:
            result += 1
    return result


    #topological sort
    indegree = [0] * (n + 1)
    graph = [[False] * (n + 1) for _ in range(n + 1)]

    for x, y in results:
            graph[x][y] = True  # a 순위가 b순위보다 높다면, a -> b로 연결, b입장에서 진입차수 +1
            indegree[y] += 1

    result = []  # 위상정렬 결과 담을 리스트
    cycle = False  # 큐에서 n번 노드가 나오기 전에 큐의 원소 길이가 0이 되었다 = 사이클 발생 = 정렬할 수 없음(순위 알 수 없음)
    certain = True  # 매번 큐 길이를 계산할 때, 큐 길이가 2이상(원소가 2개 이상)일 때, 즉 한 번에 큐에 2개 이상 원소가 들어감 = 위상 정렬 결과가 여러개!

    queue = collections.deque()
    for i in range(1, n+1):
        if indegree[i] == 0:
            queue.append(i)

    for i in range(n):
        if len(queue) == 0:
            cycle = True
            break
        if len(queue) >= 2:
            certain = False
            break

        # 위상 정렬 수행
        node = queue.popleft()
        print(node)
        pri
        result.append(node)  # 큐에서 원소를 pop하는 순서대로 결과 = 위상정렬 결과
        # 큐에서 빼낸 노드와 연결된 노드 탐색
        for j in range(1, n + 1):
            if graph[node][j]:
                indegree[j] -= 1
                # 새롭게 진입차수가 0이 된 노드 큐에 넣기
                if indegree[j] == 0:
                    queue.append(j)

    print(result)

#방의 개수

def room_number(arrows):

    answer = 0
    visited = collections.defaultdict(list)
    x, y = 0, 0
    dx, dy = [0,1,1,1,0,-1,-1,-1],[1,1,0,-1,-1,-1,0,1]
    for arrow in arrows:
        #for _ in range(2):
            nx = x + dx[arrow]
            ny = y + dy[arrow]

            if (nx, ny) in visited and (x, y) not in visited[(nx, ny)]:  # 방문했던 점이지만 경로가 겹치지 않는 경우, 방+1
                answer += 1
                visited[(x, y)].append((nx, ny))
                visited[(nx, ny)].append((x, y))

            elif (nx, ny) not in visited:  # 방문하지 않았던 경우
                visited[(x, y)].append((nx, ny))
                visited[(nx, ny)].append((x, y))
            # 경로가 겹치는 경우는 따로 해줄 필요 없음
            x, y = nx, ny
    return answer


#정수론 및 조합론
def divisor(N, real):
    return max(real) * min(real)

def gcd(n, m):
    if n == 0:
        return m
    if m == 0:
        return n
    if n > m:
        return gcd(m, n % m)
    if n <= m:
        return gcd(n, m % n)

def lcm(x, y):
    return (x * y) // gcd(x, y)


def binonmial_coefficient(N,K):#이항 계수

    return math.factorial(N) // (math.factorial(N-K) * math.factorial(K))


#다리 놓기
# 6C4 를 뽑고, 차례대로 배치 6!/(6-4)!4!

def bridge(r, n):

    return math.factorial(n) // (math.factorial(n-r) * math.factorial(r))




def sugar(n):

    p = n // 5
    while p >= 0:
        if (n - (p * 5))  % 3 == 0:
            return p +  (n - (p * 5))  // 3
        p -= 1
    return -1

    count += n % 5 // 3

    if n % 5 % 3 > 0:
        return -1
    return count

def vertran(n):

    prime = [True] * (2*n+1)
    for i in range(2, int(math.sqrt(2*n))+1):
        j = 1
        while i * j < 2*n+1:
            prime[i*j] = False
            j += 1
    count = 0
    for i in range(n+1, 2*n+1):
        if prime[i]:
            count += 1
    return count

def snail(A, B, V):
    v_ = V - A
    count = v_ // (A - B)
    if v_ % (A-B) > 0 :
        count +=1
    return count+1


def ACMhotel(H, W, N):
    #층 방 몇번째손님
    Y = N % H #
    X = N // H + 1

    if Y == 0:
        Y = H
        X = N // H

    return str(Y) + str(X).zfill(2)


def get_prime(M, N):
    prime = [True] * (N+1)

    for i in range(2, int(math.sqrt(N))+1):
        j = 2
        while i * j <= N:
            prime[i*j]=False
            j += 1

    for i in range(M, N+1):
        if prime[i]:
            print(i)


def add_cycle(N):
    current = str(N).zfill(2) # 26
    count = 1

    while True:
        next = 0
        for i in current:
            next += int(i) #8

        next = current[-1] + str(next)[-1]

        #print(current, next)
        if int(next) == N:
            return count

        current = next
        count += 1


def Fly_me_to_the_Alpha_Centauri(x, y): #현재위치, 목표위치
    #y - x
    #n(n+1) n**2 보다 작을 때 2*n -1
    d = y - x
    n = 0
    while True:
        if d <= n*(n+1):
            break
        n += 1

    if d <= n**2:
        print(n*2 - 1)
    else:
        print(n*2)

def turet(x1, y1, r1, x2, y2, r2):
    distance = math.sqrt((x1-x2)**2 + (y1-y2)**2) # 두 원의 거리
    if distance == 0 and r1 == r2: #
        print(-1)
    elif abs(r1-r2) == distance or r1+r2 == distance: #외접, 내접
        print(1)
    elif abs(r1-r2) < distance < (r1+r2): #서로다른두점에서 만날때
        print(2)
    else:
        print(0)



#스택큐해쉬우선순위큐
def stack():
    zero = []

    zero.append()
    zero.pop()
    sum(zero)

#def queue():

# def stack_lin():
#     stck = []
#     for i in list1:
#             if stck and stck[-1] > i:
#                 print("+")
#                 stck.append(i)
#             while stck[-1] > i :
#                 print("-")
#                 stck.pop()
#

def rotated_queue(n,mlist):
    queue = collections.deque()
    for i in range(n):
        queue.append(i+1)
    count = 0

    for m in mlist:
        mid = len(queue) // 2
        if queue.index(m) > mid:
            while m != queue[0]:
                queue.appendleft(queue.pop())
                count += 1
            queue.popleft()
        else:
            while m != queue[0]:
                queue.append(queue.popleft())
                count += 1
            queue.popleft()

    return count



def bracket(brack):
    stack = []
    for s in brack:
        if s == '[' or s == '(':
            stack.append(s)
        elif s == ']':
            if stack and stack[-1] == '[' :
                stack.pop()
            else:
                return "no"
        elif s == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                return "no"
    if stack:
        return "no"

    return "yes"


def bracket2(sentence):
    stack = []
    for s in sentence:
        if s == '[' or s == '(':
            stack.append(s)
        elif s == ']':
            if stack and stack[-1] == '[' :
                stack.pop()
            else:
                return "no"
        elif s == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                return "no"
    if stack:
        return "no"

    return "yes"

def yosepus(n, k):
    queue = collections.deque()
    answer = []
    for i in range(1, n+1):
        queue.append(i)

    while queue:
        for i in range(k-1):
            queue.append(queue.popleft())
        answer.append(queue.popleft())
    return answer





#프로그래머스
#힙

import heapq
def solution(scoville, K):
    answer = 0
    # heapq

    heapq.heapify(scoville)
    # print(heapq.heappop(scoville))
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

    return answer

def disk_controller(jobs):
    heap = []

    answer, now, i = 0, 0, 0
    start = -1

    while i < len(jobs):
        for j, job in jobs:
            if start < j <= now: # 0 3 9
                heapq.heappush(heap, (job, j))
        if heap: #
            job, t = heapq.heappop(heap)
            start = now #start 2
            now += job #3 + 6
            answer += now - t #now - t 7
            i += 1
        else:
            now += 1
    return int(answer / len(jobs))


def twoPriorityQueue(operations):
    heap = []
    heap_ = []
    for o in operations:
        o =  o.split(' ')
        if o[0] == 'I':
            heapq.heappush(heap, int(o[1]))
            heapq.heappush(heap_, -int(o[1]))

        elif heap and o[0] == 'D' and o[1] == '1':
            x = heapq.heappop(heap_) #최댓값
            heap.remove(-x)
        elif heap and o[0] == 'D' and o[1] == '-1':
            x = heapq.heappop(heap)  # 최솟값
            heap_.remove(-x)
    if len(heap) == 0:
        return [0, 0]
    else:
        x = heapq.heappop(heap_)  # 최댓값
        y = heapq.heappop(heap)  # 최솟값
        return [-x, y]


def func_develop(progresses, speeds):
    # Q=[]
    # for p, s in zip(progresses, speeds):
    #     if len(Q) == 0 or Q[-1][0] < -((p - 100) // s):
    #         Q.append([-((p - 100) // s), 1])
    #     else:
    #         Q[-1][1] += 1
    # return [q[1] for q in Q]

    queue = collections.deque()

    for i in range(len(progresses)):
        queue.append(math.ceil((100 - progresses[i]) / speeds[i]))

    results = []
    x = queue.popleft()
    count = 1
    while queue:
            y = queue.popleft()
            if x >= y:
                count += 1
            else:
                results.append(count)
                x = y
                count = 1
    results.append(count)
    return results
    # [7 4 10] 2 1
    # [5 10 1 1 20 1] 1 3 2


def printer(priorities, location):

    queue = collections.deque()
    for i, p in enumerate(priorities):
        queue.append((p, i))
    priorities.sort(reverse=True)
    result = []
    for priority in priorities:
        y = queue.popleft()
        while y[0] < priority:
            queue.append(y)
            y = queue.popleft()
        result.append(y[1])
    return result.index(location) + 1


def stock(prices):#가격이 떨어지지 않는 구간
    #n2
    #O(n) 으로 해야

    #stack 으로 비교

    stack = []
    l = len(prices)
    result = [0] * l
    for i, price in enumerate(prices):
        while stack and stack[-1][0] > price:
            p, j = stack.pop()
            result[j] = i - j
        stack.append((price, i))

    for s in stack:
        result[s[1]] = l - s[1] - 1

    return result


def truck(bridge_length	, weight, truck_weights):
    bridge = collections.deque(0 for _ in range(bridge_length))
    total_weight = 0
    step = 0
    truck_weights.reverse()
    while truck_weights:
        total_weight -= bridge.popleft()
        if total_weight + truck_weights[-1] > weight:
            bridge.append(0)
        else:
            truck = truck_weights.pop()
            bridge.append(truck)
            total_weight += truck
        step += 1

    step += bridge_length

    return step

def player(participant, completion):
    c = collections.Counter(participant)

    for play in completion:
        c[play] -= 1

    c -= collections.Counter()
    result = []
    for p in c:
        return p

def telephone(phoneBook):

    phoneBook = sorted(phoneBook)

    for p1, p2 in zip(phoneBook, phoneBook[1:]):
        if p2.startswith(p1):
            return False
    return True
    #  1. Hash map을 만든다
    hash_map = {}
    for phone_number in phoneBook:
        hash_map[phone_number] = 1

    # 2. 접두어가 Hash map에 존재하는지 찾는다
    for phone_number in phone_book:
        prefix = ""
        for number in phone_number:
            prefix += number
            # 3. 접두어를 찾아야 한다 (기존 번호와 같은 경우 제외)
            if prefix in hash_map and prefix != phone_number:
                return False
    return True




def disguise(clothes):
    c = collections.defaultdict(int)
    for p, q in clothes:
        c[q] += 1

    closet = []
    for k in c:
        closet.append(c[k])

    result = 1
    for i in closet:
        result *= (i+1) #안 입을 경우
    return result - 1

def best_album(genres, plays):

    c = collections.Counter()
    p = collections.defaultdict(list)

    for i in range(len(plays)):
        p[genres[i]].append((plays[i], i))
        c[genres[i]] += plays[i]

    genre = c.most_common(len(genres)) #순서대로

    result = []
    for g in genre:
        for k in sorted(p[g[0]], reverse=True, key = lambda x:(x[0], -x[1]))[:2]:
            result.append(k[1])
    return result





##비트
def single_number(nums):
    result = 0
    for num in nums:
        result ^= num
    return result
def hamming_distance(x, y): #몇 비트 다른지
    return bin(x^y).count('1')

def sumoftwointergers(a, b): #가산기
    #sum
    #carry
    #a^b ^ a&b<<1
    MASK = 0xFFFFFFFF
    INT_MAX = 0x7FFFFFFF
    while b != 0:
        a, b = (a^b) & MASK, ((a&b)<<1) & MASK
        # tmp = (a&b) << 1
        # a = a^b
        # b = tmp

    #음수처리
    if a > INT_MAX:
        a = ~(a ^ MASK)

    return a

def utf_8_validation(data):

    def countOnes(num):
        count = 0
        for i in range(7, -1, -1):
            if num & (1<<i):
                count += 1
            else:
                break
        return count # 110 1110 11110 0
    count = 0
    for d in data:

        if not count:
            count = countOnes(d)
            if count == 0:
                continue #하나
            if count == 1 or count>4: #10
                return False
            count -= 1
        else:
            count -= 1
            if countOnes(d) != 1:
                return False
    return count == 0

def number_of_1bits(unsigned_integer):
    #bin(unsigned_integer^ob0000000000).count('1')
    return bin(unsigned_integer).count('1')

    count = 0
    while n:
        n  &= n -1
        count += 1
    return count

#더 알아ㄷ면 좋은 알고리즘
#소수의 판별


def is_prime(x):
    #2부터 x-1까지 확인 제곱근까지만 확인
    for i in range(2, int(math.sqrt(x))+1):
        if x % i == 0:
            return False
    return True

#에라토스테네스의 체 여러개의 수가 소수인지 아닌지  # 제곱근까지
def Sieve_of_Eratosthenes():
    n = 1000
    array = [True for i in range(n+1)]

    for i in range(2, int(math.sqrt(n)) + 1):
        if array[i] == True:
            j = 2
            while i * j <= n:
                array[i*j] = False
                j += 1

    for i in range(2, n+1):
        if array[i]:
            print(i, end = ' ')

def two_pointer_sum(data, m):
    end = 0
    n = len(data)
    interval_sum = 0
    count = 0
    for start in range(n):

        while interval_sum < m  and end < n:
            interval_sum += data[end]
            end +=1
        if interval_sum == m:
            count += 1
        interval_sum -= data[start]
    return count

def two_pointer_two_list(l1, l2):
    i = j =0
    answer = []
    while i < len(l1) and j < len(l2):
        if l1[i] < l2[j]:
            answer.append(l1[i])
            i += 1
        else:
            answer.append(l2[j])
            j += 1
    if i != len(l1):
        answer.extend(l1[i:])
    if j != len(l2):
        answer.extend(l2[j:])
    return answer

def subsum(data, left, right):
    sum_value = 0
    prefix_sum = [0]
    for i in data:
        sum_value += i
        prefix_sum.append(sum_value)

    return prefix_sum[right] - prefix_sum[left-1]

def get_primenumber(m, n):
    array = [True for _ in range(100001)]
    array[1] = False

    for i in range(2, int(math.sqrt(n))+1):
        if array[i] == True:
            j=2
            while i * j <= n:
                array[i * j] = False
                j += 1
    for i in range(m, n+1):
        if array[i]:
            print(i)

def makepassword(alphabet, l):
    vowels = ('a', 'e', 'i', 'o', 'u')  # 5개의 모음 정의
    alphabet.sort()
    for password in itertools.combinations(alphabet, l):
        count = 0
        for i in password:
            if i in vowels:
                count += 1
        #최소 모음 1개와 최소 2개의 자음
        if count >= 1 and count <= l - 2:
            print(''.join(password)) #순서

#추가보충자료

def heapsort(n):
    q = []
    for i in n:
        heapq.heappush(q, i)
    while q:
        print(heapq.heappop(q))

def binaryIndexTree(n, M, K, arr, blist): #구간합 구하기

    tree = [0] * (n+1) # 합
    arr.insert(0, 0)
    def prefix_sum(i):
        result = 0
        while i > 0:
            result += tree[i]
            i -= (i & -i)
        return result

    def update(i, dif): #차이만큼
        while i <= n:
            tree[i] += dif
            i += (i & -i)

    def interval_sum(start, end):
        return prefix_sum(end) - prefix_sum(start - 1)



    for i in range(1, n+1):
        update(i, arr[i])

    for a, b, c in blist:
        if  a == 1:
            update(b, c-arr[b]) #바뀐 크기만큼
            arr[b] = c
        else:

            print(interval_sum(b, c))

def bellman_ford(n, m, edges):
    #음수 간선이 포함된 그래프에서 최단경로 찾기
    #djiakstra, 음수 간선이 cyclic 한지 확인 -> -1

    INF = int(1e9) # 무한을 의미하는 값으로 10억을 설정
    # 최단 거리 테이블을 모두 무한으로 초기화
    distance = [INF] * (n + 1)

    def bf(start):
        distance[start] = 0

        for i in range(n): #n-1 round
            for j in range(m):
                cur_node = edges[j][0]
                next_node = edges[j][1]
                edge_cost = edges[j][2]
                # 현재 간선을 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
                if distance[cur_node] != INF and distance[next_node] > distance[cur_node] + edge_cost:
                    distance[next_node] = distance[cur_node] + edge_cost
                    if i == n-1: # n 번째 라운드에서도 값이 갱신딘다면 음수순환이 존재
                        return True
        return False

    negative_cycle = bf(1)
    if negative_cycle:
        return -1
    else:
        for i in range(2, n+1):
            if distance[i] == INF:
                print(-1)
            else:
                print(distance[i])

#깊이 확인

def leastCommonAncestor(n, pairs, get):
    parent = [0] * (n+1) #부모노드
    d = [0] * (n + 1) # 각 노드의 깊이
    c = [0] * (n + 1) # 노드의 깊이가 계산되었는지 여부
    graph = [[] for _ in range(n+1)]
    for x, y in pairs:
        graph[x].append(y)
        graph[y].append(x)

    def dfs(x, depth):
        c[x] = True
        d[x] = depth
        for y in graph[x]:
            if c[y]: #이미 구했다면
                continue
            parent[y] = x
            dfs(y, depth+1)

    def LCA(a, b): #최소공통조상
        while d[a] != d[b]: #조상찾기
            if d[a] > d[b]:
                a = parent[a]
            else:
                b = parent[b]
        while a != b: #조상 같아 지도록
            a = parent[a]
            b = parent[b]
        return a

    dfs(1, 0) #루트노드

    for a, b in get:
        print(LCA(a,b))


def leastCommonAncestor2(n, pairs, get):
    LOG = 21  # 2^20 = 1,000,000

    parent = [[0] * LOG for _ in range(n + 1)]  # 부모노드
    d = [0] * (n + 1)  # 각 노드의 깊이
    c = [0] * (n + 1)  # 노드의 깊이가 계산되었는지 여부
    graph = [[] for _ in range(n + 1)]
    for x, y in pairs:
        graph[x].append(y)
        graph[y].append(x)

    def dfs(x, depth):
        c[x] = True
        d[x] = depth
        for y in graph[x]:
            if c[y]:  # 이미 구했다면
                continue
            parent[y][0] = x
            dfs(y, depth + 1)


    def set_parent():
        dfs(1, 0)
        print(parent)
        for i in range(1, LOG):
            for j in range(1, n+1):
                parent[j][i] = parent[parent[j][i-1]][i-1]


    def LCA(a, b):  # 최소공통조상
        # b가 더 깊도록 설정
        if d[a] > d[b]:
            a, b = b, a

        #깊이 동일하다록
        for i in range(LOG - 1, -1, -1):
            if d[b] - d[a] >= (1 << i): #2 4 8 16
                b = parent[b][i]

        # 부모가 같아지도록
        if a == b:
             return a;

        for i in range(LOG -1, -1, -1):
            if parent[a][i] != parent[b][i]:
                a = parent[a][i]
                b = parent[b][i]
        return parent[a][0] #

    set_parent()  # 루트노드

    for a, b in get:
        print(LCA(a, b))





def gsitm():
    print("")
def street11(A,B):
    #A, B
    A, B = str(A), str(B)
    answer = 0
    left = 0
    right = 0
    i = 0
    while left < len(B) and i < len(A):
        if B[left] == A[i]:
            answer = left
            i += 1
            right = left + 1
            while right < len(B) and B[right] == A[i]:
                right+=1
                i += 1
                if i == len(A):
                    return answer
        left += 1
        i = 0

    return -1
def street12(A):
    answer = 0
    #stack
    k = []
    #queue
    q = collections.deque()
    for a in A:
        if len(k) + 1 == a:
            answer+=1
            k.append(a)

            #for x in q:
            i = 0
            s = len(q)
            while q and i <= s:
                x = q.popleft()
                if len(k) + 1 == x:
                    k.append(x)
                else:
                    q.append(x)
                i += 1
                #break
            # else:
            #     q.
        else:
            q.append(a)


    return answer

def street13(S):#다른장소 시간 공유 - 시차때매
    #group city, sort time assign -> number + city
    dq = collections.defaultdict(list)
    count = collections.Counter()
    S_list = re.split('\n|, ', S)

    answer = list()

    i = 0
    while i < len(S_list):
        dq[S_list[i+1]].append(S_list[i+2])
        count[S_list[i+1]] += 1
        answer.append([S_list[i], S_list[i+1], S_list[i+2]])
        i += 3

    for d in dq:
        dq[d].sort()

    result = []
    for i, a in enumerate(answer):
        c = count[a[1]]
        date_list = dq[a[1]]
        idx = bisect.bisect(date_list, a[2])
        result.append(a[1] + str(idx).zfill(len(str(c))) + "."+ a[0].split('.')[1])

    return '\n'.join(result)



if __name__ == '__main__':
    print(street13(
        'photo.jpg, Warsaw, 2013-09-05 14:08:15\njohn.png, London, 2015-06-20 15:13:22\nmyFriends.png, Warsaw, 2013-09-05 14:07:13\nEiffel.jpg, Paris, 2015-07-23 08:03:02\npisatower.jpg, Paris, 2015-07-22 23:59:59\nBOB.jpg, London, 2015-08-05 00:02:03\nnotredame.png, Paris, 2015-09-01 12:00:00\nme.jpg, Warsaw, 2013-09-06 15:40:22\na.png, Warsaw, 2016-02-13 13:33:50\nb.jpg, Warsaw, 2016-01-02 15:12:22\nc.jpg, Warsaw, 2016-01-02 14:34:30\nd.jpg, Warsaw, 2016-01-02 15:15:01\ne.png, Warsaw, 2016-01-02 09:49:09\nf.png, Warsaw, 2016-01-02 10:55:32\ng.jpg, Warsaw, 2016-02-29 22:13:11'))

    # print(street12([2,1,3,5,4]))
    # print(street12([2,3,4,1,5]))
    # print(street12([1,3,4,2,5]))
    # print(street12([1,4,3,2,5]))

    # print(street11(53,195453786))
    # print(street11(78,195378678))
    # print(street11(57,193786))

    # print(exploreguild([2,3,1,2,2]))
    # print(timeoradd("02984"))
    # print()
    # print(flipnum('0001100'))
    # print(flipnum('11111'))
    # print(flipnum('00000001'))
    # print(flipnum('11001100110011000001'))
    # print(flipnum('11101101'))
    # print(cannotmake([3,2,1,1,9]))
    # print(cannotmake([1, 2, 3, 5,13]))
    # print(bowling([1,3,2,3,2]))
    # print(muzimukbang([3,1,2], 5))
    #
    # print(luckystraing("123402"))
    # print(recostructstr('K1KA5CB7'))
    # print(compressstr("aabbaccc"))
    # print(key([[0, 0, 0], [1, 0, 0], [0, 1, 1]], [[1, 1, 1], [1, 1, 0], [1, 0, 1]]	))
    # print(snake_game(6,3,[[3,4],[2,5],[5,3]],[[3,'D'],[15,'L'], [17,'D']]))
    # print(snake_game(10,4,[[1,2],[1,3],[1,4],[1,5]],[[8,'D'],[10,'D'], [11,'D'],[13,'L']]))
    # print(snake_game(10,5,[[1,5],[1,3],[1,2],[1,6],[1,7]],[[8,'D'],[10,'D'], [11,'D'],[13,'L']]))
    # print(columns_and_beams([[1,0,0,1],[1,1,1,1],[2,1,0,1],[2,2,1,1],[5,0,0,1],[5,1,0,1],[4,2,1,1],[3,2,1,1]]))
    # print(columns_and_beams([[0,0,0,1],[2,0,0,1],[4,0,0,1],[0,1,1,1],[1,1,1,1],[2,1,1,1],[3,1,1,1],[2,0,0,0],[1,1,1,0],[2,2,0,1]]))
    # print(chicken(5,3,[[0,0,1,0,0],[0,0,2,0,1],[0,1,2,0,0],[0,0,1,0,0],[0,0,0,0,2]]))
    # print(chicken(5,2,[[0,2,0,1,0],[1,0,1,0,0],[0,0,0,0,0],[2,0,0,1,1],[2,2,0,1,2]]))
    # print(chicken(5,1,[[1,2,0,0,0],[1,2,0,0,0],[1,2,0,0,0],[1,2,0,0,0],[1,2,0,0,0]]))
    # print(chicken(5,1,[[1,2,0,2,1],[1,2,0,2,1],[1,2,0,2,1],[1,2,0,2,1],[1,2,0,2,1]]))
    # print(inspection(12, [1, 5, 6, 10], [1, 2, 3, 4]))
    # print(inspection(12, 	[1, 3, 4, 9, 10], [3, 5, 7]))
    # print(secret_map(5,[9, 20, 28, 18, 11], 	[30, 1, 21, 17, 28]))
    # print(dart_game("1S2D*3T"))
    # print(cache_cities(3, ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "Jeju", "Pangyo", "Seoul", "NewYork", "LA"]))
    # print(shuttle(1,1,5,["08:00", "08:01", "08:02", "08:03"]))
    # print(shuttle(2,10,2,["09:10", "09:09", "08:00"]))
    # print(news_clustering('FRANCE', 'FRENCH'))
    # print(friends4block(4, 5, ["CCBDE", "AAADE", "AAABF", "CCBBF"]))
    print()
#     print(traffic( [
# "2016-09-15 01:00:04.002 2.0s",
# "2016-09-15 01:00:07.000 2s"
# ]
#
# ))
#     print(traffic([
#         "2016-09-15 20:59:57.421 0.351s",
# "2016-09-15 20:59:58.233 1.181s",
# "2016-09-15 20:59:58.299 0.8s",
# "2016-09-15 20:59:58.688 1.041s",
# "2016-09-15 20:59:59.591 1.412s",
# "2016-09-15 21:00:00.464 1.466s",
# "2016-09-15 21:00:00.741 1.581s",
# "2016-09-15 21:00:00.748 2.31s",
# "2016-09-15 21:00:00.966 0.381s",
# "2016-09-15 21:00:02.066 2.62s"
#     ]
#
#     ))

    # print(two_pointer_sum([1,2,3,2,5], 5))
    # print(two_pointer_two_list([1,3,5], [2,4,6,8]))
    # print(subsum([10,20,30,40,50],3,4))
    # print(get_primenumber(3, 16))
    # print(makepassword(['a', 't', 'c', 'i', 's', 'w'],4))

    # print(lostb('55-50+40'))
    # print(lostb('10+20+30+40'))
    # print(lostb('00009-00009'))
    # print(coin(10, 4200,[1,5,10,50,100,500,1000,5000,10000,50000]))
    # print(ATM([3, 1 ,4 ,3, 2]))
    # print(movie666(2))
    # print(movie666(3))
    # print(movie666(6))
    # print(movie666(187))
    # print(movie666(500))
    # print(blackjak(21, [5,6,7,8,9]))
    # print(blackjak(500, [93, 181, 245, 214, 315, 36, 185, 138, 216, 295]))
    # print(divideSum(256))
    # print(mock_exam([1,2,3,4,5]))
    # print(mock_exam([1,3,2,4,2]))
    # print(find_prime("17"))
    # print(find_prime("011"))
    # print(carpet(10,2))
    # print(carpet(8,1))
    # print(carpet(24,24))
    # print(training(5, [2,4], [1,3,5]))
    # print(training(5, [2,4], [3]))
    # print(training(3, [3], [1]))
    # print(joystick("JEROEN"))
    # print(joystick("JAN"))
    # print(joystick("JAZ"))
    # print(joystick("ABABAABA"))
    # print(joystick("AAAABABAAAA"))

    # print(makebigest("1924", 2))
    # print(makebigest("1231234", 3))
    # print(makebigest("4177252841", 4))
    # print(life_boat([70, 50, 80, 50], 100))
    # print(life_boat([70, 50, 80], 100))
    # print(link_island(4, [[0,1,1],[0,2,2],[1,2,5],[1,3,1],[2,3,8]]))
    # print(surveillance([[-20,-15], [-14,-5], [-18,-13], [-5,-3]]))

    # print(kth([1, 5, 2, 6, 3, 7, 4],[[2, 5, 3], [4, 4, 1], [1, 7, 3]]))
    # print(biggest([6,10,2]))
    # print(biggest([3, 30, 34, 5, 9]))
    # print(h_index([3, 0, 6, 1, 5]))

    # print(longest(6, [[3, 6], [4, 3], [3, 2], [1, 3], [1, 2], [2, 4], [5, 2]]))
    # print(ranking(5, [[4, 3], [4, 2], [3, 2], [1, 2], [2, 5]]))
    # print(room_number([6, 6, 6, 4, 4, 4, 2, 2, 2, 0, 0, 0, 1, 6, 5, 5, 3, 6, 0]))

    # print(lcm(6,15))
    # print(lcm(1,45000))
    # print(lcm(13,17))
    # print(gcd(24, 18))
    # print(lcm(24, 18))
    # print(binonmial_coefficient(5,2))
    # print(bridge(2,2))
    # print(bridge(1,5))
    # print(bridge(13, 29))


    # print(sugar(18))
    # print(sugar(4))
    # print(sugar(6))
    # print(sugar(9))
    # print(sugar(11))

    # print(vertran(1))
    # print(vertran(10))
    # print(vertran(13))
    # print(vertran(100))

    # print(snail(2,1,5))
    # print(snail(5,1,6))
    # print(snail(100,99,1000000000))
    # print(ACMhotel(6,12,10))
    # print(ACMhotel(6,12,12))
    # print(ACMhotel(30,50,72))
    # print(get_prime(3, 16))
    # print(add_cycle(26))
    # print(add_cycle(55))
    # print(add_cycle(1))
    # print(add_cycle(0))
    # print(add_cycle(71))
    # print(Fly_me_to_the_Alpha_Centauri(0,3))
    # print(Fly_me_to_the_Alpha_Centauri(1,5))
    # print(Fly_me_to_the_Alpha_Centauri(45,50))
    #
    # print(turet(0, 0 ,13, 40, 0, 37))
    # print(turet(0, 0, 3 ,0 ,7 ,4))
    # print(turet(1, 1, 1, 1, 1, 5))


    # print(bracket("(())())"))
    # print(bracket("(((()())()"))
    # print(bracket("(()())((()))"))
    # print(bracket("((()()(()))(((())))()"))
    # print(bracket("()()()()(()()())()"))
    # print(bracket("(()((())()("))
    # print(bracket("(("))
    # print(bracket("))"))
    # print(bracket("))(()"))
    #
    # print(bracket2("So when I die (the [first] I will see in (heaven) is a score list)."))
    # print(bracket2("[ first in ] ( first out )."))
    # print(bracket2("Half Moon tonight (At least it is better than no Moon at all]."))
    # print(bracket2("A rope may form )( a trail in a maze."))
    # print(bracket2("([ (([( [ ] ) ( ) (( ))] )) ])."))
    # print(bracket2(" ."))
    # print(bracket2("."))

    #
    # print(rotated_queue(10, [1,2,3]))
    # print(rotated_queue(10, [2,9,5]))
    # print(rotated_queue(32, [27, 16, 30, 11, 6, 23]))
    # print(rotated_queue(10, [1, 6, 3, 2, 7, 9, 8, 4, 10, 5]))


    # print(yosepus(7,3))


    # print(bellman_ford(3, 4,[[1, 2, 4],ㅂ
    #                         [1, 3, 3],
#                             [2, 3, -1],
#                             [3, 1, -2]]))
#
#     print(bellman_ford(3, 4,
#                 [[1, 2, 4],
#                 [1, 3 ,3],
#                 [2, 3 ,-4],
#                 [3, 1 ,-2]]))
#
#     print(bellman_ford(3, 2,
#                         [[1, 2, 4],
#                         [1, 2, 3]]))
#     print(leastCommonAncestor(15, [[1, 2],
#                                     [1, 3],
#                                     [2, 4],
#                                     [3, 7],
#                                     [6, 2],
#                                     [3, 8],
#                                     [4, 9],
#                                     [2, 5],
#                                     [5, 11],
#                                     [7, 13],
#                                     [10, 4],
#                                     [11, 15],
#                                     [12, 5],
#                                     [14, 7]],[[ 6, 11],
#                                             [10, 9],
#                                             [2, 6],
#                                             [7, 6],
#                                             [8, 13],
#                                             [8, 15]]))
#     print(leastCommonAncestor2(15, [[1, 2],
#                                    [1, 3],
#                                    [2, 4],
#                                    [3, 7],
#                                    [6, 2],
#                                    [3, 8],
#                                    [4, 9],
#                                    [2, 5],
#                                    [5, 11],
#                                    [7, 13],
#                                    [10, 4],
#                                    [11, 15],
#                                    [12, 5],
#                                    [14, 7]], [[6, 11],
#                                               [10, 9],
#                                               [2, 6],
#                                               [7, 6],
#                                               [8, 13],
#                                               [8, 15]]))

    # print(binaryIndexTree(5,2,2,[1,2,3,4,5], [[1,3,6],[2,2,5],[1,5,2],[2,3,5]]))
    # print(heapsort([5,4,3,2,1]))

    # print(disk_controller([[0, 3], [1, 9], [2, 6]]))
    # print(twoPriorityQueue(["I 16","D 1"]))
    # print(twoPriorityQueue(["I 7","I 5","I -5","D -1"]))
    # print(twoPriorityQueue(["I 16", "I -5643", "D -1", "D 1", "D 1", "I 123", "D -1"]))
    # print(twoPriorityQueue(["I -45", "I 653", "D 1", "I -642", "I 45", "I 97", "D 1", "D -1", "I 333"]))


    # print(func_develop([93, 30, 55], [1, 30, 5]))
    # print(func_develop([95, 90, 99, 99, 80, 99]	, [1, 1, 1, 1, 1, 1]))

    # print(printer([2, 1, 3, 2], 2))
    # print(printer([1, 1, 9, 1, 1, 1], 0))
    # print(stock([1, 2, 3, 2, 3]))
    # print(truck(2, 10,[7,4,5,6]))
    # print(truck(100, 100,[10]))
    # print(truck(100, 100,[10,10,10,10,10,10,10,10,10,10]))

    # print(player(["leo", "kiki", "eden"], 	["eden", "kiki"]))
    # print(disguise([["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]]))
    # print(disguise([["crowmask", "face"], ["bluesunglasses", "face"], ["smoky_makeup", "face"]]))
    # print(best_album(["classic", "pop", "classic", "classic", "pop"], [500, 600, 150, 800, 2500]))

    # print(single_number([2,2,1]))
    # print(single_number([4,1,2,1,2]))
    # print(hamming_distance(1,4))
    # print(sumoftwointergers(1,2))
    # print(sumoftwointergers(-2,3))

    print(utf_8_validation([197,130,1]))
    print(utf_8_validation([235,140,4]))
    #
    print(number_of_1bits("00000000000000000000000000001011"))
    print(number_of_1bits("00000000000000000000000010000000"))
    print(number_of_1bits("11111111111111111111111111111101"))