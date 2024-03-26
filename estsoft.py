import collections
import itertools


def solution(scores):
    #F 2 불합격
    #A 2 합격
    # 최저점과 최고점을 하나씩 제외한 나머지 점수의 평균이 3점 이상이면 합격, 3점 미만이면 불합격 처리
    # A=5점, B=4점, C=3점, D=2점, E=1점, F=0점
    # O(n) 으로

    result = 0
    dic = {'A':5, 'B':4, 'C':3, 'D':2, 'E':1, 'F':0}
    l  = len(scores[0]) - 2

    for score in scores:

        counter = collections.Counter(score)
        if counter['F'] >= 2:
            continue
        if counter['A'] >= 2:
            result+=1
            continue

        counter.subtract(max(counter.keys()))
        counter.subtract(min(counter.keys()))

        s = 0
        for c in counter:
            s += dic[c] * counter[c]
        if s/l >= 3:
            result += 1

    return result
import pandas as pd
def solution2(needs, r):
    #x y 0 필요 없으, 필요 있으
    #결과 -> 최대로 구매할 로봇골랏을 경우 만들수 있는 완제품 수.

    l = len(needs[0])  # - 1
    l = [i for i in range(l)]
    robot = list(itertools.combinations(l,r))
    counter = collections.Counter()
    for nx in needs:
        indexes = []
        for n in range(len(nx)):
            if nx[n] ==1:
                indexes.append(n)
        for r in robot:
            if all(item in list(r) for item in indexes):
                counter[r] += 1
    return counter.most_common(1)[0][1]



def solution3(record):
    INF = int(1e9)
    result = collections.defaultdict()
    empty_size = collections.defaultdict()


    for rec in record:
        rec = rec.split(' ')
        pid = int(rec[0].split('id=')[1])

        print("result : " , result)
        print(empty_size)

        if rec[1] == 'leave':

            pos = result[pid][1] #현재 위치
            empty = empty_size[pos] #위치, k 좌 우 우 차이

            empty_size[empty[1]][2] = empty[2]  #다으 사이즈는 증가
            empty_size[empty[1]][3] = empty[2] - empty[1] #

            del empty_size[pos]

            del result[pid]
            continue

        k = int(rec[2].split('k=')[1])


        if empty_size == {}:
            result[pid] = [pid, k]
            empty_size[k] = [k, 0, INF, INF]#자신의 위치 :  k 거리, 좌, 우, 우 차이; 가장 왼쪽부터
            continue

        for empty in sorted(empty_size.keys()):#가장 좌에 삽입

            print(empty) #자신의 위치: k 거리, 좌, 우, 우 차이
            if 2*k + 1 <= empty_size[empty][3]: #5

                if k < empty_size[empty][0]:
                    p = empty_size[empty]
                    result[pid] = [pid, empty + p[0] + 1]  # 원래 위치 + k
                    empty_size[empty + p[0] + 1] = [k, empty, p[2], p[2] - empty + p[0] + 1]

                    empty_size[empty][2] = empty + p[0] + 1 #우
                    empty_size[empty][3] = empty + p[0] + 1 - empty #우 차이
                    break

                else:
                    p = empty_size[empty]
                    result[pid] = [pid, empty + k + 1]  # 원래 위치 + k
                    empty_size[empty + k + 1] = [k, empty, p[2], p[2] - empty - k - 1]  # k 좌

                    empty_size[empty][2] = empty + k + 1 #우
                    empty_size[empty][3] = k + 1  #우 차이
                    break


    answer = []
    for r in result:
        answer.append(result[r])

    answer.sort( key=lambda x: x[0])
    return answer



if __name__ == '__main__':
    # print(solution(["AAFAFA", "FEECAA", "FABBCB", "CBEDEE", "CCCCCC"]))
    # print(solution(["BCD","ABB","FEE"]))
    #print(solution2([ [ 1, 0, 0 ], [1, 1, 0], [1, 1, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1] ], 2))
    print(solution3(["id=1 sit k=1","id=2 sit k=3","id=3 sit k=2","id=2 leave","id=4 sit k=4","id=5 sit k=2"]))
