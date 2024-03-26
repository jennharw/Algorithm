import collections

# 프로그래머스
# 백준
# 해커랭크
# leetcode

# 카카오
# 신고결과받기 - 정확도
def solution(id_list, report, k):
    dp = collections.defaultdict(set)
    c = collections.Counter()
    for r in report:
        ids = r.split(' ')
        if ids[0] not in dp[ids[1]]:
            dp[ids[1]].add(ids[0])
            c[ids[1]] += 1

    ids = collections.Counter(id_list)
   # print(c)
    for v in c.keys():
        if c[v] >= k:
            users = dp[v]
            for u in users:
                ids[u] += 1
    answer = []
    for i in ids.values():
        answer.append(i - 1)
    return answer


     #c.values() > k

https://programmers.co.kr/learn/challenges

if __name__ == '__main__':
    print(solution(["muzi", "frodo", "apeach", "neo"], ["muzi frodo","apeach frodo","frodo neo","muzi neo","apeach muzi"], 2))
    print(solution(["con", "ryan"],["ryan con", "ryan con", "ryan con", "ryan con"], 3))
