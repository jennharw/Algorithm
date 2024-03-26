
def test(n,r,c):
    def sum_generator(N):
        return sum(n for n in range(1, N + 1))

    def sum_generator_reverse(N, end):
        return sum(n for n in range(N, end))

    p = 0 # 또는 1
    # 1+1 2 1
    # 2+1 1+2 3 23
    # 3+1 2+2 3+1 4 456
    # 4+1 3+2 2+3 1+4 5 78910
    # 5+1 4+2,,, 6 1112131415
    #
    # 5+2 4+3 3+4 5+2 7
    # 5+3 8
    # 5+4 9
    # 5+5 10

    k = r+c
    # print("test", k)
    if k <= n+1:

        start = sum_generator(k-2) #6

        # print("start", start, k)
        if k % 2 == 0:
            return start + k - r
        else:
            return start + r
    else:
        start = sum_generator(n)
        end = sum_generator_reverse(n - (k - n - 2), n)
        start = start + end

        # print("start===", start)

        #26
        if k % 2 == 0: #k 9 r 5 4
            return start + k - r - 1
        else:
            return start + n - c + 1

def test2(price):

    stack = []
    answer = [0] * len(price)
    #[2,1,5,6,2,3]

    for idx, p in enumerate(price):
        # print("-------")
        if stack == []:
            stack.append([idx, p]) #처응에 집어넣어
            # print("stttttt",stack)

        else:
            while stack and p > stack[-1][1]:
                #나갈때 계산
                i, q = stack.pop()
                day = idx - i
                answer[i] = day

            stack.append([idx, p])
            # print("stck", stack)
            # print("answer", answer)
    #나은거 계산
    for vs in stack:
        answer[vs[0]]=-1


    return answer

def test3(N, trees):
    #가로 가장 큰
    #세로 가장 큰p_min = N
    q_min = N
    answer = 0
    trees.sort(key=lambda x:[x[0],x[1]])
    for p, q in trees:
        # print("--p, q", p, q)
        if p_min > p and q_min > q:
            answer += 1
            p_min = p
            q_min = q
            # print(p_min, q_min)
        elif p_min > p :
            answer += 1
            p_min = p
        elif q_min > q:
            answer += 1
            q_min  = q

    return answer



if __name__ == '__main__':
    # print(test(5,3,2))
    #
    # print(test(6,5,2))
    #
    # print(test(5,4,2))
    # print(test(6,5,4))
    # print(test(6,3,5))
    # print(test2([4,1,4,7,6]))
    # print(test2([13,7,3,7,5,16,12]))

    # print(test3(5, [[4, 3], [3, 1], [2, 2], [1, 4]]))
    # print(test3(5, [[3, 3], [2, 2], [1, 1]]))

    if 0:
        print("true")
    else:
        print("false")
    print(ord("A"))
    print(2**3)
    print(8<<1)
    print(8 >> 1)
    print(1^1)