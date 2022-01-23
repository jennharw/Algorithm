# Euler's Theorem  - 모든 정점이 짝수개의 차수를 갖는다면 모든 다리를 한번씩만 건너서 도달하는 것이 성립
# ?
# Eulerian Trail, path
# 모든 간선을 한번씩 방문하는 유한 그래프
#
#
# HamiltonPAth,
# 오일러 경로는 간선을 기준으로
# 해밀터 ㄴ경로는 정점을 기준으로
# 히밀턴 경로는 각 정점을 한번씩 방문하는 무향 또는유향 그래프 경로
#
# 원래의 출발점으로 돌아오는 경로 Hamiltonian cycle - travelling salesman parblem - 각 도시를 방문하고 돌아오는 가장 짧은 경로
#
#
# 20!
#
# 다이나믹 프로그래미ㅏㅇ의 경우 최적화
#
#
# graph search
#
# dfs
# bfs - 최단경로 Diakisstra?네트어크 딜레이
#
# stack, recursion 보토 recursion
#
# bfs queue
#
#
#
# 인접행렬, 인접리스트
# adjacency matrix, adjacency list


#1 dfs (stack, recursion)
# import torch
# from torch_geometric.data import Data
#
# x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float) #feature
# edge_index = torch.tensor([[0, 1, 0, 9, 8, 7, 1, 8, 10, 7, 10, 11, 7, 11, 6, 7, 3, 7, 3, 5, 5, 6, 3, 4, 2, 3],
#                            [1, 0, 9, 0, 7, 8, 8, 1, 7, 10, 11, 10, 11, 7, 7, 6, 7, 3, 5, 3, 6, 5, 4, 3, 3, 2]], dtype=torch.long)
#
# data = Data(edge_index=edge_index)
#
# ret = []
# def dfs(node = ""): #모든 경우
#     if node == "":
#         node = 0
#     visited[node] = True
#     ret.append(node)
#     neighbors = edge_index[node][1]
#     for neighbor in neighbors:
#         if visited[neighbor] == False:
#             dfs(neighbor)
#     return ret, visited
#
# def dfs(G, node): # stack
#     visited= []
#     ret = []
#
#     ret.append(node)
#     stack = []
#     stack.append(node)
#     while stack:
#         p = stack.pop()
#         neighbors =  p 의 neiughbors
#         ret.append(p)
#
#
#         for neighbor in neighbors:
#             #validation .ㅑif ~
#                 stack.append(node)
#
#     return ret
#
#
#
# def bfs(graph, start, to): #최단경로


#32 섬의 개수
class GraphProblem:
    def numIslands(self, grid):
        #bfs 재귀 를 통해 component 구하기
        left  = len(grid)
        right = len(grid[0])

        visited = [[False for _ in range(right)] for _ in range(left)]

        #ret = [[False for _ in range(right)] for _ in range(left)]

        def dfs(l , r):

            #ret[l][r] = True
            visited[l][r] = True
            i = [1, 0, -1, 0]
            j = [0, 1, 0, -1]

            for k in range(len(i)):
                if 0 <= l + i[k] < left and 0 <=  r + j[k] < right:
                    if grid[l + i[k]][r + j[k]] == "1" and visited[l + i[k]][r + j[k]] == False:
                        dfs(l + i[k], r + j[k])
            return visited

        count = 0

        #print(dfs(l = 0, r = 0))
        for p in range(4):
            for q in range(5):
                if grid[p][q] == "1" and visited[p][q] == False:
                    dfs(p, q)
                    count+=1
        # print(visited)
        # print(count)
        # print(dfs(l=2, r=2))
        # print(dfs(l=3, r=3))
        return count
        #print(neighbors)

    #순열, 조합 -> 수를 세는 것은 쉬우나, 조합을 생성할 때 dfs
    def letterCombinationsOfPhoneNumber(self, digits:str):
        #3 * 3
        def dfs(index, path):
            if len(path) == len(digits):
                result.append(path)
                return
            for i in range(index, len(digits)):
                for j in dict[digits[i]]:
                    dfs(i+1, path+j)

        if not digits:
            return []
        dict = {
            "2": "abc", "3": "def", "4": "ghi", "5": "jkl",
            "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"
        }
        result = []
        dfs(0, "")
        return result

    def permutation(self, l):
        # node = l[0]
        # print(node)
        # l.pop(0)
        print(l[:0] + l[1:] )
        print(l[:1] + l[2:] )
        print(l[:2] + l[3:] )
        # neighbors = l
        # print(neighbors)
        # result = []
        # path = []
        # def dfs(index, path):
        #     if path == len(l):
        #         print(path)
        #         result.append(path)
        #         print("result", result)
        #         return
        #     path.append(index)
        #     node = l[index]
        #     l.remove(node)
        #     for neighbor in neighbors:
        #         path.append(neighbor)
        #         dfs(neighbor, path)
        # print(dfs(0, path))
        # #
        # #

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    gp = GraphProblem()
    grid = [
        ["1", "1", "1", "1", "0"],
        ["1", "1", "0", "1", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "0", "0", "0"]
    ]
    grid2 = [
        ["1", "1", "0", "0", "0"],
        ["1", "1", "0", "0", "0"],
        ["0", "0", "1", "0", "0"],
        ["0", "0", "0", "1", "1"]
    ]

    #print(gp.numIslands(grid))
    #print(gp.numIslands(grid2))

    #print(gp.letterCombinationsOfPhoneNumber("23"))

    print(gp.permutation([1,2,3]))