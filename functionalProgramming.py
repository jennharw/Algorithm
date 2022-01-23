# 일급 함수(일급 객체)
# 파이썬 함수 특징
# 1.런타임 초기화
# 2.변수 할당 가능
# 3.함수 인수 전달 가능
# 4.함수 결과 반환 가능

# 함수 객체
def factorial(n):
    '''Factorial Function -> n : int'''
    if n == 1: # n < 2
        return 1
    return n * factorial(n-1)

# 변수 할당
var_func = factorial

print(var_func(10))
print(map(var_func, range(1,11)))
print(list(map(var_func, range(1,6))))


# 함수 인수 전달 및 함수로 결과 반환 -> 고위 함수(Higher-order function)
# map, filter, reduce 등
print(list(map(var_func, filter(lambda x: x % 2, range(1,6)))))
print([var_func(i) for i in range(1,6) if i % 2])


# reduce()
from functools import reduce
from operator import add

print(reduce(add, range(1,11))) # 누적
print(sum(range(1,11)))
#lambda
print(reduce(lambda x, t: x + t, range(1,11)))

# partial 사용법 : 인수 고정 -> 콜백 함수에 사용
from operator import mul
from functools import partial

print(mul(10,10))

# 인수 고정
five = partial(mul, 5)

# 고정 추가
six = partial(five, 6)

print(five(10))
print(six())
print([five(i) for i in range(1,11)])
print(list(map(five, range(1,11))))



# 클래스 이용

# Closure(클로저) 사용 이유
# 서버 프로그래밍 -> 동시성(Concurrency)제어 -> 메모리 공간에 여러 자원이 접근 -> 교착상태(Dead Lock)
# 메모리를 공유하지 않고 메시지 전달로 처리하기 위한 -> Erlang
# 클로저는 공유하되 변경되지 않는(Immutable, Read Only) 적극적으로 사용 -> 함수형 프로그래밍
# 클로저는 불변자료구조 및 atom, STM -> 멀티스레드(Coroutine) 프로그래밍에 강점
class Averager():
    def __init__(self):
        self._series = []

    def __call__(self, v):
        self._series.append(v)
        print('inner >>> {} / {}'.format(self._series, len(self._series)))
        return sum(self._series) / len(self._series)


# 인스턴스 생성
averager_cls = Averager()

# 누적
print(averager_cls(15))
print(averager_cls(35))
print(averager_cls(40))


def closure_ex1():
    # Free variable
    series = []

    # 클로저 영역
    def averager(v):
        # series = [] # 주석 해제 후 확인
        series.append(v)
        print('inner >>> {} / {}'.format(series, len(series)))
        return sum(series) / len(series)

    return averager


avg_closure1 = closure_ex1()

print(avg_closure1(15))
print(avg_closure1(35))
print(avg_closure1(40))


# Nonlocal -> Free variable
def closure_ex3():
    # Free variable
    cnt = 0
    total = 0

    def averager(v):
        nonlocal cnt, total
        cnt += 1
        total += v
        return total / cnt

    return averager


avg_closure3 = closure_ex3()

print(avg_closure3(15))


# 데코레이터 실습
import time

def perf_clock(func):
    def perf_clocked(*args):
        # 함수 시작 시간
        st = time.perf_counter()
        result = func(*args)
        # 함수 종료 시간 계산
        et = time.perf_counter() - st
        # 실행 함수명
        name = func.__name__
        # 함수 매개변수
        arg_str = ', '.join(repr(arg) for arg in args)
        # 결과 출력
        print('[%0.5fs] %s(%s) -> %r' % (et, name, arg_str, result))
        return result
    return perf_clocked

@perf_clock
def time_func(seconds):
    time.sleep(seconds)

@perf_clock
def sum_func(*numbers):
    return sum(numbers)


# 데코레이터 미사용
none_deco1 = perf_clock(time_func)
none_deco2 = perf_clock(sum_func)

print(none_deco1, none_deco1.__code__.co_freevars)
print(none_deco2, none_deco2.__code__.co_freevars)

print('-' * 40, 'Called None Decorator -> time_func')
print()
none_deco1(1.5)
print('-' * 40, 'Called None Decorator -> sum_func')
print()
none_deco2(100, 150, 250, 300, 350)

print()
print()


# 데코레이터 사용
print('*' * 40, 'Called Decorator -> time_func')
print()
time_func(1.5)
print('*' * 40, 'Called Decorator -> sum_func')
print()
sum_func(100, 150, 250, 300, 350)
print()


#Iterator, Generator

# Chapter06 - 01
# 병행성(Concurrency)
# 이터레이터, 제네레이터
# Iterator, Generator

# 파이썬 반복 가능한 타입
# for, collections, text file, List, Dict, Set, Tuple, unpacking, *args

# 반복 가능한 이유? -> iter(x) 함수 호출
t = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

# for 반복
for c in t:
    print(c)

print()

# while 반복

w = iter(t)

while True:
    try:
        print(next(w))
    except StopIteration:
        break

print()

from collections import abc

# 반복형 확인
print(hasattr(t, '__iter__'))
print(isinstance(t, abc.Iterable))

print()
print()


# next 사용
class WordSplitter:
    def __init__(self, text):
        self._idx = 0
        self._text = text.split(' ')

    def __next__(self):
        # print('Called __next__')
        try:
            word = self._text[self._idx]
        except IndexError:
            raise StopIteration('Stopped Iteration.')
        self._idx += 1
        return word

    def __repr__(self):
        return 'WordSplit(%s)' % (self._text)


wi = WordSplitter('Do today what you could do tomorrow')

print(wi)
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
print(next(wi))
# print(next(wi))

print()
print()


# Generator 패턴
# 1.지능형 리스트, 딕셔너리, 집합 -> 데이터 양 증가 후 메모리 사용량 증가 -> 제네레이터 사용 권장
# 2.단위 실행 가능한 코루틴(Coroutine) 구현과 연동
# 3.작은 메모리 조각 사용

class WordSplitGenerator:
    def __init__(self, text):
        self._text = text.split(' ')

    def __iter__(self):
        # print('Called __iter__')
        for word in self._text:
            yield word  # 제네레이터
        return

    def __repr__(self):
        return 'WordSplit(%s)' % (self._text)


wg = WordSplitGenerator('Do today what you could do tomorrow')

wt = iter(wg)

print(wt)
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
print(next(wt))
# print(next(wt))

print()
print()



# 파이썬 반복 가능한 타입
# for, collections, text file, List, Dict, Set, Tuple, unpacking, *args

# Generator Ex1
def generator_ex1():
    print('Start')
    yield 'A Point.'
    print('continue')
    yield 'B Point.'
    print('End')

temp = iter(generator_ex1())

# print(next(temp))
# print(next(temp))
# print(next(temp))

for v in generator_ex1():
    pass
    # print(v)

print()

# Generator Ex2
temp2 = [x * 3 for x in generator_ex1()]
temp3 = (x * 3 for x in generator_ex1())

print(temp2)
print(temp3)

for i in temp2:
    print(i)

print()
print()

for i in temp3:
    print(i)

print()
print()


# Generator Ex3(중요 함수)
# filterfalse, accumulate, chain, product, product, groupby
import itertools

gen1 = itertools.count(1, 2.5)

print(next(gen1))
print(next(gen1))
print(next(gen1))
print(next(gen1))
# ... 무한

print()

# 조건
gen2 = itertools.takewhile(lambda n : n < 1000, itertools.count(1, 2.5))

for v in gen2:
    print(v)


print()

# 필터 반대
gen3 = itertools.filterfalse(lambda n : n < 3, [1,2,3,4,5])

for v in gen3:
    print(v)


print()

# 누적 합계
gen4 = itertools.accumulate([x for x in range(1, 101)])

for v in gen4:
    print(v)

print()

# 연결1
gen5 = itertools.chain('ABCDE', range(1,11,2))

print(list(gen5))

# 연결2

gen6 = itertools.chain(enumerate('ABCDE'))

print(list(gen6))

# 개별
gen7 = itertools.product('ABCDE')

print(list(gen7))

# 연산(경우의 수)
gen8 = itertools.product('ABCDE', repeat=2)

print(list(gen8))

# 그룹화
gen9 = itertools.groupby('AAABBCCCCDDEEE')

# print(list(gen9))

for chr, group in gen9:
    print(chr, ' : ', list(group))

print()

# Futures 동시성
# 비동기 작업 실행
# 지연시간(Block) CPU 및 리소스 낭비 방지 -> (File)Network I/O 관련 작업 -> 동시성 활용 권장
# 비동기 작업과 적합한 프로그램일 경우 압도적으로 성능 향상

# futures : 비동기 실행을 위한 API를 고수준으로 작성 -> 사용하기 쉽도록 개선
# concurrent.Futures
# 1. 멀티스레딩/멀티프로세싱 API 통일 -> 매우 사용하기 쉬움
# 2. 실행중이 작업 취소, 완료 여부 체크, 타임아웃 옵션, 콜백추가, 동기화 코드 매우 쉽게 작성 -> Promise 개념

# 2가지 패턴 실습
# concurrent.futures 사용법1
# concurrent.futures 사용법2

# GIL : 두 개 이상의 스레드가 동시에 실행 될 때 하나의 자원을 엑세스 하는 경우 -> 문제점을 방지하기 위해
#       GIL 실행 , 리소스 전체에 락이 걸린다. -> Context Switch(문맥 교환)

# GIL : 멀티프로세싱 사용, CPython

import os
import time
from concurrent import futures

WORK_LIST = [100000, 1000000, 10000000, 10000000]

# 동시성 합계 계산 메인함수
# 누적 합계 함수(제네레이터)
def sum_generator(n):
    return sum(n for n in range(1, n+1))

def main():
    # Worker Count
    worker = min(10, len(WORK_LIST))
    # 시작 시간
    start_tm = time.time()
    # 결과 건수
    # ProcessPoolExecutor
    with futures.ThreadPoolExecutor() as excutor:
        # map -> 작업 순서 유지, 즉시 실행
        result = excutor.map(sum_generator, WORK_LIST)
    # 종료 시간
    end_tm = time.time() - start_tm
    # 출력 포멧
    msg = '\n Result -> {} Time : {:.2f}s'
    # 최종 결과 출력
    print(msg.format(list(result), end_tm))
