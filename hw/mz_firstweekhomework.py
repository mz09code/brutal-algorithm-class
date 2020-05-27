####################question 1：matrix multiplication####################

import numpy as np

# a 3*4 matrix
matrix_one = [[6,7,8,9],
              [7,6,5,4],
              [3,6,8,9]]
# a 4*5 matrix
matrix_two = [[5,7,9,4,6],
              [6,8,9,3,2],
              [9,6,3,5,6],
              [8,5,9,6,7]]
# a 2*3 matrix
matrix_three = [[2,3,4],[5,6,7]]
# a 1*5 matrix
matrix_four = [[1,23,4,6,7]]
# a 2D list not rectangular
matrix_five = [[1,2,3],
               [4,5,6],
               [7,8,9,11]]

def matrix_product(list_one, list_two):
    if len(list_one) == len(list_two) == 0:
        print("The product of two empty matrices is []")
        matrix_result = []
        return matrix_result
    elif not all(len(i) == len(list_one[0]) for i in list_one) or not all(len(i) == len(list_two[0]) for i in list_two) :
        print("Not rectagngular.")
        return False
    else:
        if len(list_one[0]) == len(list_two):
            matrix_result = [[sum(row*col for row,col in zip(row_one,col_two))for col_two in zip(*list_two)]for row_one in list_one]
            return matrix_result
        elif len(list_one) == len(list_two[0]):
            matrix_result = [[sum(row*col for row,col in zip(row_two,col_one))for col_one in zip(*list_one)]for row_two in list_two]
            return matrix_result
        else:
            print("The numbers of rows/columns in the first matrix should be equal to those of second matrix's. ")
            matrix_result =[]
            return matrix_result     

 # Check if equal    
def matrix_check(productionResult,numpyResult):
    if np.array(productionResult).all() == numpyResult.all():
        return True
    return False

matrix_product([],[])  # two empty matrices,return []

productionOne = matrix_product(matrix_one,matrix_two) # 3*4 matrix × 4*5 matrix = 3*5, from func matrix_production
numpyOne = np.dot(np.array(matrix_one),np.array(matrix_two)) # 3*4 matrix × 4*5 matrix from Numpy
print(matrix_check(productionOne,numpyOne)) # True, matrix_production is the same as the one in Numpy

productionTwo = matrix_product(matrix_one,matrix_three) # 3*4 matrix × 2*3 matrix = 2*4,from func matrix_production
numpyTwo = np.dot(np.array(matrix_three),np.array(matrix_one)) # 3*4 matrix × 2*3 matrix from Numpy
print(matrix_check(productionTwo,numpyTwo)) # True, matrix_production is the same as the one in Numpy

# matrix_one is 3*4(H1*W1), matrix_four is 1*5(H2,W2)
#W1!=H2 AND W2!=H1, cannot process the matrix_procuction. print error message, return []
matrix_product(matrix_one,matrix_four) 

# matrix_one is 3*4(H1*W1), matrix_five is not rectangular
# cannot process the matrix_procuction. print error message, return false
matrix_product(matrix_one,matrix_five)

'''
复杂度分析
两个矩阵相乘，要满足：其中一个的 行/列 等于 另一个的 列/行
一个矩阵是M*N,另一个是O*P，假设N=O，结果得到一个M*P矩阵
sum()是O(1),里面的for loop是O(N),中间for loop是O(P)，外面for loop是O(M)
就是((O(1)*O(N))*O(P))*O(M)=O(M*N*P)=O(n^3)
'''

####################question 2 : quicksort shuffle####################

import random
import statistics as st

def quicksort2(q):
    less=[]
    greater=[]
    if len(q) <= 1:
        return q
    pivot = random.randint(0, len(q))
    for i in q[0:pivot]+ q[pivot+1:]:
        if random.randint(0, 1)>0.5:
            less.append(i)
        else:
            greater.append(i)
    return quicksort2(less)+q[pivot:pivot+1]+quicksort2(greater)


def test(run_time):
    first_list= [0,1,2,3,4,5,6,7,8,9]
    results =[quicksort2([0,1,2,3,4,5,6,7,8,9]) for i in range(run_time)]
    status = [[0 for i in range(10)] for j in range(10)]

    for result in results:
        for j in range(10):
            for i in range(10):
                if first_list[j] == result[i]:
                    status[i][j] += 1
                else:

                    status[i][j] +=0
    return status



# run quicksort shuffle 10000 times
a=test(10000)
# print the random distribution of 0~9
for count,row in enumerate(a):
    print(count,row)
# calculate standard deviation
for col in zip(*a):
    print(col)
    varianceResult = st.stdev(col)
    print(varianceResult)
'''
复杂度分析
用quicksort来洗牌
任意选取一个基数，剩下的数任意放到小的那部分或者大的那部分
小的那部分和大的那部分再quicksort洗牌
返回由小的那部分，基数，大的那部分 组成的数组
O(nlogn)
'''
