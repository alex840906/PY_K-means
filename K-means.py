import numpy as np
import random
data_size = 150
dim = 4
k = 3
iris = np.loadtxt('iris.txt', delimiter=' ')
random_point = [-1]*k
ini_path = []
j = 1


def evaluate_distance_from_centroid(centroid_point, distance):
    for i in range(data_size):
        for j in range(k):
            for dimention in range(dim):
                distance[i][j] += (centroid_point[j]
                                   [dimention]-iris[i][dimention])**2
            distance[i][j]**0.5


def partition(distance):
    for i in range(data_size):
        min_distance = distance[i][0]
        for j in range(k):
            if distance[i][j] <= min_distance:
                min_distance = distance[i][j]
                part[i] = j


def find_new_centroid(centroid_point):
    centroid_point = np.zeros([k, dim])
    count = np.zeros([k])
    for i in range(data_size):
        for j in range(k):
            if part[i] == j:
                for h in range(dim):
                    centroid_point[j][h] += iris[i][h]
                count[j] += 1
    for i in range(k):
        for j in range(dim):
            centroid_point[i][j] /= int(count[i])
    return centroid_point


def find_SSE(centroid_point, SSE_distance):
    for i in range(data_size):
        for h in range(dim):
            SSE_distance += (iris[i][h]-centroid_point[int(part[i])][h])**2
    return SSE_distance


def find_iris_centroid(centroid_point):
    for i in range(data_size):
        for j in range(k):
            if iris[i][dim] == j+1:
                for h in range(dim):
                    centroid_point[j][h] += iris[i][h]
    for i in range(k):
        for j in range(dim):
            centroid_point[i][j] /= 50
    return centroid_point


def evaluate_distance_between_centroid(centroid_point, iris_centroid_point):
    distance_between_centroid = np.zeros([k, k])
    for i in range(k):
        for j in range(k):
            for h in range(dim):
                distance_between_centroid[i][j] = (
                    centroid_point[i][h]-iris_centroid_point[j][h])**2
            distance_between_centroid[i][j] = (
                distance_between_centroid[i][j])**0.5
    return distance_between_centroid


def determine_centroid(distance_between_centroid):
    correspond_centroid = np.zeros([k])
    new_part = np.zeros([data_size])
    for i in range(k):
        min_distance = 100
        for j in range(k):
            if (min_distance > distance_between_centroid[i][j]):
                min_distance = distance_between_centroid[i][j]
                correspond_centroid[i] = j+1

    for i in range(data_size):
        for j in range(k):
            if part[i]==j:
                new_part[i]=correspond_centroid[j]      
    return new_part

def calculate_accuracy(new_part):
    right=0
    for i in range(data_size):
        if new_part[i]==iris[i][dim]:
            right+=1

    accuracy=0.0
    accuracy=(right/data_size)*100
    return accuracy


centroid_point = np.zeros([k, dim+1])
i = 0
while i < k:
    r = random.randint(0, data_size-1)
    flag = 1
    for j in range(k):
        if random_point[j] == r:
            flag == 0
    if flag == 1:
        random_point[i] = r
        centroid_point[i, :] = np.copy(iris[r, :])
        i += 1

itteration = 0
run=0
total_SSE=0.0
total_accuracy=0.0
while run < 15:
    while itteration < 15:
        distance_from_centroid = np.zeros([data_size, k])
        evaluate_distance_from_centroid(centroid_point, distance_from_centroid)

        part = np.zeros([data_size])

        partition(distance_from_centroid)
        centroid_point = find_new_centroid(centroid_point)

        SSE_distance = 0
        SSE_distance = find_SSE(centroid_point, SSE_distance)

        iris_centroid = np.zeros([k, dim])
        iris_centroid = find_iris_centroid(iris_centroid)

        distance_between_centroid = np.zeros([k, k])
        distance_between_centroid = evaluate_distance_between_centroid(
            centroid_point, iris_centroid)

        correspond_centroid = np.zeros([k])
        new_part = determine_centroid(distance_between_centroid)

        accuray=0.0
        accuray=calculate_accuracy(new_part)

        itteration += 1

    total_SSE+=SSE_distance
    total_accuracy+=accuray
    run+=1
total_accuracy/=run
total_SSE/=run

print('run =',run)
print('itteration =',itteration)
print('SSE :',total_SSE)
print('Accuracy :',total_accuracy)

