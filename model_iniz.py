import numpy as np
import matplotlib.pyplot as plt
from math import ceil

class quad_mesh:
    def __init__(self, N):
        self.N = N
        self.points = np.zeros((N,N,3))
        self.weights = np.ones((N,N))
        self.check = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                self.points[i,j,0] = i+1
                self.points[i,j,1] = N-j
                self.points[i,j,2] = 1
        rand_mat = (np.random.random((N,N,2)) - 1/2)/2
        for i in range(N):
            for j in range(N):
                self.points[i,j,0] = self.points[i,j,0] + rand_mat[i,j,0]
                self.points[i,j,1] = self.points[i,j,1] + rand_mat[i,j,1]

    def plot_mesh(self):
        N = self.points.shape[0]
        p = 0
        x = [0]*(N*N)
        y = [0]*(N*N)
        plt.axis([0, N+1, 0, N+1])
        for i in range(N):
            for j in range(N):
                x[p] = self.points[i,j,0]
                y[p] = self.points[i,j,1]
                p = p + 1
        plt.plot(x,y,'ro')
        for i in range(N):
            for j in range(N):
                plt.annotate('{}'.format(ceil(self.weights[i][j]*1000)/1000.0), xy=(self.points[i][j][0],self.points[i][j][1]), xytext=(-5, 5), ha='right', textcoords='offset points')

        for i in range(N):
           for j in range(N-1):
               plt.plot([self.points[i,j,0], self.points[i,j+1,0]], [self.points[i,j,1], self.points[i,j+1,1]], color = 'red')

        for j in range(N):
            for i in range(N-1):
                plt.plot([self.points[i,j,0], self.points[i+1,j,0]], [self.points[i,j,1], self.points[i+1,j,1]], color = 'red')

        plt.show()

    def quad_change(self, q00, q01, q11, q10):
        p_temp = (np.dot(np.cross(self.points[q01[0]][q01[1]][:],self.points[q00[0]][q00[1]][:]), self.points[q10[0]][q10[1]][:])*
                  np.dot(np.cross(self.points[q10[0]][q10[1]][:],self.points[q11[0]][q11[1]][:]), self.points[q01[0]][q01[1]][:])/
                  np.dot(np.cross(self.points[q11[0]][q11[1]][:],self.points[q01[0]][q01[1]][:]), self.points[q00[0]][q00[1]][:])/
                  np.dot(np.cross(self.points[q00[0]][q00[1]][:],self.points[q10[0]][q10[1]][:]), self.points[q11[0]][q11[1]][:]))
        if not self.check[q00[0]][q00[1]]:
            self.weights[q00[0]][q00[1]] = p_temp * self.weights[q01[0]][q01[1]] * self.weights[q10[0]][q10[1]] / self.weights[q11[0]][q11[1]]
        elif not self.check[q01[0]][q01[1]]:
            self.weights[q01[0]][q01[1]] = self.weights[q11[0]][q11[1]] * self.weights[q00[0]][q00[1]] / p_temp / self.weights[q10[0]][q10[1]]
        elif not self.check[q10[0]][q10[1]]:
            self.weights[q10[0]][q10[1]] = self.weights[q11[0]][q11[1]] * self.weights[q00[0]][q00[1]] / p_temp / self.weights[q01[0]][q01[1]]
        elif not self.check[q11[0]][q11[1]]:
            self.weights[q11[0]][q11[1]] = p_temp * self.weights[q01[0]][q01[1]] * self.weights[q10[0]][q10[1]] / self.weights[q00[0]][q00[1]]
        else:
            print("Error!!!")

        self.check[q00[0]][q00[1]] = 1
        self.check[q01[0]][q01[1]] = 1
        self.check[q10[0]][q10[1]] = 1
        self.check[q11[0]][q11[1]] = 1

    def count_weights(self):
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                self.quad_change([i,j+1],[i,j],[i+1,j],[i+1,j+1])
        #print(self.weights)

    def check_weight(self, q00, q01, q11, q10):
        p_temp = (np.dot(np.cross(self.points[q01[0]][q01[1]][:],self.points[q00[0]][q00[1]][:]), self.points[q10[0]][q10[1]][:])*
                  np.dot(np.cross(self.points[q10[0]][q10[1]][:],self.points[q11[0]][q11[1]][:]), self.points[q01[0]][q01[1]][:])/
                  np.dot(np.cross(self.points[q11[0]][q11[1]][:],self.points[q01[0]][q01[1]][:]), self.points[q00[0]][q00[1]][:])/
                  np.dot(np.cross(self.points[q00[0]][q00[1]][:],self.points[q10[0]][q10[1]][:]), self.points[q11[0]][q11[1]][:]))
        return abs(p_temp - self.weights[q00[0]][q00[1]]*self.weights[q11[0]][q11[1]]/self.weights[q01[0]][q01[1]]/self.weights[q10[0]][q10[1]]) < 0.0001

    def check_all_weights(self):
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                print(self.check_weight([i,j+1],[i,j],[i+1,j],[i+1,j+1]))

N = 3
mesh = quad_mesh(N)
mesh.count_weights()
mesh.plot_mesh()