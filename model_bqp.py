import numpy as np
import matplotlib.pyplot as plt

class quad_mesh:
    def __init__(self, N, rand_check):
        self.N = N
        self.points = np.zeros((N,N,3))
        self.weights = np.ones((N,N))
        self.check = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                self.points[i,j,0] = i+1
                self.points[i,j,1] = N-j
                self.points[i,j,2] = 1
        if rand_check:
            rand_mat = (np.random.random((N,N,2)) - 1/2)/2
            for i in range(N):
                for j in range(N):
                    self.points[i,j,0] = self.points[i,j,0] + rand_mat[i,j,0]
                    self.points[i,j,1] = self.points[i,j,1] + rand_mat[i,j,1]

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

    def find_max(self):
        max = 0
        for i in range(self.N):
            for j in range(self.N):
                if self.points[i][j][0] > max:
                    max = self.points[i][j][0]
                if self.points[i][j][1] > max:
                    max = self.points[i][j][1]
        self.max = max
        return max

    def find_parameter(self, q00, q01, q11, q10, point):
        point_temp = np.zeros(3)
        point_temp[0] = point[0]
        point_temp[1] = point[1]
        point_temp[2] = 1
        a_s = np.cross(
            np.cross(self.points[q00[0]][q00[1]][:],self.points[q01[0]][q01[1]][:]),
            np.cross(self.points[q10[0]][q10[1]][:],self.points[q11[0]][q11[1]][:]))
        a_t = np.cross(
            np.cross(self.points[q00[0]][q00[1]][:],self.points[q10[0]][q10[1]][:]),
            np.cross(self.points[q01[0]][q01[1]][:],self.points[q11[0]][q11[1]][:]))
        param = np.zeros(2)
        param[0] = (np.dot(np.cross(a_s,self.weights[q01[0]][q01[1]]*self.points[q01[0]][q01[1]][:]), point_temp)/
                    (np.dot(np.cross(a_s,self.weights[q01[0]][q01[1]]*self.points[q01[0]][q01[1]][:]), point_temp) -
                            np.dot(np.cross(a_s,self.weights[q11[0]][q11[1]]*self.points[q11[0]][q11[1]][:]), point_temp)))
        param[1] = (np.dot(np.cross(a_t,self.weights[q10[0]][q10[1]]*self.points[q10[0]][q10[1]][:]), point_temp)/
                    (np.dot(np.cross(a_t,self.weights[q10[0]][q10[1]]*self.points[q10[0]][q10[1]][:]), point_temp) -
                            np.dot(np.cross(a_t,self.weights[q11[0]][q11[1]]*self.points[q11[0]][q11[1]][:]), point_temp)))
        return param

    def count_parameter(self, q00, q01, q11, q10, param):
        res_temp = ((self.weights[q00[0]][q00[1]]*self.points[q00[0]][q00[1]][:])*(1 - param[0])*(1 - param[1])+
               (self.weights[q10[0]][q10[1]]*self.points[q10[0]][q10[1]][:])*param[0]*(1 - param[1])+
               (self.weights[q01[0]][q01[1]]*self.points[q01[0]][q01[1]][:])*(1 - param[0])*param[1]+
               (self.weights[q11[0]][q11[1]]*self.points[q11[0]][q11[1]][:])*param[0]*param[1])
        res = np.zeros(2)
        res[0] = res_temp[0]/res_temp[2]
        res[1] = res_temp[1]/res_temp[2]
        return res
    def locate_point(self, point, check_quad):
        if check_quad[0] >= 0:
            param = self.find_parameter([check_quad[0],check_quad[1]+1],[check_quad[0],check_quad[1]],
                                        [check_quad[0]+1,check_quad[1]],[check_quad[0]+1,check_quad[1]+1], point)
            if((param[0] >= 0) and (param[1] >= 0) and (param[0] <= 1) and (param[1] <= 1)):
                return [[check_quad[0],check_quad[1]],param]
        for i in range(self.N - 1):
            for j in range(self.N - 1):
                param = self.find_parameter([i,j+1],[i,j],[i+1,j],[i+1,j+1], point)
                if((param[0] >= 0) and (param[1] >= 0) and (param[0] <= 1) and (param[1] <= 1)):
                    return [[i,j],param]
        return [[float('nan'), float('nan')],[float('nan'), float('nan')]]