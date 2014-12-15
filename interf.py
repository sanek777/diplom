import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import math
from model_bqp import *

class interface:
    def __init__(self, mesh, img_norm, use_mesh):
        self.img_norm = img_norm
        self.img = np.copy(img_norm)
        self.mesh = mesh
        self.mesh_norm = quad_mesh(mesh.N, False)
        #self.find_img_coor()
        [self.fig, self.ax] = self.plot_first(self.mesh, use_mesh)
        [self.fig1, self.ax1] = self.plot_first(self.mesh_norm, use_mesh)
        self.plot_img()
        self.find_grid()
        cid = self.fig.canvas.mpl_connect('button_press_event', self)
        plt.ion()
        plt.show()

    def find_img_coor(self):
        self.img_coor = np.zeros((self.img_norm.shape[0],self.img_norm.shape[1],2))
        for i in  range(self.img_norm.shape[0]):
            for j in range(self.img_norm.shape[1]):
                self.img_coor[i,j,0] = (self.mesh.N - 1)*i/self.img_norm.shape[0] + 1
                self.img_coor[i,j,1] = (self.mesh.N - 1)*j/self.img_norm.shape[1] + 1


    def find_grid(self):
        min_coor = np.amin(self.mesh.points[:,:,0:2])
        max_coor = np.amax(self.mesh.points[:,:,0:2])
        quad_num = [ -1, -1]
        count_out = 0
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                x = (max_coor - min_coor)/self.img.shape[1]*(j) + min_coor
                y = (max_coor - min_coor)/self.img.shape[0]*(self.img.shape[0] - i - 1) + min_coor
                [quad_num, param] = self.mesh.locate_point([x,y],quad_num)
                if (not np.isnan(quad_num[0])):
                     #param = self.mesh.find_parameter(np.add(quad_num,[0,1]),quad_num,np.add(quad_num,[1,0]), np.add(quad_num,[1,1]),[x,y])
                     point_temp = (self.mesh_norm.points[quad_num[0]][quad_num[1]+1][:]*(1 - param[0])*(1 - param[1])+
                          self.mesh_norm.points[quad_num[0]][quad_num[1]][:]*(1 - param[0])*param[1]+
                          self.mesh_norm.points[quad_num[0]+1][quad_num[1]][:]*param[0]*param[1]+
                          self.mesh_norm.points[quad_num[0]+1][quad_num[1]+1][:]*param[0]*(1 - param[1]))
                     img_j = math.floor(self.img_norm.shape[1]/(self.mesh_norm.N - 1)*(point_temp[0] - 1))
                     img_i = math.floor(self.img_norm.shape[0]/(self.mesh_norm.N - 1)*(point_temp[1] - 1))
                     self.img[i,j] = self.img_norm[self.img_norm.shape[0] - img_i - 1,img_j]
                else:
                     count_out += 1
                     self.img[i,j] = [1,1,1,1]
            print([i, j])
        print(count_out)
        self.ax.imshow(self.img,extent=[min_coor,(max_coor - min_coor)/self.img.shape[1]*(self.img.shape[1] - 1) + min_coor,
                                        min_coor,(max_coor - min_coor)/self.img.shape[0]*(self.img.shape[0] - 1) + min_coor])

    def plot_img(self):
        #for i in  range(self.img.shape[0]):
        #    for j in range(self.img.shape[1]):
        #        self.ax1.plot(self.img_coor[i,j,0],self.img_coor[i,j,1],'ro')
        self.ax1.imshow(self.img_norm,extent=[1,self.mesh_norm.N,1,self.mesh_norm.N])

    def plot_first(self, mesh, use_mesh):
        N = mesh.N
        p = 0
        x = [0]*(N*N)
        y = [0]*(N*N)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis([0, N+1, 0, N+1])
        if use_mesh:
            for i in range(N):
                for j in range(N):
                    x[p] = mesh.points[i,j,0]
                    y[p] = mesh.points[i,j,1]
                    p = p + 1
            ax.plot(x,y,'ro')
            #for i in range(N):
            #    for j in range(N):
            #        ax.annotate('{}'.format(ceil(mesh.weights[i][j]*1000)/1000.0), xy=(mesh.points[i][j][0],mesh.points[i][j][1]), xytext=(-5, 5), ha='right', textcoords='offset points')

            for i in range(N):
               for j in range(N-1):
                   ax.plot([mesh.points[i,j,0], mesh.points[i,j+1,0]], [mesh.points[i,j,1], mesh.points[i,j+1,1]], color = 'red')

            for j in range(N):
                for i in range(N-1):
                    ax.plot([mesh.points[i,j,0], mesh.points[i+1,j,0]], [mesh.points[i,j,1], mesh.points[i+1,j,1]], color = 'red')
        return [fig, ax]

    def __call__(self, event):
        self.ax.plot(event.xdata,event.ydata,'ro')
        [quad_num,param] = self.mesh.locate_point([event.xdata,event.ydata],[-1, -1])
        if (not np.isnan(quad_num[0])):
            #param = self.mesh.find_parameter(np.add(quad_num,[0,1]),quad_num,np.add(quad_num,[1,0]),np.add(quad_num,[1,1]),[event.xdata,event.ydata])
            #self.ax.annotate('{}'.format([ceil(param[0]*1000)/1000.0, ceil(param[1]*1000)/1000.0]), xy=(event.xdata,event.ydata),
            #                xytext=(-5, 5), ha='right', textcoords='offset points')
            point_temp = (self.mesh_norm.points[quad_num[0]][quad_num[1]+1][:]*(1 - param[0])*(1 - param[1])+
                          self.mesh_norm.points[quad_num[0]][quad_num[1]][:]*(1 - param[0])*param[1]+
                          self.mesh_norm.points[quad_num[0]+1][quad_num[1]][:]*param[0]*param[1]+
                          self.mesh_norm.points[quad_num[0]+1][quad_num[1]+1][:]*param[0]*(1 - param[1]))
            self.ax1.plot(point_temp[0],point_temp[1],'ro')
        self.fig.canvas.draw()
        self.fig1.canvas.draw()