from model_bqp import *
from interf import *
import matplotlib.image as mpimg
import numpy as np
N = 8
img=mpimg.imread('C:\\Users\\Alex\\Desktop\\shrilanka.png')
mesh = quad_mesh(N, True)
mesh.count_weights()
inter = interface(mesh, img, False)
