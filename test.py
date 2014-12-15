import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
img=mpimg.imread('C:\\Users\\Alex\\Desktop\\flag.png')
imgplot = plt.imshow(img[:,:,1])
plt.show()
print(img[3,3,0:2])