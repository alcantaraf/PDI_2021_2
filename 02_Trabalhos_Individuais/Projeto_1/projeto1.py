import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class buttons(object):
    color = 0
    def Square(self,event):
        self.axis+= 8
        self.L.append(self.color)
        print("Squares: ",self.axis/8)







image_height = 1024
image_width = 1024
button_callback = buttons()
img = np.full((image_height, image_width, 3),0,dtype = np.uint8)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
# plt.xticks([]), plt.yticks([]);
fig.suptitle("Projeto 01 - Adaptação ao Brilho e Discriminação")
ax.imshow(img, cmap='gray', vmin = 0, vmax = 255)
plt.draw()

btn_plus = Button(plt.axes([0.7, 0.02, 0.2, 0.08]), 'New Square')
btn_plus.on_clicked(button_callback.Square)

plt.show()

