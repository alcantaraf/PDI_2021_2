import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import math

class buttons(object):
    bright = 0
    axis = 0
    Lf = [0]
    Li = [0]
    DeltaL = [0]
    n = 0
    def Brightness(self,event):
        if(self.bright >= 255):
            ax.set_title("Brilho Maximo")
        else:
            self.bright += 1
            if(((self.axis -1)>1023) or ((image_height-self.axis-1) < (self.axis -1))):
                ax.set_title("Minimo tamanho de Quadrado")
            else:       
                cv.rectangle(img,(self.axis-1,self.axis-1),(image_height-self.axis-1,image_width-self.axis-1), color=(self.bright,self.bright,self.bright), thickness = -1)      
                ax.clear()
                ax.imshow(img)  
                ax.set_axis_off() 
                plt.draw()    
                print("Bright:", self.bright)

    def Square(self,event):
        if(((self.axis -1)>1023) or ((image_height-self.axis-1) < (self.axis -1))):
                ax.set_title("Minimo tamanho de Quadrado")
        else:
            if(self.n == 0):
                self.n = 1 
                self.axis+= 8
            else:
                
                self.axis+= 8
                self.Lf.append(self.bright)
                print("Lf: ", self.Lf[self.n])
                if(self.n == 1):
                    self.DeltaL.append(self.Lf[self.n])
                else:    
                    self.DeltaL.append(self.Lf[self.n] - self.Li[self.n -1])
                print("Delta L: ", self.DeltaL[self.n])
                self.Li.append(self.Lf[self.n])
                print("Li: ", self.Lf[self.n])
                self.n +=1    
                print("Squares: ",self.axis/8)
            
    def Finish(self, event):
    
        # Subplots
        fig = plt.figure(2)
        figManager1 = plt.get_current_fig_manager()
        figManager1.window.showMaximized()
        ax1 = plt.subplot(121)
        ax2 = plt.subplot(322)
        ax3 = plt.subplot(324)
        ax4 = plt.subplot(326)

        # Plot Image, L and ΔIL
        ax1.imshow(img)
        ax1.set_axis_off()
        ax1.set_title('Figura Final')
        coefL = np.polyfit(np.arange(len(self.Lf)),self.Lf,3)
        predictL = np.polyval(coefL,np.arange(len(self.Lf)))
        ax2.plot(np.arange(len(self.Lf)),self.Lf,'--bo',label= r'$I_L$')
        ax2.plot(np.arange(len(self.Lf)),predictL,'-r',label='Regressão')
        ax2.legend()
        ax2.set_xlabel('Squares')
        ax2.set_ylabel(r'$I_L$')
        coefDL = np.polyfit(np.arange(len(self.DeltaL)),self.DeltaL,3)
        predictDL = np.polyval(coefDL,np.arange(len(self.DeltaL)))
        ax3.plot(np.arange(len(self.DeltaL)),self.DeltaL, '--bo',label =r'$\Delta$$I_L$')
        ax3.plot(np.arange(len(self.DeltaL)),predictDL,'-r',label='Regressão')
        ax3.legend()
        ax3.set_ylabel(r'$\Delta$$I_L$')
        ax3.set_xlabel('Squares')
        divres = [i/j for i, j in zip(self.DeltaL[1:],self.Lf[1:])]
        res_logdiv = [math.log(y) for y in divres]
        res_logI = [math.log(x) for x in self.Lf[1:]]
        print(res_logdiv)
        print(res_logI)
        ax4.plot(res_logI,res_logdiv, '-')
        ax4.set_ylabel(r'log($\frac{\Delta I_L}{I_L}$)')
        ax4.set_xlabel(r'$log(I_L)$')
        print("Finalizado")

        plt.tight_layout()
        fname = input("Nome da Figura:")
        fig.savefig(fname + '.png')
        plt.show()

  




image_height = 1024
image_width = 1024
button_callback = buttons()
img = np.full((image_height, image_width, 3),0,dtype = np.uint8)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
# plt.xticks([]), plt.yticks([]);
fig.suptitle("Projeto 01 - Adaptação ao Brilho e Discriminação")
ax.imshow(img)
ax.set_axis_off()
plt.draw()

btn_square = Button(plt.axes([0.7, 0.02, 0.2, 0.08]), 'New Square')
btn_square.on_clicked(button_callback.Square)
btn_bright = Button(plt.axes([0.1, 0.02, 0.2, 0.08]), 'Brightness')
btn_bright.on_clicked(button_callback.Brightness)
btn_finish = Button(plt.axes([0.4, 0.02, 0.2, 0.08]), 'Finish')
btn_finish.on_clicked(button_callback.Finish)


plt.show()

