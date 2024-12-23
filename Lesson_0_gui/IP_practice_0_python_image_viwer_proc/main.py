import cv2
import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfile
from processing.processing_factory import ProcessingFactory as pf
from processing.complex_filter import ComplexFilter as cf
from processing.filter import *
import numpy as np
from PIL import Image, ImageTk

img = None
def read_image_menu():
   global img, image, label_image, photo, root, image_label
   name = askopenfilename(initialdir="./",
                          filetypes=(("PNG", "*.png"), ("JPG", "*.JPG")),
                          title="Choose an image."
                          )
   img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
   update_image()

def save_image_menu():
   global img
   name = asksaveasfile(mode='w', filetypes=(("PNG", "*.png"), ("JPG", "*.JPG")),defaultextension=".png")
   cv2.imwrite(name.name, img)

def help_info_menu():
   pass

def empty_filter_menu():
   global img, image
   filt = cf()
   proc = pf("EmptyFilter", filt)
   img = proc.process(img)
   update_image()


def low_pass_filter_menu():
   global img
   filt = cf()
   filt.add_filter(LowPassFilter(10))
   proc = pf("LowPassFilter", filt)

   img = proc.process(img)
   update_image()

def high_pass_filter_menu():
   global img
   filt = cf()
   filt.add_filter(HighPassFilter(10))
   proc = pf("LowPassFilter", filt)

   img = proc.process(img)
   update_image()


def rotate90_filter_menu():
   global img
   filt = cf()
   filt.add_filter(RotationFilter(90))
   proc = pf("Rotate90", filt)

   img = proc.process(img)
   update_image()

def noisedegradation_filter_menu():
   global img
   filt = cf()
   filt.add_filter(DegradateFilter())
   proc = pf("NoiseDegradate", filt)

   img = proc.process(img)
   update_image()


def convolution_average_filter_menu():
   global img

   filt = cf()

   filt.add_filter(ConvolutionFilter(np.ones((3,3), np.float32) / 9.0))
   img_res = np.zeros((img.shape[0]*2, img.shape[1]*2), np.float32)
   img_res[:img.shape[0], :img.shape[1]] = img
   proc = pf("AverageLowPass", filt)

   img = proc.process(img_res)[: img.shape[0], : img.shape[1]] * 2

   update_image()

def filterset_highpass_degradate_filter_menu():
   global img
   filt = cf()
   filt.add_filter(HighPassFilter(10))
   filt.add_filter(DegradateFilter())
   proc = pf("Complex:All", filt)

   img = proc.process(img)
   update_image()

def filterset_lowpass_rotation135_filter_menu():
   global img
   filt = cf()
   filt.add_filter(LowPassFilter(10))
   filt.add_filter(RotationFilter(135))
   proc = pf("Complex: LowPass|Rotation135", filt)

   img = proc.process(img)

   update_image()


def add_image_to_gui():
   global root
   image_label = tkinter.Label(root)
   image_label.pack()
   return image_label


def update_image():
   global root, image_label, img
   image = Image.fromarray(img)
   tk_image = ImageTk.PhotoImage(image)
   image_label.configure(image=tk_image)
   image_label.image = tk_image
   root.after(100, update_image, root, image_label)

root = Tk()
root.geometry("750x450")
menubar = Menu(root)

image_label = add_image_to_gui()

# первый выпадающий список
filemenu = Menu(menubar, tearoff=0)
filemenu.add_command(label="Open", command=read_image_menu)
filemenu.add_command(label="Save", command=save_image_menu)
filemenu.add_separator()
filemenu.add_command(label="Exit", command=root.quit)
menubar.add_cascade(label="File", menu=filemenu)
###

# второй выпадающий список
editmenu = Menu(menubar, tearoff=0)
editmenu.add_command(label="EmptyFilter", command=empty_filter_menu)
editmenu.add_command(label="LowPassFilter", command=low_pass_filter_menu)
editmenu.add_command(label="HighPassFilter", command=high_pass_filter_menu)
editmenu.add_command(label="Rotate 90 grad", command=rotate90_filter_menu)
editmenu.add_command(label="NoiseDegradation", command=noisedegradation_filter_menu)
editmenu.add_command(label="Convolution: Average", command=convolution_average_filter_menu)
editmenu.add_command(label="Complex: HighPass|Degradation", command=filterset_highpass_degradate_filter_menu)
editmenu.add_command(label="Complex: LowPass|Rotation135", command=filterset_lowpass_rotation135_filter_menu)
menubar.add_cascade(label="Edit", menu=editmenu)
###

# третий выпадающий список
helpmenu = Menu(menubar, tearoff=0)
helpmenu.add_command(label="Info", command=help_info_menu)
menubar.add_cascade(label="Help", menu=helpmenu)
###

root.config(menu=menubar)
root.mainloop()