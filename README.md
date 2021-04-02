This task is about imlemanting some basic computer vision oprations on images.
the functions are defined in 'ex1_utils.py' file, and include:
imReadAndConvert: read a gray scale or RGB image from the computer
imDisplay: display an image from the computer
transformRGB2YIQ: tranform an RBG image to YIQ representation
transformYIQ2RGB: tranform an image from YIQ representation to RGB
hsitogramEqualize: perform the histogram equalazetion algorithm in order to increase the image contrast
quantizeImage: decrece the number of colors in the image
the 'gamma.py' file define one function- gammaDisplay, that gives a slidebar to choose gamma for gamma currection (and dispaly the image after currection)
the 'ex1_main.py' file is a main program that demonstrate the use of the differate functions. the file was given and not implemeted by me.
the task has provided some images for testing - 'bac_con.pnj', 'beach.png', 'dark.jpg', 'water_bear.png'. I added two images - 'testImg1.jpg' and 'testImg2.jpg'.

*note*: all functions that get an image (not image path) as an argument, assume range [0,1].

the progam was tested on PyCharm IDE with python 3.9
