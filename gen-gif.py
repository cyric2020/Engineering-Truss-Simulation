# Generate a gif from all the images in the "images" folder

import os
import imageio

images = []
# for filename in os.listdir('images'):
#     images.append(imageio.imread('images/' + filename))
i = 0
while os.path.isfile('images/truss-{}.png'.format(i)):
    try:
        images.append(imageio.imread('images/truss-{}.png'.format(i)))
    except:
        break
    i += 1
imageio.mimsave('truss.gif', images, duration=0.05)