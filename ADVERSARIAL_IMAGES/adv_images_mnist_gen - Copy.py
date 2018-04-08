import numpy as np
# import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/', one_hot=True, reshape=False)
X_train = np.vstack((mnist.train.images, mnist.validation.images))
Y_train = np.vstack((mnist.train.labels, mnist.validation.labels))
X_test = mnist.test.images
Y_test = mnist.test.labels
print(type(X_train), X_train.shape)
print(type(Y_train), Y_train.shape)
print(type(X_test), X_test.shape)
print(type(Y_test), Y_test.shape)

# from PIL import Image
# from matplotlib import pyplot as plt
# import numpy as np
# from scipy import ndimage, misc


# # a = np.array([[1, 2.99], [3, 4]], dtype=np.float64)
# # print(a)
# # # a[2] = 3.92
# # # print(a)
# # a=a*2
# # print(a)
# # b= a.astype(np.int64)
# # print(b)

# # arr = np.arange(6).reshape(2, 3)
# # np.place(arr, arr==2, 255)#[44, 55])
# # print(arr)
# # print(arr.dtype)

# # a= np.array([[1, 256, 3, 43], [23, 34, 256, 256]], dtype=np.float32)
# # print(a)
# # np.place(a, a==256, 255)
# # # c = a == 256
# # print(a)
# # print(a.dtype)
# # a = a.astype(np.int64)
# # print(a)
# # print(a.dtype)

# def details(fn):
#     adv = np.load(fn)
#     print(type(adv), adv.shape, adv[0].shape, adv.dtype)
#     print(adv.min(), adv.max())
#     # adv= adv*256
#     # print(adv.min(), adv.max())
#     # adv = adv.astype(np.int64)
#     # print(type(adv))
#     # print(adv.min(), adv.max())
#     # np.place(adv, adv == 256, 255)
#     # print(type(adv))
#     # adv = adv.astype(np.int8)
#     # print(type(adv), adv.shape)
#     # print(adv.min(), adv.max())
#     # print(type(adv[0]), adv[0].shape)
#     # print(adv[0].min(), adv[0].max())
#     return adv

# def display(x):
#     print(x[img_ind], type(x[img_ind]))
#     data = (x[img_ind].squeeze())#.resize(256, 256)
#     # img = ndimage.zoom(data, 20)
#     plt.imshow(data, cmap="gray", interpolation='bicubic')#, vmin=0, vmax= 255, interpolation='nearest')
#     # plt.figure(figsize=(20,20))
#     # plt.xticks([]), 
#     plt.show()


# def work():
#     x_adv1 = details('x_adv_npa1.dat')
#     print ('***************')
#     x_adv = details('x_adv_npa.dat')
#     print ('***************')
#     x = details('X_test.dat')

#     display(x)
#     display(x_adv1)
#     display(x_adv)

# # plt.imshow(adv[0], cmap="gray", vmin=0, vmax= 255)
# # plt.show()

# # print('***')
# # # data = np.zeros( (512,512,3), dtype=np.uint8)
# # # data[256,256] = [255,0,0]
# # # print(data.min(), data.max(), data.shape)

# # #     plt.imshow(data, interpolation='nearest')
# # #     plt.show()


# def play():
#     dat = np.zeros( (28,28,1), dtype=np.float64)
#     print(dat.shape[0])
#     data = dat.squeeze()
#     print(data.shape)

#     # data[14,14] = 0.5#[255,0,0]
#     # data[7, 7] = .2
#     # data[21, 21] = .7
#     mid = int(data.shape[0]/2)
#     # print(mid)
#     data[mid, mid] = 1
#     # data.resize((14, 14))
#     # img = Image.fromarray(data)
#     # print(type(img))
#     # img.thumbnail((64, 64), Image.ANTIALIAS)
#     img = ndimage.zoom(data, 20)#, order=0)
#     plt.imshow(img, cmap='gray')#, vmin=0, vmax=1)#, interpolation='nearest')
#     # plt.figure(figsize=(28,28), dpi=2)
#     plt.show()



# img_ind=1223

# # play()

# work()

# # import numpy as np
# # import scipy.ndimage

# # x = np.arange(9).reshape(3,3)

# # print( 'Original array:')
# # print (x)

# # print ('Resampled by a factor of 2 with nearest interpolation:')
# # a = scipy.ndimage.zoom(x, 2, order=0)
# # print(type(a), a)


# # print ('Resampled by a factor of 2 with bilinear interpolation:')
# # b = scipy.ndimage.zoom(x, 2, order=1)
# # print(type(b), b)

# # print( 'Resampled by a factor of 2 with cubic interpolation:')
# # c = scipy.ndimage.zoom(x, 2, order=3)
# # print(type(c), c)