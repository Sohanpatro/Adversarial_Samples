from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from scipy import ndimage, misc
import cv2
import sklearn
# a = np.array([[1, 2.99], [3, 4]], dtype=np.float64)
# print(a)
# # a[2] = 3.92
# # print(a)
# a=a*2
# print(a)
# b= a.astype(np.int64)
# print(b)

# arr = np.arange(6).reshape(2, 3)
# np.place(arr, arr==2, 255)#[44, 55])
# print(arr)
# print(arr.dtype)

# a= np.array([[1, 256, 3, 43], [23, 34, 256, 256]], dtype=np.float32)
# print(a)
# np.place(a, a==256, 255)
# # c = a == 256
# print(a)
# print(a.dtype)
# a = a.astype(np.int64)
# print(a)
# print(a.dtype)

def details(fn):
    adv = np.load(fn)
    # adv /= 255.0
    print(type(adv), adv.shape, adv[0].shape, adv.dtype)
    print(adv.min(), adv.max())
    # adv= adv*256
    # print(adv.min(), adv.max())
    # adv = adv.astype(np.int64)
    # print(type(adv))
    # print(adv.min(), adv.max())
    # np.place(adv, adv == 256, 255)
    # print(type(adv))
    # adv = adv.astype(np.int8)
    # print(type(adv), adv.shape)
    # print(adv.min(), adv.max())
    # print(type(adv[0]), adv[0].shape)
    # print(adv[0].min(), adv[0].max())
    return adv

def display(x):
    print(x[img_ind], type(x[img_ind]))
    data = (x[img_ind].squeeze())#.resize(256, 256)
    # img = ndimage.zoom(data, 20)
    plt.imshow(data, cmap="gray", interpolation='bicubic')#, vmin=0, vmax= 255, interpolation='nearest')
    # plt.figure(figsize=(20,20))
    # plt.xticks([]), 
    plt.show()

def mnistResults():
    x_adv = details('x_adv_npa.dat')
    x_adv1 = details('x_adv_npa1.dat')
    x_adv2 = details('x_adv_npa2.dat')
    x_advm = details('x_adv_npa_mnist.dat')
    print ('***************')
    x = details('X_test2.dat')
    xm = details('X_test_mnist.dat')
    
    f, axarr = plt.subplots(6,6)
    for i in range(0, 6):
        axarr[i, 0].imshow(x[i].squeeze(), cmap="gray", interpolation='bicubic')
        axarr[i, 1].imshow(xm[i].squeeze(), cmap="gray", interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])
        # axarr[1, 0].imshow(x[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[2, 0].imshow(x[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[3, 0].imshow(x[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[4, 0].imshow(x[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[5, 0].imshow(x[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        axarr[i, 5].imshow(x_adv[i].squeeze(), cmap="gray", interpolation='bicubic')
        axarr[i, 3].imshow(x_adv1[i].squeeze(), cmap="gray", interpolation='bicubic')
        axarr[i, 4].imshow(x_adv2[i].squeeze(), cmap="gray", interpolation='bicubic')
        axarr[i, 2].imshow(x_advm[i].squeeze(), cmap="gray", interpolation='bicubic')
        # plt.xticks([]), plt.yticks([])
        # axarr[1, 1].imshow(x_adv[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[2, 1].imshow(x_adv[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[3, 1].imshow(x_adv[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[4, 1].imshow(x_adv[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
        # axarr[5, 1].imshow(x_adv[img_ind].squeeze(), cmap="gray", interpolation='bicubic')
    plt.show()

def cifar10Results():
    x = details('X_test_cifar10_255__Fri Apr  6 17-12-11 2018.dat')
    x_adv = details('x_adv_npa_cifar10_255__Fri Apr  6 17-12-11 2018.dat')
    # x_old = details('X_test_cifar10.dat')
    # x_adv_old = details('x_adv_npa_cifar10.dat')
    # blur = cv2.GaussianBlur(x_adv[4],(15,15),0)
    # print(type(blur), blur.shape)
    # print("STOP")
    # return
    # print("DONT")
    # x_1 = details('X_test_cifar10_1.dat')
    # x_adv_1 = details('x_adv_npa_cifar10_1.dat')
    # print("WGO")
    print ('***************')    
    numImages_toDisplay = 6
    f, axarr = plt.subplots(numImages_toDisplay,2)
    if 1 == numImages_toDisplay:
        axarr[0].imshow(x[4].squeeze(), interpolation='bicubic')
        axarr[1].imshow(x_adv[4].squeeze(), interpolation='bicubic')
        axarr[0].xaxis.set_major_locator(plt.NullLocator())
        axarr[0].yaxis.set_major_locator(plt.NullLocator())
        axarr[1].xaxis.set_major_locator(plt.NullLocator())
        axarr[1].yaxis.set_major_locator(plt.NullLocator())
        axarr[2].imshow(cv2.GaussianBlur(x_adv[4],(15,15),0).squeeze(), interpolation='bicubic')
        # axarr[3].imshow(x_adv_1[4].squeeze(), interpolation='bicubic')
        axarr[2].xaxis.set_major_locator(plt.NullLocator())
        axarr[2].yaxis.set_major_locator(plt.NullLocator())
        # axarr[3].xaxis.set_major_locator(plt.NullLocator())
        # axarr[3].yaxis.set_major_locator(plt.NullLocator())

    else:
        for i in range(0, numImages_toDisplay):
            axarr[i, 0].imshow(x[i].squeeze(), interpolation='bicubic')
            axarr[i, 1].imshow(x_adv[i].squeeze(), interpolation='bicubic')
            axarr[i, 0].set_xticks([])
            axarr[i, 0].set_yticks([])
            axarr[i, 1].set_xticks([])
            axarr[i, 1].set_yticks([])
            # axarr[i, 2].imshow(cv2.GaussianBlur(x_adv[i],(7,7),0).squeeze(), interpolation='bicubic')
            # axarr[i, 2].imshow(x_old[i].squeeze(), interpolation='bicubic')
            # axarr[i, 3].imshow(x_adv_old[i].squeeze(), interpolation='bicubic')
            # axarr[i, 2].set_xticks([])
            # axarr[i, 2].set_yticks([])
            # axarr[i, 3].set_xticks([])
            # axarr[i, 3].set_yticks([])
            # axarr[i, 0].xaxis.set_major_locator(plt.NullLocator())
            # axarr[i, 0].yaxis.set_major_locator(plt.NullLocator())
            # axarr[i, 1].xaxis.set_major_locator(plt.NullLocator())
            # axarr[i, 1].yaxis.set_major_locator(plt.NullLocator())
    plt.show()
def work():
    # x_adv1 = details('x_adv_npa1.dat')
    # print ('***************')
    # mnistResults()
    cifar10Results()
    # axarr[1,0].imshow(image_datas[2])
    # axarr[1,1].imshow(image_datas[3])

    # display(x)
    # # display(x_adv1)
    # display(x_adv)

# plt.imshow(adv[0], cmap="gray", vmin=0, vmax= 255)
# plt.show()

# print('***')
# # data = np.zeros( (512,512,3), dtype=np.uint8)
# # data[256,256] = [255,0,0]
# # print(data.min(), data.max(), data.shape)

# #     plt.imshow(data, interpolation='nearest')
# #     plt.show()


def play():
    img = cv2.imread('./cifar10_0res.png')

    blur = cv2.blur(img,(5,5))
    print(type(blur), blur.shape)
    return
    print(type("DONT"))
    dat = np.zeros( (28,28,1), dtype=np.float64)
    print(dat.shape[0])
    data = dat.squeeze()
    print(data.shape)

    # data[14,14] = 0.5#[255,0,0]
    # data[7, 7] = .2
    # data[21, 21] = .7
    mid = int(data.shape[0]/2)
    # print(mid)
    data[mid, mid] = 1
    # data.resize((14, 14))
    # img = Image.fromarray(data)
    # print(type(img))
    # img.thumbnail((64, 64), Image.ANTIALIAS)
    img = ndimage.zoom(data, 20)#, order=0)
    plt.imshow(img, cmap='gray')#, vmin=0, vmax=1)#, interpolation='nearest')
    # plt.figure(figsize=(28,28), dpi=2)
    plt.show()



img_ind=1223

# play()

work()

# import numpy as np
# import scipy.ndimage

# x = np.arange(9).reshape(3,3)

# print( 'Original array:')
# print (x)

# print ('Resampled by a factor of 2 with nearest interpolation:')
# a = scipy.ndimage.zoom(x, 2, order=0)
# print(type(a), a)


# print ('Resampled by a factor of 2 with bilinear interpolation:')
# b = scipy.ndimage.zoom(x, 2, order=1)
# print(type(b), b)

# print( 'Resampled by a factor of 2 with cubic interpolation:')
# c = scipy.ndimage.zoom(x, 2, order=3)
# print(type(c), c)