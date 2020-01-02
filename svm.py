#coding=utf-8
import cv2
import numpy as np
import random
import os

#图片resize的目标大小
size = (36,36)
#svm保存文件和读取的文件名
svm_data_name = "svm_data.xml"
#svm 参数c
SVM_C = 1.5

def origin_feature(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    shape = img.shape
    result = img.reshape(shape[0]*shape[1]*shape[2],)
    return result

def hsv_feature(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    shape = hsv.shape
    result = hsv.reshape(shape[0] * shape[1] * shape[2])
    return result

def hog_feature(img):
    #hog参数默认设置
    # winSize = (128,128)
    # blockSize = (64,64)
    # blockStride = (8,8)
    # cellSize = (16,16)
    # nbins = 9
    # winStride = (8,8)
    # padding = (8,8)
    winSize = (32,32)
    blockSize = (16,16)
    blockStride = (2,2)
    cellSize = (4,4)
    nbins = 9
    winStride = (1,1)
    padding = (0,0)
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    #定义对象hog，同时输入定义的参数，剩下的默认即可
    result = hog.compute(img, winStride, padding).reshape((-1,))
    return result

def lbp_feature(img):
    """
    TODO:需要加上判断，如果为灰度图则不作处理
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    W, H = img.shape  # 获得图像长宽
    # 中心点八个方向的方向偏执，从左下角逆时针旋转
    #     #↖↑↗   0,0 1,0 2,0
    #     #←  →   0,1 1,1 2,1
    #     #↙↓↘   0,2 1,2 2,2
    #     #由于生成的lbp特征相对于原图少了一圈 所以lbp特征的中心点对应于原图的左上角点
    x_ = [-1, 0, 1, 1, 1, 0, -1, -1]
    y_ = [-1, -1, -1, 0, 1, 1, 1, 0]
    res = np.zeros((W - 2, H - 2),dtype="uint8")
    for i in range(1, W - 2):
        for j in range(1, H - 2):
            temp = ""
            for m in range(8):
                Xtemp = x_[m] + i
                Ytemp = y_[m] + j  # 分别获得对应坐标点
                if img[Xtemp, Ytemp] > img[i, j]:  # 像素比较
                    temp = temp + '1'
                else:
                    temp = temp + '0'
            # print int(temp, 2)
            res[i - 1][j - 1] = int(temp, 2)  # 写入结果中
    res = res.reshape(res.shape[0]*res.shape[1], )
    return res

"""
函数解释：
连接特征，输入为要调用的特征函数列表，以及要计算特征的图片
会将函数的计算结果连接起来，方便作为svm输入
"""
def feature_conbine(fuc_list, img):
    list_length = len(fuc_list)
    result = fuc_list[0](img)
    #print result.shape
    for i in range(1,list_length):
        temp = fuc_list[i](img)
        #print temp.shape
        result = np.hstack((result, temp))
        #print result.dtype
    return result

def hist_feature(img):
    hist = np.zeros((3,256), np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for c in range(3):
                hist[c][img[i][j][c]] += 1
    print hist.shape, hist.dtype
    print hist
    result = hist.reshape(hist.shape[0]*hist.shape[1],)
    return result

#必用函数：为了使得图像用不同特征提取算法提取出相同维度的特征
def resize(img):
    dst = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
    #print "大小调整成功"
    return dst

"""
opencv svm 输入 特征必须为float32
lable 必须为 int32
优先检查输入数据的类型。
"""
def svm_train(input_data, lable):
    svm = cv2.ml.SVM_create() #创建SVM model
    #属性设置
    svm.setType(cv2.ml.SVM_C_SVC)
    #svm 核函数设置 如果使用参数拼接方案线性核函数就足够了
    svm.setKernel(cv2.ml.SVM_LINEAR)
    #svm 参数 c 正则项权重 防止过拟合
    svm.setC(SVM_C)
    #训练
    result = svm.train(input_data,cv2.ml.ROW_SAMPLE,lable)
    svm.save(svm_data_name)


#svm预测函数
def svm_test(path, fuc_list):
    svm = cv2.ml.SVM_load(svm_data_name)
    for root, dirs, files in os.walk(path):
        for f in files:
            img_path = os.path.join(root, f)
            img = cv2.imread(img_path, 1)
            img = resize(img)
            input_x = feature_conbine(fuc_list, img)
            input_x = input_x.astype('float32')
            input_x = input_x.reshape((1, input_x.shape[0]))
            #print input_x.shape, input_x.dtype
            a,b = svm.predict(input_x)
            print f,':', b
    
    


#训练数据输入接口
#return:训练数据和标签，numpy数组
#训练数据为(n,features_num)float32
#标签数据为(n,1)int32
def data_reader(path, fuc_list):
    pos_path = os.path.join(path, '0')
    neg_path = os.path.join(path, '1')
    pos_count = 0
    neg_count = 0
    for root, dirs, files in os.walk(pos_path):
        for f in files:
            img_path = os.path.join(root, f)
            img = cv2.imread(img_path, 1)
            img = resize(img)
            if pos_count == 0:
                pos_feature = feature_conbine(fuc_list, img)
            else:
                temp = feature_conbine(fuc_list, img)
                pos_feature = np.vstack((pos_feature, temp))
            pos_count += 1
    print '正：', pos_count, pos_feature.shape
    pos_lable = np.ones((pos_count,1), np.int32)

    for root, dirs, files in os.walk(neg_path):
        for f in files:
            img_path = os.path.join(root, f)
            img = cv2.imread(img_path, 1)
            img = resize(img)
            if neg_count == 0:
                neg_feature = feature_conbine(fuc_list, img)
            else:
                temp = feature_conbine(fuc_list, img)
                neg_feature = np.vstack((neg_feature, temp))
            neg_count += 1
    print '负：', neg_count, neg_feature.shape
    neg_lable = np.zeros((neg_count,1), np.int32)
    #正负样本竖向拼接 制作训练集
    dataset = np.vstack((pos_feature, neg_feature))
    #opencv默认读图uint8需要转换为float32才能正常使用
    dataset = dataset.astype('float32')
    #正负样本标签拼接 制作lable集label
    lable = np.vstack((pos_lable, neg_lable))
    return dataset, lable



if __name__=="__main__":
    #svm测试数据
    # test = np.random.normal(size=(10,2))
    # test = np.float32(test)
    # print test.shape,test.dtype
    # lable = np.zeros((10,1))
    # for i in range(5):
    #     lable[i] = 1
    # lable = np.int32(lable)
    # print lable.shape,test.dtype

    #单图测试
    #img1 = cv2.imread("./data/dog.jpg", 1)
    #img2 = cv2.imread("./data/eagle.jpg", 1)
    
    #连接函数测试
    #test_list = [lbp_feature, origin_feature]
    #feature_conbine(test_list, img)
    #hog test
    #img1 = resize(img1)
    #img2 = resize(img2)
    #features1 = hog_feature(img1)
    #features1 = None
    #features2 = hog_feature(img2)
    #features = np.vstack((features1, features2))
    #print features1.shape, features2.shape, features.shape
    
    #lbp test
    #features = LBP(img)

    #print features.shape

    #训练集读取测试、样例
    feature_fuc_list = [hog_feature,lbp_feature]
    d, l = data_reader("F:\\螺栓故障svm用图", feature_fuc_list)
    print d.shape, d.dtype, l.shape, l.dtype
    svm_train(d, l)

    # 预测测试、样例
    test_path = "/home/junyingtianda/桌面/螺栓故障svm用图/test"
    svm_test(test_path, feature_fuc_list)

