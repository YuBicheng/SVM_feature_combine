# SVM_feature_combine
图片的多个特征合并为一个特征输入svm训练和测试<br>
combine two or more feature to train svm and test

### 环境env <br>
python2.7<br>
opencv-python<br>
numpy<br>

### 使用说明 help:<br>
使用
```Bash
python svm.py
```
查看使用说明并使用.<br>
try 
```Bash
python svm.py
```
to get help

# 例子（example）<br>
数据集制作请打卡dataset文件夹查看[readme.md](dataset/readme.md)<br>
if you want to make a dataset for your own,please open folder ***dataset***，and check [readme.md](dataset/readme.md)<br>

### 训练 train<br>
默认特征为hog特征和lbp特征
如果想要更改特征组合可以更改代码233行函数列表，支持的特征提取函数都在最上方<br>
default features are HOG and LBP ,if you want to use other feature, you can change svm.py line 233 feature_fuc_list. suport function is on the top of code, you can also edit your own function.
```Bash
python svm.py ./dataset weights1
```

### 测试 test <br>
为了方便计算正确率 会将正样本图片的文件名中加入关键字，这样测试时如果有关键字在文件名中就会认为标签为正样本。
if your keyworr is 'pos' ,when the picture filename have this word,the algorithm will count it in postive samples,when the test finished will print acc.<br>
```Bash
python svm.py foldername keyword
```
