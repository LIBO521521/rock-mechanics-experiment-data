import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, Input, GlobalAveragePooling2D, Reshape, Dense, Multiply
from tensorflow.keras import Model
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
names = locals()

# ====================== SE 注意力机制 ======================
def se_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]
    se = GlobalAveragePooling2D()(input_feature)
    se = Reshape((1, 1, channel))(se)
    se = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(channel, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = Multiply()([input_feature, se])
    return x

# ====================== 带 SE 的 U-Net ======================
def unet(pretrained_weights=None, input_size=(1024, 1, 1)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = se_block(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)

    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = se_block(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)

    conv3 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = se_block(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)

    conv4 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = se_block(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)

    conv5 = Conv2D(1024, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = se_block(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = se_block(conv6)

    up7 = Conv2D(256, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = se_block(conv7)

    up8 = Conv2D(128, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = se_block(conv8)

    up9 = Conv2D(64, (2, 1), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 1))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = se_block(conv9)

    conv9 = Conv2D(2, (3, 1), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)
    return model

# ====================== 加载训练好的模型权重 ======================
model = unet(pretrained_weights="./checkpoint/MSCNN_SE.ckpt")


#checkpoint_save_path_new = "./checkpointengineering/MSCNN.ckpt"     # 新密  郑宏恒泰煤矿

#model.load_weights(checkpoint_save_path_new)

list3 = [[], [], [], [], [], [], [], []]  # 空表，存读出来的数

wave_train_path = './data2/'# 设置波形数据路径

for dirpath, dirnames, filenames in os.walk('./data2/'): #  遍历数据目录
    file_counts = len(filenames)  # 获取文件夹里面的文件数量

# 存储波形
for e in range(file_counts):
    with open(wave_train_path + str(1) + "." + str(e + 1) + ".txt",
              "r") as f: # 循环打开每个波形文件（格式：1.1.txt, 1.2.txt...）
        dataaa = f.read() # 读取文件全部内容
    dataaa = dataaa.split("\n") # 按换行符分割成列表
    del dataaa[len(dataaa) - 1] # 删除最后一个空元素
    for j in range(len(dataaa)):
        list3[e].append(dataaa[j]) # 将数据存入对应列表
for i in range(file_counts):
    names['wave' + str(i + 1)] = []  # 波形数据数组，动态创建wave1, wave2...等空列表
for j in range(file_counts):
    for i in range(1024):
        names['wave' + str(j + 1)].append(float(list3[j][i]))  # 波形数据数组赋值， 将字符串转为浮点数存入对应wave列表

for i in range(file_counts):
    names['y' + str(i + 1)] = []  # 波形数据数组，创建y1,y2...用于存储处理后的数据

for j in range(file_counts):
    total = 0  # 单通道的和重置
    ave = 0  # 某个通道平均值重置
    for i in range(len(names['wave' + str(j + 1)])):
        total = total + names['wave' + str(j + 1)][i]  # 求和
    ave = total / 1024  # 求平均，计算每个波形的平均值（中心点）

    for i in range(len(names['wave' + str(j + 1)])):
        names['wave' + str(j + 1)][i] = names['wave' + str(j + 1)][i] - ave  # 去平均值减去平均值（去除直流分量）
    min = 0  # 单通道的最小值
    max = 0  # 单通道的最大值
    for i in range(len(names['wave' + str(j + 1)])):
        if names['wave' + str(j + 1)][i] <= min:
            min = names['wave' + str(j + 1)][i]  # 求减去平均值后的最小值
        if names['wave' + str(j + 1)][i] >= max:
            max = names['wave' + str(j + 1)][i]  # 求减去平均值后的最大值
    names['y' + str(j + 1)] = np.zeros([1024, 1]) # 创建1024×1的零数组
    for i in range(1024):
        names['wave' + str(j + 1)][i] = (names['wave' + str(j + 1)][i]-min)/(max-min)  # 归一化到0-1之间
    for i in range(1024):
        names['y' + str(j + 1)][i] = names['wave' + str(j + 1)][i] # 存储归一化后的值

for i in range(file_counts):
    names['y_tf' + str(i + 1)] = names['y' + str(i + 1)][
        tf.newaxis, ..., tf.newaxis]  # 给数据增加维度：从[1024,1]→[1,1024,1,1]（符合模型输入要求）  把y1变为tf格式，赋值给y_tf

for i in range(file_counts):
    names['y_result' + str(i + 1)] = model.predict(names['y_tf' + str(i + 1)])  # y_tf输入   输出y_result，用模型预测每个波形，结果存储在y_result1,y_result2...

print('\n')

x_length = []  # 可视化x轴数值
for i in range(1024):
    x_length.append(i)  # 可视化x轴数值， 填充0-1023的x轴坐标

for i in range(file_counts):
    names['y_float' + str(i + 1)] = []  # tf类型转浮点型声明，创建y_float1,y_float2...用于存储浮点数结果
for i in range(1024):
    for j in range(file_counts):
        names['y_float' + str(j + 1)].append(names['y_result' + str(j + 1)][0][i][0])  # 赋值，将预测结果从张量转为普通浮点数列表
# 第六部分：结果可视化
fig = plt.figure() # 创建画布
ax = fig.subplots(file_counts, 2) # 创建file_counts行×2列的子图

for i in range(file_counts):
    ax[i, 0].set_title('波形' + str(i + 1))
    ax[i, 0].plot(x_length, names['wave' + str(i + 1)], "black") # 左侧子图绘制原始波形
for i in range(file_counts):
    ax[i, 1].set_title('特征图' + str(i + 1))
    ax[i, 1].plot(x_length, names['y_float' + str(i + 1)], "black") # 在右侧子图绘制模型输出的特征图
# 第七部分：事件点检测
for i in range(file_counts):
    names['wavemin' + str(i + 1)] = 0
    names['wavemax' + str(i + 1)] = 0 # 初始化每个波形的最大最小值（为画红线准备）

# 找到波形最大最小值 画 ---到时线--- 用
for i in range(1023):
    for j in range(file_counts):
        if names['wave' + str(j + 1)][i] > names['wavemax' + str(j + 1)]:
            names['wavemax' + str(j + 1)] = names['wave' + str(j + 1)][i]
        if names['wave' + str(j + 1)][i] < names['wavemin' + str(j + 1)]:
            names['wavemin' + str(j + 1)] = names['wave' + str(j + 1)][i] # 找到每个波形的真实最大最小值

# 峰值得索引
for j in range(file_counts):
    names['cba' + str(j + 1)] = 0
    names['waveabsmax' + str(j + 1)] = 1 # 初始化峰值检测变量
for i in range(1023):
    for j in range(file_counts):
        if names['cba' + str(j + 1)] <= abs(names['wave' + str(j + 1)][i]):
            names['waveabsmax' + str(j + 1)] = i + 1  # 峰值索引
            names['cba' + str(j + 1)] = abs(names['wave' + str(j + 1)][i])  # 峰值，找到每个波形的峰值位置（绝对值最大的点）

print(names['y_float' + str(1)]) # 打印第一个波形的特征图值（调试用）

# 判断事件
for i in range(1024):
    for j in range(file_counts):
        if names['y_float' + str(j + 1)][i] <= 0.50: names['y_float' + str(j + 1)][i] = 0
        if names['y_float' + str(j + 1)][i] > 0.50: names['y_float' + str(j + 1)][i] = 1
# 二值化处理：>0.5=1（事件点），≤0.5=0（非事件点）
# 每个0变1都记录下来，哪个离波峰近，选哪个
# 创建波形波动空表，用于存储波动点
for i in range(file_counts):
    names['wavewavewave' + str(i + 1)] = [] # 创建列表存储所有上升沿位置

for j in range(file_counts):
    for i in range(1023):
        if names['y_float' + str(j + 1)][i] == 0 and names['y_float' + str(j + 1)][i + 1] == 1:
            names['wavewavewave' + str(j + 1)].append(i + 1) # 检测0→1跳变点（上升沿位置）

for i in range(file_counts):
    print(names['wavewavewave' + str(i + 1)]) # 打印所有候选事件点
# 定义距离差，预测值和峰值abs之间的距离
for i in range(file_counts):
    names['julicha' + str(i + 1)] = 1024 # 初始化距离差为最大值1024
for i in range(file_counts):
    if len(names['wavewavewave' + str(i + 1)]) == 1:
        names['predicttime' + str(i + 1)] = names['wavewavewave' + str(i + 1)][0]
    # 如果预测值不止一个，判断预测值和峰值之间绝对值的距离，找到最小的距离，赋值给输出值
    if len(names['wavewavewave' + str(i + 1)]) > 1:
        for j in range(len(names['wavewavewave' + str(i + 1)])):
            if abs(names['wavewavewave' + str(i + 1)][j] - names['waveabsmax' + str(i + 1)]) <= names[
                'julicha' + str(i + 1)]:
                names['julicha' + str(i + 1)] = abs(
                    names['wavewavewave' + str(i + 1)][j] - names['waveabsmax' + str(i + 1)])
                names['predicttime' + str(i + 1)] = names['wavewavewave' + str(i + 1)][j] # 如果有多个候选点，选择最接近波峰的那个
    if len(names['wavewavewave' + str(i + 1)]) == 0:
        names['predicttime' + str(i + 1)] = 0 # 如果没有候选点，预测点为0
# 画预测线用
for i in range(file_counts):
    # 上升沿（保持原逻辑）
    ax[i, 1].vlines(ymax=1, ymin=0, x=names['predicttime' + str(i + 1)],
                    linestyles="solid", colors='red')
    ax[i, 1].text(0, 1.10, "预测开始:" + str(names['predicttime' + str(i + 1)]))

    # ===== 新增：只找第一个下降穿越 0.5，且距离上升沿 ≥ 20 =====
    pred_idx = names['predicttime' + str(i + 1)]
    fall_idx = 0
    if pred_idx > 0 and pred_idx < 1023:
        for k in range(pred_idx + 20, 1023):  # 从上升沿+20开始找下降沿
            v1 = names['y_result' + str(i + 1)][0][k][0]
            v2 = names['y_result' + str(i + 1)][0][k + 1][0]
            if v1 > 0.5 and v2 <= 0.5:  # 从上往下穿越0.5
                fall_idx = k + 1
                break

    # ===== 在特征图与波形图上同步画下降沿虚线 =====
    if fall_idx > 0:
        # 特征图：画红虚线
        ax[i, 1].vlines(ymax=1, ymin=0, x=fall_idx,
                        linestyles="solid", colors='red')
        ax[i, 1].text(500, 1.10, f"预测结束:{fall_idx}")

        # 波形图：同步画红虚线
        y_min = names['wavemin' + str(i + 1)]
        y_max = names['wavemax' + str(i + 1)]
        ax[i, 0].vlines(ymax=y_max, ymin=y_min, x=fall_idx,
                        linestyles="solid", colors='red')

# 保存结果 在特征图上画红色竖线标记预测点，并添加文字说明
f = open('./' + "time" + '.txt', 'w') # 创建结果文件time.txt
for i in range(file_counts):
    f.writelines(str(1) + '.' + str(i + 1) + '.txt' + " " + str(names['predicttime' + str(i + 1)]) + '\n') # 写入每个文件的预测点（格式：1.1.txt 预测点）
f.close() # 关闭文件
for i in range(file_counts):
    ax[i, 0].vlines(ymax=names['wavemax' + str(i + 1)], ymin=names['wavemin' + str(i + 1)],
                    x=names['predicttime' + str(i + 1)], linestyles="solid", colors='red') # 在原始波形图上画红色竖线标记预测点
plt.legend()#显示图例（虽然代码中未添加图例）
plt.subplots_adjust(hspace=0.5, wspace=0.25)
import pandas as pd

for j in range(file_counts):
    rows = []

    start_idx = names['predicttime' + str(j + 1)]

    # 找结束点（与你画图用的逻辑完全一致）
    end_idx = 0
    if start_idx > 0:
        for k in range(start_idx + 20, 1023):
            v1 = names['y_result' + str(j + 1)][0][k][0]
            v2 = names['y_result' + str(j + 1)][0][k + 1][0]
            if v1 > 0.5 and v2 <= 0.5:
                end_idx = k + 1
                break

    for x in range(1024):
        rows.append({
            "x": x,
            "feature_value": float(names['y_result' + str(j + 1)][0][x][0]),
            "binary_event": int(names['y_float' + str(j + 1)][x]),
            "is_start": 1 if x == start_idx else 0,
            "is_end": 1 if x == end_idx else 0
        })

    df = pd.DataFrame(rows)
    df.to_excel(f"feature_wave_{j+1}.xlsx", index=False)

print("8 个波形特征图已分别保存为 8 个 Excel 文件")
import pandas as pd

for j in range(file_counts):
    rows = []

    start_idx = names['predicttime' + str(j + 1)]

    # 找结束点（与你画图用的逻辑完全一致）
    end_idx = 0
    if start_idx > 0:
        for k in range(start_idx + 20, 1023):
            v1 = names['y_result' + str(j + 1)][0][k][0]
            v2 = names['y_result' + str(j + 1)][0][k + 1][0]
            if v1 > 0.5 and v2 <= 0.5:
                end_idx = k + 1
                break

    for x in range(1024):
        rows.append({
            "x": x,
            "feature_value": float(names['y_result' + str(j + 1)][0][x][0]),
            "binary_event": int(names['y_float' + str(j + 1)][x]),
            "is_start": 1 if x == start_idx else 0,
            "is_end": 1 if x == end_idx else 0
        })

    df = pd.DataFrame(rows)
    df.to_excel(f"feature_wave_{j+1}.xlsx", index=False)

print("8 个波形特征图已分别保存为 8 个 Excel 文件")

plt.show()
# 清理表格
list3 = [[], [], [], [], [], [], [], []]  # 清零, 清空数据列表（准备下次使用）
