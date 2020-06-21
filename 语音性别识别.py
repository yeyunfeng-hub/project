import numpy as np
import csv
import matplotlib.pyplot as plt

def native_bayes(train_set,test_set,new_data_set,list_class):
    list_classes_1 = []
    train_data_1 = []
    list_classes_0 = []
    train_data_0 = []
    train_mat = []
    train_classes = []
    for index in train_set:
        train_mat.append(new_data_set[index])
        train_classes.append(list_class[index])
    test_mat = []
    test_classes = []
    for index in test_set:
        test_mat.append(new_data_set[index])
        test_classes.append(list_class[index])
    num_train_data = len(train_mat)
    num_feature = len(train_mat[0])
    p_1_class = sum(train_classes) / float(num_train_data)
    p_0_class = 1 - p_1_class
    n = N + 1
    for i in list(range(num_train_data)):
        if train_classes[i] == 1:
            list_classes_1.append(i)
            train_data_1.append(train_mat[i])
        else:
            list_classes_0.append(i)
            train_data_0.append(train_mat[i])
    train_data_1 = np.matrix(train_data_1)
    p_1_feature = {}
    for i in list(range(num_feature)):
        feature_values = np.array(train_data_1[:, i]).flatten()

        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))
        p_1_feature[i] = p

    train_data_0 = np.matrix(train_data_0)
    p_0_feature = {}
    for i in list(range(num_feature)):
        feature_values = np.array(train_data_0[:, i]).flatten()
        feature_values = feature_values.tolist() + list(range(n))
        p = {}
        count = len(feature_values)
        for value in set(feature_values):
            p[value] = np.log(feature_values.count(value) / float(count))
        p_0_feature[i] = p
    return p_1_feature, p_1_class, p_0_feature, p_0_class

def test_bayes(a):
    file_name = 'data/voice.csv'
    data = []
    male_data = []
    female_data = []
    list_class = []
    train_mat = []
    test_mat = []
    train_classes = []
    test_classes = []
    csv_reader = csv.DictReader(open(file_name, encoding='utf-8'))
    label_name = list(csv_reader.fieldnames)
    num = len(label_name) - 1
    male_count=0
    female_count=0
    for line in csv_reader.reader:
        data.append(line[:num])
        if line[-1][0] == 'm':
            gender = 1.0
            male_count+=1
            male_data.append(line[:num])
        else:
            gender = 0.0
            female_count+=1
            female_data.append(line[:num])
        list_class.append(gender)

    data = np.array(data).astype(float)
    male_data=np.array(male_data).astype(float)
    female_data=np.array(female_data).astype(float)
    min_vector = data.min(axis=0)
    max_vector = data.max(axis=0)
    diff_vector = max_vector - min_vector
    diff_vector /= 9
    male_sum_vector = np.sum(male_data, axis=0)
    male_mean_vector = male_sum_vector / male_count
    female_sum_vector = np.sum(female_data, axis=0)
    female_mean_vector = female_sum_vector / female_count
    for row in range(len(data)):
        for col in range(num):
            if data[row][col] == 0.0:
                if list_class[row]==0.0:
                   data[row][col] = male_mean_vector[col]
                else:
                    data[row][col] = female_mean_vector[col]
    new_data_set = []
    for i in range(len(data)):
        line = np.array((data[i] - min_vector) / diff_vector).astype(int)
        new_data_set.append(line)

    test_set = list(range(len(new_data_set)))
    train_set = []
    for i in range(a):
        random_index = int(np.random.uniform(0, len(test_set)))
        train_set.append(test_set[random_index])
        del test_set[random_index]
    for index in train_set:
        train_mat.append(new_data_set[index])
        train_classes.append(list_class[index])
    for index in test_set:
        test_mat.append(new_data_set[index])
        test_classes.append(list_class[index])
    p_1_feature, p_1_class, p_0_feature, p_0_class = native_bayes(train_set,test_set,new_data_set,list_class)
    male_accurate = 0.0
    male_wrong = 0.0
    female_accurate = 0.0
    female_error = 0.0
    male_num = 0.0
    female_num = 0.0
    for i in list(range(len(test_mat))):
        test_vector = test_mat[i]
        sum_1 = 0.0
        sum_0 = 0.0
        for j in list(range(len(test_vector))):
            sum_1 += p_1_feature[j][test_vector[j]]
            sum_0 += p_0_feature[j][test_vector[j]]
        p1 = sum_1 + np.log(p_1_class)
        p0 = sum_0 + np.log(p_0_class)
        if p1 > p0:
            result=1
        else:
            result=0

        if test_classes[i] == 1:
            male_num += 1
            if result == test_classes[i]:
                male_accurate += 1
            else:
                male_wrong += 1
        else:
            female_num += 1
            if result == test_classes[i]:
                female_accurate += 1
            else:
                female_error += 1
    male_accurate_rate.append(male_accurate / male_num)
    male_wrong_rate.append(male_wrong / male_num)
    female_accurate_rate.append(female_accurate / female_num)
    female_error_rate.append(female_error / female_num)
    total_accurate_rate.append((male_accurate + female_accurate) / (male_num + female_num))
    return (male_accurate + female_accurate) / (male_num + female_num)

N = 10
a = np.array([100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000])
male_accurate_rate = []
male_wrong_rate = []
female_accurate_rate = []
female_error_rate = []
total_accurate_rate = []
for k in a:
    test_bayes(k)
i=1
for i in range(20):
    print("训练集数量为%d" %(a[i]))
    print("男性正确率：", male_accurate_rate[i-1])
    print("男性错误率：", male_wrong_rate[i-1])
    print("女性正确率：", female_accurate_rate[i-1])
    print("女性错误率：", female_error_rate[i-1])
    print("总正确率：", total_accurate_rate[i-1])

plt.figure()
plt.plot(a,total_accurate_rate)
plt.title('total_accurate_rate')
plt.xlabel('train_num')
plt.ylabel('rate')
plt.tick_params(axis='both',labelsize=14)

plt.figure()
plt.plot(a,male_accurate_rate)
plt.title('male_accurate_rate')
plt.xlabel('train_num')
plt.ylabel('rate')
plt.tick_params(axis='both', labelsize=14)

plt.figure()
plt.plot(a,female_accurate_rate)
plt.title('female_accurate_rate')
plt.xlabel('train_num')
plt.ylabel('rate')
plt.tick_params(axis='both',   labelsize=14)


i = 0
N = 10
k=2000
male_accurate_rate = []
male_wrong_rate = []
female_accurate_rate = []
female_error_rate = []
total_accurate_rate = []

a=0.0
for i in range(50):
    a+=test_bayes(k)
a=a/50
print(a)

for i in range(50):
    print("第%d次：" % (i + 1))
    print("男性正确率：", male_accurate_rate[i])
    print("男性错误率：", male_wrong_rate[i])
    print("女性正确率：", female_accurate_rate[i])
    print("女性错误率：", female_error_rate[i])
    print("总正确率：", total_accurate_rate[i])

plt.figure()
plt.plot(total_accurate_rate)
plt.title('total_accurate_rate')
plt.xlabel('times')
plt.ylabel('rate')
plt.tick_params(axis='both',labelsize=14)

plt.figure()
plt.plot(male_accurate_rate)
plt.title('male_accurate_rate')
plt.xlabel('times')
plt.ylabel('rate')
plt.tick_params(axis='both', labelsize=14)

plt.figure()
plt.plot(female_accurate_rate)
plt.title('female_accurate_rate')
plt.xlabel('times')
plt.ylabel('rate')
plt.tick_params(axis='both',   labelsize=14)
plt.show()



