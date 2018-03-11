import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics

#数据准备
#数据读取及数据探索
data = pd.read_csv('/Users/xuyadan/Data_Analysis/projects/Employee_turnover/train.csv')
print(data.head(20))
#print(data.tail(20))
print(data.info())
print(data.columns)
print(data.describe())

pos_data = data[data['Attrition'] == 1]
neg_data = data[data['Attrition'] == 0]
print('总样本数量：',len(data))
print('离职员工有{}人'.format(len(pos_data)))
print('未离职员工有{}人'.format(len(neg_data)))

# plt.figure()
# sns.countplot(x='Attrition',hue='Education',data=data)
# plt.figure()
# sns.countplot(x='Attrition',hue='MaritalStatus',data=data)
# plt.figure()
# sns.countplot(x='Attrition',hue='YearsSinceLastPromotion',data=data)
# plt.show()
#数据展示
# sns.pairplot(data,hue='Attrition')
# plt.show()

#数据处理
#有两项数据跟所研究的问题无关，一是standard hours标准工时，二是EmployeeNumber工号。去除这两项数据进行余下数据的分析
#按照数据的类型分为三类和一个目标列，一是数值型数据，二是类别型数据，三是有序类别型数据，目标列即所要预测的那一列。
num_cols = ['Age', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear',
           'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

cat_cols = ['BusinessTravel', 'Department','EducationField','Gender','JobRole','MaritalStatus',
            'Over18', 'OverTime']

ord_cols = ['DistanceFromHome','Education','EnvironmentSatisfaction','JobInvolvement', 'JobLevel',
            'JobSatisfaction','PerformanceRating','RelationshipSatisfaction','StockOptionLevel','WorkLifeBalance']

tar_col = ['Attrition']



# #将数据进行转换
# ##将类别型的数据转换成one-hot编码（独热编码）格式
# #关于one-hot有两点，一是它将数据特征转换到欧式空间，二是根据one-hot计算所得的距离更加合理
# #原始类别型数据——>label encoding——>one-hot
Gender_encoder = preprocessing.LabelEncoder()
data['Gender_label'] = Gender_encoder.fit_transform(data['Gender'])


BusinessTravel_encoder = preprocessing.LabelEncoder()
data['BusinessTravel_label'] = BusinessTravel_encoder.fit_transform(data['BusinessTravel'])

Department_encoder = preprocessing.LabelEncoder()
data['Department_label'] = Department_encoder.fit_transform(data['Department'])

EducationField_encoder = preprocessing.LabelEncoder()
data['EducationField_label'] = EducationField_encoder.fit_transform(data['EducationField'])

JobRole_encoder = preprocessing.LabelEncoder()
data['JobRole_label'] = JobRole_encoder.fit_transform(data['JobRole'])

MaritalStatus_encoder = preprocessing.LabelEncoder()
data['MaritalStatus_label'] = MaritalStatus_encoder.fit_transform(data['MaritalStatus'])

Over18_encoder = preprocessing.LabelEncoder()
data['Over18_label'] = Over18_encoder.fit_transform(data['Over18'])

OverTime_encoder = preprocessing.LabelEncoder()
data['OverTime_label'] = OverTime_encoder.fit_transform(data['OverTime'])
#
# #再转换成one-hot
labeled = data[['Gender_label','BusinessTravel_label','Department_label','EducationField_label',
                'JobRole_label','MaritalStatus_label','Over18_label','OverTime_label']]

one_hot_encoder = preprocessing.OneHotEncoder()
cat_labeled = one_hot_encoder.fit_transform(labeled).toarray()


##将数值型数据进行归一化



#将数据进行划分，分为训练数据集和验证数据集
all_new_input_data = np.hstack((data[num_cols].values,data[ord_cols].values,cat_labeled))
output_data = data[tar_col].values
x_train,x_test,y_train,y_test = train_test_split(all_new_input_data,output_data,test_size=0.2,random_state=10)

#将数据输入到模型中进行学习
#随机森林，是一个集合模型
RF = RandomForestClassifier(random_state=20)
RF.fit(x_train,y_train)
#逻辑回归
LR = LinearRegression()
LR.fit(x_train,y_train)

#模型验证
RF_predict = RF.predict(x_test)
print('随机森林的准确率为：',metrics.accuracy_score(y_test,RF_predict))

LR_predict = LR.predict(x_test)
print('线性回归的准确率为：',metrics.accuracy_score(y_test,LR_predict))
