import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

TEST_DATASET = "test.csv"
TRAIN_DATASET = "train.csv"
pd.set_option('display.max_columns', None) 
class_labels = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']

# load data
df_train = pd.read_csv(TRAIN_DATASET)
df_test = pd.read_csv(TEST_DATASET)

# examine data
print(df_train.head())
print(df_train.info())

# analyze target variable
# print(df_train['NObeyesdad'].value_counts())
# plt.figure(figsize=(10, 6))
# sns.countplot(data=df_train, x='NObeyesdad')
# plt.xticks(rotation=45)
# plt.title('Analiza target varijable')
# plt.show()

# analyze gender distribution per weight class
# plt.figure(figsize=(12, 6))
# sns.countplot(x='NObeyesdad', hue='Gender', data=df_train, palette='coolwarm', order = df_train['NObeyesdad'].value_counts().index)
# plt.title('Gender Distribution in Each Weight Class', fontsize=16)
# plt.xlabel('Obesity Level', fontsize=14)
# plt.ylabel('Count', fontsize=14)
# plt.xticks(rotation=45)
# plt.show()

# analyzing and encoding CALC, CAEC and MTRANS (mode of transport)
# for df in [df_train, df_test]:
#     calc_distribution = df['CALC'].value_counts(normalize=True) * 100
#     print(calc_distribution)
#     calc_distribution.plot(kind='bar', title='Distribution of Alcohol Consumption')
#     plt.ylabel('Percentage')
#     plt.show()

#     mtrans_distribution = df['MTRANS'].value_counts(normalize=True) * 100
#     print(mtrans_distribution)
#     mtrans_distribution.plot(kind='bar', title='Distribution of Mode of Transportation')
#     plt.ylabel('Percentage')
#     plt.show()

#     caec_distribution = df['CAEC'].value_counts(normalize=True) * 100
#     print(caec_distribution)
#     caec_distribution.plot(kind='bar', title='Consumption of food between meals')
#     plt.ylabel('Percentage')
#     plt.show()


# # binary encoding
df_train['Gender'] = df_train['Gender'].map({'Male': 1, 'Female': 0})
df_train['family_history_with_overweight'] = df_train['family_history_with_overweight'].map({'yes': 1, 'no': 0})
df_train['FAVC'] = df_train['FAVC'].map({'yes': 1, 'no': 0})
df_train['SMOKE'] = df_train['SMOKE'].map({'yes': 1, 'no': 0})
df_train['SCC'] = df_train['SCC'].map({'yes': 1, 'no': 0})

df_train['CALC'] = df_train['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
df_train['CAEC'] = df_train['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
df_train = pd.get_dummies(df_train, columns=['MTRANS'])

# add BMI feature
df_train['BMI'] = df_train['Weight'] / (df_train['Height'] ** 2)

# # add  Diet Quality Score feature
# w_fcvc = 1  # Frequency of consumption of vegetables
# w_favc = -1  # Frequent consumption of high caloric food
# w_ch2o = 1  # Consumption of water daily 
# w_calc = -1  # Consumption of alcohol
# w_caec = 0.5  # Consumption of food between meals 
# w_faf = 1  # Physical activity frequency

# df_train['DQS'] = df_train['FCVC'] + \
#                   (1 - df_train['FAVC']) + \
#                   df_train['CH2O'] + \
#                   (3 - df_train['CALC']) + \
#                   (3 - df_train['CAEC']) + \
#                   df_train['FAF']

# # use age groups instead of age
# bins = [-np.inf, 17, 29, 59, np.inf]
# labels = [0, 1, 2, 3] # children, young adults, adults, seniors
# df_train['Age'] = pd.cut(df_train['Age'], bins=bins, labels=labels).astype(int)
# print(df_train['Age'].value_counts())
# print(df_train.head())

# correlation analysis
# target_mapping = {
#     'Insufficient_Weight': 0,
#     'Normal_Weight': 1,
#     'Overweight_Level_I': 2,
#     'Overweight_Level_II': 3,
#     'Obesity_Type_I': 4,
#     'Obesity_Type_II': 5,
#     'Obesity_Type_III': 6
# }
# df_train['NObeyesdad'] = df_train['NObeyesdad'].map(target_mapping)
# correlation_with_target = df_train.corr()['NObeyesdad'].sort_values()
# print(correlation_with_target)

# plt.figure(figsize=(10, 8))
# sns.barplot(x=correlation_with_target.values, y=correlation_with_target.index)
# plt.show()

# remove SMOKE, NCP and id feature due to low correlation
df_train = df_train.drop('SMOKE', axis=1)
df_train = df_train.drop('NCP', axis=1)
df_train = df_train.drop('id', axis=1)

# # model training
X = df_train.drop('NObeyesdad', axis=1)
y = df_train['NObeyesdad'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# # Decision Tree
# dt_classifier = DecisionTreeClassifier(random_state=1)
# dt_classifier.fit(X_train, y_train)

# y_pred = dt_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy for Decision Tree: {accuracy*100:.2f}%")

# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"F1 Score for Decision Tree: {f1:.2f}")

# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
# sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Labels')
# plt.xlabel('Predicted Labels')
# plt.show()

# Random Forest classifier
# rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_classifier.fit(X_train, y_train)

# y_pred = rf_classifier.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy for Random Forest: {accuracy*100:.2f}%")

# f1 = f1_score(y_test, y_pred, average='weighted')
# print(f"F1 Score for Random forest: {f1:.2f}")

# cm = confusion_matrix(y_test, y_pred)
# cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
# sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
# plt.title('Confusion Matrix')
# plt.ylabel('Actual Labels')
# plt.xlabel('Predicted Labels')
# plt.show()

# XGBoost
target_mapping_xg_boost = {
    'Insufficient_Weight': 0,
    'Normal_Weight': 1,
    'Overweight_Level_I': 2,
    'Overweight_Level_II': 3,
    'Obesity_Type_I': 4,
    'Obesity_Type_II': 5,
    'Obesity_Type_III': 6
}
y_train_encoded = y_train.map(target_mapping_xg_boost)
y_test_encoded = y_test.map(target_mapping_xg_boost)

xgb_classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss') #, n_estimators=500, max_depth=8, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8, gamma=1, reg_alpha=0.01, reg_lambda=1
xgb_classifier.fit(X_train, y_train_encoded)

y_pred = xgb_classifier.predict(X_test)
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Accuracy for XGBoost: {accuracy*100:.2f}%")

f1 = f1_score(y_test_encoded, y_pred, average='weighted')
print(f"F1 Score for XGBoost: {f1:.2f}")

cm = confusion_matrix(y_test_encoded, y_pred)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Neural Network
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# Normalization of attributes
# y_train_categorical = tf.keras.utils.to_categorical(y_train_encoded)
# y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded)

# model = Sequential()
# model.add(Dense(128, input_dim=X_train_scaled.shape[1], activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(y_train_categorical.shape[1], activation='softmax'))  # Use softmax for multi-class classification
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# history = model.fit(X_train_scaled, y_train_categorical, validation_split=0.2, epochs=20, batch_size=32)
# loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical)
# print(f"Accuracy for NN: {accuracy*100:.2f}%")

# y_pred = model.predict(X_test_scaled)
# y_pred_classes = y_pred.argmax(axis=-1)
# print(f"Final accuracy for NN: {accuracy_score(y_test_encoded, y_pred_classes):.2f}")

# CatBoost
catboost_clf = CatBoostClassifier(logging_level='Silent',iterations=1000, 
    learning_rate=0.1, 
    depth=6,eval_metric='Accuracy', 
    random_seed=1)
catboost_clf.fit(X_train, y_train)
y_pred = catboost_clf.predict(X_test)
y_pred = y_pred.ravel()

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy for CatBoost: {accuracy*100:.2f}%")

f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 Score for CatBoost: {f1:.2f}")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_labels, columns=class_labels)
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# # Rule based post processing based on gender
# gender = X_test['Gender'].values
# y_pred = np.where((gender == 1) & (y_pred == 'Obesity_Type_III'), 'Obesity_Type_II', y_pred)
# y_pred = np.where((gender == 0) & (y_pred == 'Obesity_Type_II'), 'Obesity_Type_III', y_pred)

# # Test preprocessing
# df_test['Gender'] = df_test['Gender'].map({'Male': 1, 'Female': 0})
# df_test['family_history_with_overweight'] = df_test['family_history_with_overweight'].map({'yes': 1, 'no': 0})
# df_test['FAVC'] = df_test['FAVC'].map({'yes': 1, 'no': 0})
# df_test['SMOKE'] = df_test['SMOKE'].map({'yes': 1, 'no': 0})
# df_test['SCC'] = df_test['SCC'].map({'yes': 1, 'no': 0})
# df_test['CALC'] = np.where(df_test['CALC'] == 'Always', 'Frequently', df_test['CALC'])
# df_test['CALC'] = df_test['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
# df_test['CAEC'] = df_test['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
# df_test = pd.get_dummies(df_test, columns=['MTRANS'])
# # df_test['DQS'] = df_test['FCVC'] + \
# #                   (1 - df_test['FAVC']) + \
# #                   df_test['CH2O'] + \
# #                   (3 - df_test['CALC']) + \
# #                   (3 - df_test['CAEC']) + \
# #                   df_test['FAF']
# df_test['BMI'] = df_test['Weight'] / (df_test['Height'] ** 2)

# df_test_ids = df_test['id'].copy()
# df_test = df_test.drop('id', axis=1)
# df_test = df_test.drop('SMOKE', axis=1)
# df_test = df_test.drop('NCP', axis=1)

# y_test_pred = xgb_classifier.predict(df_test)
# y_test_pred = y_test_pred.ravel()

# reverse_target_mapping = {v: k for k, v in target_mapping_xg_boost.items()}
# y_test_pred_names = [reverse_target_mapping[label] for label in y_test_pred]

# submission = pd.DataFrame({
#     'id': df_test_ids, 
#     'NObeyesdad': y_test_pred_names
# })
# submission.to_csv('submission.csv', index=False)