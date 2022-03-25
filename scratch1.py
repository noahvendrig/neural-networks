# import dependencies
import numpy as np
import tensorflow as tf
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score

LABELS = ["Normal", "Fraud"]

dataset = pd.read_csv(r"C:\Users\elect\Downloads\creditcard.csv")


# exploratory data analysis

# check for any  nullvalues
# print("Any nulls in the dataset ", dataset.isnull().values.any())
# print('-------')
# print("No. of unique labels ", len(dataset['Class'].unique()))
# print("Label values ", dataset.Class.unique())
# # 0 is for normal credit card transaction
# # 1 is for fraudulent credit card transaction
# print('-------')
# print("Break down of the Normal and Fraud Transactions")
# print(pd.value_counts(dataset['Class'], sort=True))


# visualise the dataset
count_classes = pd.value_counts(dataset['Class'], sort=True)
# count_classes.plot(kind='bar', rot=0)
# plt.xticks(range(len(dataset['Class'].unique())), dataset.Class.unique())
# plt.title("Frequency by observation number")
# plt.xlabel("Class")
# plt.ylabel("Number of Observations")
# plt.show()


# Save the normal and fraudulent transactions in separate dataframe
normal_dataset = dataset[dataset.Class == 0]
fraud_dataset = dataset[dataset.Class == 1]

# Visualise transaction amounts for normal and fraudluent transactions
bins = np.linspace(200, 2500, 100)
plt.hist(normal_dataset.Amount, bins=bins,
         alpha=1, density=True, label="Normal")
plt.hist(fraud_dataset.Amount, bins=bins, alpha=0.5,
         density=True, label="Fraudulent")
plt.legend(loc='upper right')
plt.title("Transaction amount vs Percentage of transactions")
plt.xlabel("Transaction amount (USD)")
plt.ylabel("Percentage of transactions")
plt.show()
