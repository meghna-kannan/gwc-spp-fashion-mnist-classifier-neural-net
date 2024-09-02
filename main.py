import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report
from collections import defaultdict
from sklearn.metrics import confusion_matrix


# 8<------------ cut here ---------------
# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

#train_data = train_data.loc[(train_data['class']==3) | (train_data['class'] == 7) | (train_data['class']==5)]

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=20, random_state=2, stratify=y_train)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

#test_data = test_data.loc[(test_data['class']==3) | (test_data['class'] == 7) | (test_data['class']==5)]

# Separate the data (features) and the  classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)
#print("Loaded Data")

# 8<------------ cut here ---------------
neural_net_model = MLPClassifier( hidden_layer_sizes=(20),random_state=42,tol=0.005)
#print("creating model")

# 8<------------ cut here ---------------
neural_net_model.fit(X_train, y_train)
#print("fiting model")
# Determine model architecture
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Layer sizes: {layer_size_str}")


# 8<------------ cut here ---------------
# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1


# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test)*100
print(f"Overall Validation Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")

conf_matrix = confusion_matrix(y_test, y_pred)
class_ids = sorted(total_counts.keys())

# For better formatting
print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()  # Newline for next row

for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))

