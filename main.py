import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def print_image(image, ax=None):
    """
    Paint image with ax of number
    """
    img = image.reshape((28, 28))
    img = img.astype('uint8')

    if ax is None:
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    else:
        ax.imshow(img, cmap='gray')
        ax.axis('off')

data = pd.read_csv('data/images.csv')

X = data.drop(columns='label')
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sample_features = X_train.iloc[2].values

X_y_train = X_train.copy(deep=True)
X_y_train['y'] = y_train

#fig, ax = plt.subplots()
#print_image(sample_features, ax=ax)
#plt.show()


selector = VarianceThreshold(threshold=0)
selector.fit(X_train)

X_train = X_train[selector.get_feature_names_out()]


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

#model by Tree
modelDTC = DecisionTreeClassifier(min_samples_leaf=5, criterion='gini', random_state=1)
modelDTC.fit(X_train, y_train)

print(modelDTC.score(X_val, y_val)) #acuracy

#model by forest

modelRF = RandomForestClassifier(n_estimators=100,min_samples_leaf=3, max_features=int(len(list(X_train.columns))**0.5), criterion='gini',random_state=1)
modelRF.fit(X_train, y_train)
print(modelRF.score(X_val, y_val))


#predictions1 = modelRF.predict(X_test)
predictions1 = modelRF.predict(X_val)

ps = precision_score(y_val, predictions1,average='macro')
rs = recall_score(y_val, predictions1,average='macro')
f1 = f1_score(y_val, predictions1,average='macro')

print(ps, rs, f1)

X_test = X_test[selector.get_feature_names_out()]

print("//////////////////////")
print(modelDTC.score(X_test, y_test))
print(modelRF.score(X_test, y_test))

