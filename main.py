import pandas as pd

df = pd.read_csv('parkinsons.csv')
df.head(15)
x= df[['MDVP:Fo(Hz)','MDVP:Flo(Hz)']]
y= df['status']
x.head()
y.head()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x= scaler.fit_transform(x)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()
model.fit(x_train, y_train)
y_perd= model.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_perd)
