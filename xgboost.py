# Importar librerías usadas
import numpy as np
import pandas as pd

# Librerías para visualización
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
pd.options.display.max_rows = None
pd.options.display.max_columns = None

# Lectura de datos y forma de los mismos
data = pd.read_csv('Base_entrenamiento.csv', delimiter=',')
prueba = pd.read_csv('Base_prueba.csv', delimiter=',')
data.shape, prueba.shape

#Visualizamos algunos datos del dataframe "data"
data.head(10)

#Visualizamos algunos datos de la tabla "prueba"
prueba.head(10)

# Para empezar el data cleaning, observamos los missing values
data.isnull().sum()

#Como podemos observar la lista anterior, ninguna variable tiene missing value, para estar más seguro sumamos todos los valores.
data.isnull().sum().sum()

# Obteniendo valores únicos de cada variable, para observar rápidamente que variables pueden ser eliminadas
data.nunique()

#Lista de las variables con un único valor.
for x in data.columns : 
    if data[x].nunique()==1 :
        print(x)

#Eliminación de las variables con un único valor.
for x in data.columns : 
    if data[x].nunique()==1 :
        data = data.drop([x], axis=1)
#Forma de los datos que quedan     
data.shape

#Tipo de dato de cada variable
data.dtypes

#Información general del problema en un pastel
labels = 'auto cura', 'No auto cura'
sizes = [data.y_auto_cura[data['y_auto_cura']==1].count(), data.y_auto_cura[data['y_auto_cura']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Porción de clientes que auto curan", size = 20)
plt.show()

#Diagrama de barras para anhodia
sns.countplot(x='anhomes_ciclo', hue = 'y_auto_cura',data = data)

#X - y values
X = data.drop(['y_auto_cura'],axis=1).values
y = data.iloc[:, 106].values

#División de las variables X e y en entrenamiento y testeo. Esto lo hago con el fin de tener una medida de mi modelo
#pues el fin último es hacer una medición respecto a los y_test correspondientes a los datos de 'Base_prueba.csv'
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Cargamos el modelo que usaremos, a saber un XGBoost y lo ajustamos a nuestro datos
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

#Predicción para los datos de testeo
y_pred = classifier.predict(X_test)

#Matriz de confución
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

print((411+2503)/4000)

#Cross validation score CV
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean(), accuracies.std()



