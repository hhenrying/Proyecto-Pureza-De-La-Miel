#Exportamos las librerias necesarias para el proyecto
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import scipy.stats as stats
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from yellowbrick.regressor import ResidualsPlot, PredictionError
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")


#Cargamos el dataset
dt_miel = pd.read_csv('/content/honey_purity_dataset.csv')
# Hacer una copia del DataSet original
dt = dt_miel.copy()


#Presentamos los datos
dt.head(10) 

#Presentamos la estructura de la data
dt.info() 

# Estadisticas Básicas que se pueden aplicar a las columnas numéricas
dt.describe().T

# Calculemos cuanto datos nulos y mostramos un porcentaje de esos datos nulos si se encuentran presentes
missing_count = dt.isnull().sum()
value_count = dt.isnull().count()
missing_percentage = round(missing_count / value_count * 100, 2)
missing_df = pd.DataFrame({"Cuantos Nulos": missing_count, "Porcentaje": missing_percentage})
missing_df


# Renombrar las columnas para mayor entendimiento
dt.rename(columns={ 'CS': 'PuntuacionColor',
                    'Density': 'Densidad',
                    'WC': 'ContenidoAgua',
                    'pH': 'pH',
                    'EC': 'ConductividadElectrica',
                    'F': 'Fructosa',
                    'G': 'Glucosa',
                    'Pollen_analysis': 'AnalisisPolen',
                    'Viscosity': 'Viscosidad',
                    'Purity': 'Pureza',
                    'Price': 'Precio'
                    }, inplace=True)

# Mostrar las nuevas columnas renombradas
dt.columns


#Histograma
numerical_variables=['PuntuacionColor','Densidad','ContenidoAgua','pH','Viscosidad','AnalisisPolen','Pureza','Precio']

plt.figure(figsize=(30, 20))
ind=1
for  variable in (numerical_variables):
    plt.subplot(4,3, ind)
    sns.histplot(x=dt[variable],kde=True)
    plt.title(variable)
    ind+=1
    plt.xlabel('')
plt.show()

#Label Encoding
label_encoder = LabelEncoder()
dt["AnalisisPolenEncoded"] = label_encoder.fit_transform(dt["AnalisisPolen"])
dt


#Observamos que tenemos una nueva columna llamada "AnalisisPolenEncoded" tipo numerico
dt.info()


# Dividir la Data, entre la función objetivo y el resto de los datos
X = dt[['PuntuacionColor','Densidad','ContenidoAgua','pH','Viscosidad','AnalisisPolenEncoded','Precio']]
y = dt['Pureza']

# Dividir el set de datos de pruebas y el set de datos de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Utilizo la libreria de SKLearn para hacer el llamado del algortimo en especifico que nos sirve para aplicar regresiones.
model_lin_reg_exp1 = LinearRegression()

# Mostrar como se ha definido la distribución de Datos
print("Tamaño de la muestra incluida en el set de entrenamiento: {}y Tamaño de la muestra de entrenamiento: {}\nX Tamaño de la muestra que se incluye en el set de testing: {}\ny Tamaño de la muestra usada en prueba: {}"
      .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
      
 
 #Feature variables
dt_new = dt[['PuntuacionColor','Densidad','ContenidoAgua','pH','Viscosidad','AnalisisPolenEncoded','Pureza','Precio']]

#Creating a new dataframe with required features
X1=dt_new[['PuntuacionColor','Densidad','ContenidoAgua','pH','Viscosidad','AnalisisPolenEncoded']]
#Target variable
y1=dt_new['Pureza']
print("Shape of feature variable :",X1.shape)
print("Shape of target variable :",y1.shape)


#Utilizamos el modelo LinearRegression como modelo predictivo
model_simple_lin_reg = LinearRegression()
model_simple_lin_reg.fit(X_train, y_train)
y_train_pred = model_simple_lin_reg.predict(X_train) # Datos conocidos
y_pred = model_simple_lin_reg.predict(X_test) # Datos no conocidos

#Creamos una función para validar el score del R2
def train_val(y_train, y_train_pred, y_test, y_pred, i):

    scores = {
    i+"_train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},

    i+"_test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}
    }

    return pd.DataFrame(scores)

slr_score = train_val(y_train, y_train_pred, y_test, y_pred, 'linear')
slr_score


#El R2 nos da un score del 73% hacemos un escalamiento de datos para estandarizar o normalizar los datos 
scaler = StandardScaler()
X1_train = scaler.fit_transform(X1_train)
X1_test= scaler.transform(X1_test)

#Para este nuevo entrenamiento utilizamos el modelo XGBoost 
xgb1 = XGBRegressor()
xgb1.fit(X1_train, y1_train)
#--------------------------------------------------------------
# Compute predictions on the training set
xgb1_pred_train = xgb1.predict(X1_train)
xgb1_mse_train = mean_squared_error(y1_train, xgb1_pred_train)
xgb1_r2_train = r2_score(y1_train, xgb1_pred_train)
xgb1_mae_train = mean_absolute_error(y1_train, xgb1_pred_train)
#---------------------------------------------------------------
xgb1_pred = xgb1.predict(X1_test)
xgb1_mse = mean_squared_error(y1_test, xgb1_pred)
xgb1_r2 = r2_score(y1_test, xgb1_pred)
xgb1_mae = mean_absolute_error(y1_test, xgb1_pred)

param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1]
            }
grid_search1 = GridSearchCV(estimator=xgb1, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search1.fit(X1_train, y1_train)
xbg1_pred = grid_search1.best_estimator_.predict(X1_test)

mae_1 = mean_absolute_error(y1_test, xgb1_pred)
mse_1 = mean_squared_error(y1_test, xgb1_pred)
r2_1 = r2_score(y1_test, xgb1_pred)
#------------------------------------------------------------------------
print("Best parameters found: ", grid_search1.best_params_)
print("MAE:", mae_1)
print("MSE:", mse_1)
print("R2 Score:", r2_1)
#Al imprimir el R2 con estalamiento, obtenemos un R2 del 97% de precisión


#Creamos una función para predecir
def prediccion_miel(modelo,PuntuacionColor,Densidad,ContenidoAgua,pH,Viscosidad,AnalisisPolenEncoded):
  datos_entrada = pd.DataFrame({
      'PuntuacionColor': [PuntuacionColor],
      'Densidad': [Densidad],
      'ContenidoAgua': [ContenidoAgua],
      'pH': [pH],
      'Viscosidad': [Viscosidad],
      'AnalisisPolenEncoded': [AnalisisPolenEncoded]
  })
  prediccion = modelo.predict(datos_entrada)
  return prediccion[0]
  
 pureza = prediccion_miel(
    modelo=xgb1,
    AnalisisPolenEncoded = 2,
    PuntuacionColor= 5.78,
    Densidad= 1.74,
    ContenidoAgua= 14.96,
    pH= 6.81,
    Viscosidad= 4417.74
)