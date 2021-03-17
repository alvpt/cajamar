###########################################################
#
# DATATHON CAJAMAR UNIVERSITYHACK 2021
#
# RETO: ATMIRA STOCK PREDICTION
# 
# Este archivo contiene el codigo desarrollado para la competicion
# de analisis de datos organizada por el banco Cajamar en colaboración
# con la empresa PcComponentes
#
###########################################################
#
# Autores:
#     - Miguel Chacón Maldonado
#     - Álvaro Pérez Trasancos
#     - Miguel Chacón 
#     
#
# Representación: Universidad de Navarra - Sede de Postgrado
# Máster en Big Data Science
# Fecha: 17/03/2021
# Lugar: Madrid, España
#
###########################################################

###########################################################
# Impresion en la consola de el mensaje de bienvenida
###########################################################

print('''
###########################################################
DATATHON CAJAMAR UNIVERSITYHACK 2021
RETO: ATMIRA STOCK PREDICTION
Este archivo contiene el codigo desarrollado para la competicion
de analisis de datos organizada por el banco Cajamar en colaboración
con la empresa PcComponentes.
Autores:
    - Roberto Flores.
    - Miguel Chacón.
    - Álvaro Pérez Trasancos
Representación: Universidad de Navarra - Sede de Postgrado.
Máster en Big Data Science.
Fecha: 17/03/2021.
Lugar: Madrid, España.
##########################################################

Pasos para la ejecución del programa:

1. Importacion de los modulos a emplear.

2. Tratamiento del dataset de modelado.
    2.1. Lectura de los datos de modelado.
    2.2. Preprocesamiento de los campos.
    2.3. Imputacion del campo 'precio'.
    2.4. Imputacion del campo 'antiguedad'
    2.5. Variables dummy para el campo 'categoria_uno'.
    2.6. Variables dummy para el campo 'estado'.
    2.7. Variables dummy para el campo 'dia_atipico'.
    2.8. Variables dummy para el campo 'campaña'.
    2.9. Eliminacion de campos no numericos.
    
3. Tratamiento del dataset de estimacion.
    3.1. Lectura de los datos de modelado.
    3.2. Preprocesamiento de los campos.
    3.3. Imputacion del campo 'precio'.
    3.4. Imputacion del campo 'antiguedad'
    3.5. Variables dummy para el campo 'categoria_uno'.
    3.6. Variables dummy para el campo 'estado'.
    3.7. Variables dummy para el campo 'dia_atipico'.
    3.8. Variables dummy para el campo 'campaña'.
    3.9. Eliminacion de campos no numericos.
    
4. Algoritmo de prediccion de las unidades vendidas.
    4.1. Definición de los datos de entrenamiento y de test.
    4.2. Optimización de hiperparámetros.
    4.3. Entrenamiento del modelo.
    4.4. Predicción y recogida de resultados.
    
##########################################################
''')

###########################################################
# Importacion de los modulos a emplear
###########################################################

print("1. Importacion de los modulos a emplear.")

# Manejo de estructuras de datos ordenados
import pandas as pd 

# Operaciones con matrices más eficientes
import numpy as np

# Para la comprobacion de valores nulos
import math

# Informa sobre el avance de un bucle
from tqdm import tqdm

# Creacion del modelo de prediccion
import lightgbm as lgb

# Optimización de hiperparámetros del modelo de predicción
from bayes_opt import BayesianOptimization



##############################################################
# Tratamiento del dataset
##############################################################

def tratamiento_dataset(df):

    ###########################################################
    # Preprocesamiento de los campos
    ###########################################################

    print("\n  ", apartado + "2. Preprocesamiento de los campos.")

    def preprocesamiento_de_campos(df):

        # Conversion del tipo de dato del campo 'precio' de tipo string a float
        df['precio'] = df['precio'].astype('str', copy=True, errors='raise')
        df['precio'] = [x.replace(',', '.') for x in df['precio']]
        df['precio'] = df['precio'].astype('float64', copy=True, errors='raise')

        # Conversion del tipo de dato del campo 'fecha' de tipo string a time
        df['fecha'] = df['fecha'].dt.strftime('%Y/%m/%d')
        
        # Conversion del tipo de dato del campo 'antiguedad' de tipo string a int
        
        if 'unidades_vendidas' not in df.columns:
            df = df.replace({'antiguedad': {'-': 0}})
            df['antiguedad'] = df['antiguedad'].astype('int', copy=True, errors='raise')
            df['antiguedad'] = df['antiguedad'].replace({0: math.nan})

        # Creacion del campo fechaid como union de los campos 'fecha' e 'id
        df['fecha'] = df['fecha'].astype('str', copy=True, errors='raise')
        df['id'] = df['id'].astype('str', copy=True, errors='raise')
        df['fechaid'] = df['fecha'] + df['id']
        
        df['visitas'] = df['visitas'].astype('int')

        # Se eliminan las filas duplicadas con antigüedad, nos quedarán filas con antigüedad duplicada
        def preproc(df):

            #Eliminamos los duplicados del DF
            df = df.drop_duplicates()

            #Creamos un nuevo dataframe donde introducimos los registros con antiguedad nulo.
            df_nan_ant = df.dropna(subset = ["antiguedad"])

            #Comparamos los valores del dataset original con los del dataset de los valores antiguedad nulos
            diff_df = list(set(df['id'].unique())-set(df_nan_ant['id'].unique()))

            #Si diff_df contiene algún registo entonces procedemos a introducir dichos registros a df_extract para después añadirlo
            #a nuestro DataFrame
            if len(diff_df) != 0:
                df_extract = pd.DataFrame(columns= df.columns)

                for x in tqdm(diff_df):  
                    df_extract = df_extract.append(df[df['id'].isin([x])])

                df = df.dropna(subset=['antiguedad'])
                df = pd.concat([df,df_extract])
            
            # Se devuelve el dataframe sobre el que se han realizado las operaciones
            return df

        df = preproc(df)

        return df

    df = preprocesamiento_de_campos(df)

    ###########################################################
    # Imputacion del campo 'precio' 
    ###########################################################

    print("\n  ", apartado + "3. Imputacion del campo 'precio'.")

    def imputacion_precio(df):

        # Se realiza un ordenamiento de los valores en función de los campos
        # 'fechaid' y 'camapaña'
        df= df.sort_values(['fechaid', 'campaña']).drop_duplicates('fechaid', keep='last')

        # Conversion del tipo de dato del campo 'id' de tipo string a int
        df['id'] = df['id'].astype('int', copy=True, errors='raise')

        # Conversion del tipo de dato del campo 'fecha' de tipo string a time
        df_fechaid_campaña = df[["id","fecha","precio","fechaid"]].sort_values(["id", "fecha"])

        # Lista de identificadores sobre la que se iterara
        id_unique = df_fechaid_campaña["id"].unique()
        list_final = []

        # Este bucle complementa las filas cuyo precio sea nulo, con 
        # el precio de la fila anterior más próxima, del mismo id, que no lo sea
        for x in tqdm(id_unique):

            # Se crea un dataframe que contenga solo las filas con el id que se está tratando
            df_corte_id = df_fechaid_campaña[df_fechaid_campaña["id"]==x]

            # Lista con los precios asociados a ese id
            column_precio_id = df_corte_id["precio"].tolist()

            # Diccionario de ids
            dict_idx = {}

            # Lista de ids
            list_idx = []

            # Valor
            value = 0

            for idx, x in enumerate(column_precio_id):

                if value == 0:

                    if math.isnan(x):
                        list_idx.append(idx)

                    else:
                        value = x
                        for i in list_idx:
                            dict_idx[i] = x
                        list_idx=[]
                else:

                    if math.isnan(x):
                        dict_idx[idx] = value


                    else:
                        value = x
                        list_idx=[]

            list_precios = []

            for idx, x in enumerate(column_precio_id):
                if idx in dict_idx:
                    list_precios.append(dict_idx[idx])
                else:
                    list_precios.append(x)

            list_final = list_final + list_precios


        column_precio = df_fechaid_campaña["precio"].tolist()
        column_id = df_fechaid_campaña["id"].tolist()
        column_fecha = df_fechaid_campaña["fecha"].tolist()
        column_fechaid = df_fechaid_campaña["fechaid"].tolist()


        df_precios = pd.DataFrame({'id':column_id,
                                   'completado' : list_final,
                                   'sin completar' : column_precio,
                                    'fecha':column_fecha,
                                     'fechaid':column_fechaid})


        df = df.merge(df_precios[["fechaid","completado"]], how="left", on="fechaid")

        # Se elimina el campo 'precio', pues es el que contiene algunos valores nulos
        df = df.drop(['precio'], axis = 1)

        # Se cambia el nombre del campo 'completado' por 'precio'
        df = df.rename(columns={'completado': 'precio'})
        
        # Se devuelve el dataframe sobre el que se han realizado las operaciones
        return df
    
    # Se llama a la función encargada de imputar los valores nulos del campo 'precio'
    df = imputacion_precio(df)


    ###########################################################
    # Imputacion del campo 'antiguedad' 
    ###########################################################

    print("\n  ", apartado + "4. Creacion del campo 'antiguedad'.")

    def imputacion_antiguedad(df):
        
        # Se definen las categorías únicas que hay en el dataset
        categorias = df['categoria_uno'].unique()
        
        # Se definen las categorías únicas que hay en el dataset
        fechas = df['fecha'].unique()

        # Se itera sobre las categorias
        for letra in tqdm(categorias):

            # Se itera sobre las fechas
            for fecha in tqdm(fechas):
                
                # Se crea un dataframe temporal con la categoria y la fecha sobre las que se
                # está iterando
                df_temp = df.loc[df['categoria_uno'].isin([letra]) & df['fecha'].isin([fecha])]  
                
                # Se define la antiguedad media de ese dataframe temporal
                antiguedad_media = df_temp['antiguedad'].mean()
                
                # Se extrae el indice de ese dataframe
                df_temp_nan_index = df_temp[df_temp['antiguedad'].isin([np.nan])].index
                
                # Se rellenan los valores nulos con la media
                df.loc[df_temp_nan_index, 'antiguedad'] = antiguedad_media

        df['antiguedad'] = df['antiguedad'].fillna(0)
        
        # Se devuelve el dataframe sobre el que se han realizado las operaciones
        return df

    # Se llama a la función encargada de imputar los valores nulos del campo 'antiguedad'
    df = imputacion_antiguedad(df)
    


    ###########################################################
    # Variables dummy del campo 'categoria_uno'
    ###########################################################

    print("\n  ", apartado + "5. Creacion de las variables dummy para el campo 'categoria_uno'.")

    def dummies_uno(df):

        # Creacion de un dataframe con las variables dummies extraidas para cada valor 
        # del campo 'categoria_uno'
        dummies_categoria_uno = pd.get_dummies(df['categoria_uno'], prefix='categoria_uno')

        # Eliminacion del campo 'categoria_uno'
        df = df.drop(['categoria_uno'], axis = 1)

        # Union del dataframe con variables dummies al dataframe 'df'
        df = pd.merge(df, dummies_categoria_uno, left_index=True, right_index=True)
        
        # Se devuelve el dataframe sobre el que se han realizado las operaciones
        return df

    # Se llama a la función encargada crear las variables dummies para el campo 'categoria_uno'
    df = dummies_uno(df)
    
    
    ###########################################################
    # Variables dummy del campo 'estado'
    ###########################################################


    print("\n  ", apartado + "6. Creacion de las variables dummy para el campo 'estado'.")

    def dummies_estado(df):
                 
        df['estado'] = df['estado'].replace({'Transito': 'Rotura'})

        # Creacion de un dataframe con las variables dummies extraidas para cada valor 
        # del campo 'categoria_dos'
        dummies_estado = pd.get_dummies(df['estado'], prefix='estado',dummy_na=True)

        # Eliminacion del campo 'categoria_dos'
        df = df.drop(['estado'], axis = 1)

        # Union del dataframe con variables dummies al dataframe 'df'
        df = pd.merge(df, dummies_estado, left_index=True, right_index=True)
        
        # Se devuelve el dataframe sobre el que se han realizado las operaciones
        return df
    
    # Se llama a la función encargada crear las variables dummies para el campo 'estado'
    df = dummies_estado(df)
    
    
    ###########################################################
    # Variables dummy del campo 'dia_atipico'
    ###########################################################


    print("\n  ", apartado + "7. Creacion de las variables dummy para el campo 'dia_atipico'.")

    def dummies_dia(df):

        # Creacion de un dataframe con las variables dummies extraidas para cada valor 
        # del campo 'categoria_dos'
        dummies_dia = pd.get_dummies(df['dia_atipico'], prefix='dia_atipico',dummy_na=True)

        # Eliminacion del campo 'categoria_dos'
        df = df.drop(['dia_atipico'], axis = 1)

        # Union del dataframe con variables dummies al dataframe 'df'
        df = pd.merge(df, dummies_dia, left_index=True, right_index=True)
        
        # Se devuelve el dataframe sobre el que se han realizado las operaciones
        return df
    
    # Se llama a la función encargada crear las variables dummies para el campo 'dia_atipico'
    df = dummies_dia(df)
    
    ###########################################################
    # Variables dummy del campo 'campaña'
    ###########################################################


    print("\n  ", apartado + "8. Creacion de las variables dummy para el campo 'campaña'.")

    def dummies_campaña(df):

        # Creacion de un dataframe con las variables dummies extraidas para cada valor 
        # del campo 'categoria_dos'
        dummies_campaña = pd.get_dummies(df['campaña'], prefix='campaña',dummy_na=True)

        # Eliminacion del campo 'categoria_dos'
        df = df.drop(['campaña'], axis = 1)

        # Union del dataframe con variables dummies al dataframe 'df'
        df = pd.merge(df, dummies_campaña, left_index=True, right_index=True)
        
        # Se devuelve el dataframe sobre el que se han realizado las operaciones
        return df
    
    # Se llama a la función encargada crear las variables dummies para el campo 'campaña'
    df = dummies_campaña(df)
    

    ###########################################################
    # Últimas modificaciones
    ###########################################################

    # Se eliminan los campos que no serán de utilidad para el modelado
    df = df.drop(['fechaid', 'categoria_dos'], axis=1)
    
    # 
    df['visitas'] = df['visitas'].astype('int')
    
    df = df.sort_values(by=['fecha', 'id'])
        
    # Se devuelve el dataframe sobre el que se han realizado las operaciones
    return df



##############################################################
# Llamada a las funciones que inician el tratamiento de los datos
##############################################################

print('\n2. Tratamiento del dataset de modelado')

print('\n  2.1. Lectura del archivo de modelado')

apartado = "2."

# Creacion de un dataframe con los datos de modelado
df = pd.read_table("Modelar_UH2021.txt", delimiter = "|", encoding='utf8', parse_dates=["fecha"])

# Creación del dataframe de entrenamiento del modelo
df_train = tratamiento_dataset(df)

apartado = "3."

print('\n3. Tratamiento del dataset de estimacion')

print('\n  3.1. Lectura del archivo')

# Creacion de un dataframe con los datos de estimacion
df = pd.read_table("Estimar_UH2021.txt", delimiter = "|", encoding='utf8', parse_dates=["fecha"])

# Creacion de un dataframe para la estimacion
df_test = tratamiento_dataset(df)



##########################################################
# Definición de los datos de entrenamiento y de test
##########################################################
print("\n4. Algoritmo de prediccion de las unidades vendidas")

print("\n  4.1. Definición de los datos de entrenamiento y de test.")

# Se eliminan dos campos que no se usarán en el modelado
df_train = df_train.drop(['fecha', 'id'], axis=1)

# Se convierte el campo 'visitas' en entero
df_train['visitas'] = df_train['visitas'].astype(str).astype(int)
df_test['visitas'] = df_test['visitas'].astype(str).astype(int)

# Se convierte el campo 'unidades_vendidas' a entero
df_train['unidades_vendidas'] = pd.to_numeric(df_train['unidades_vendidas'])


# Se definen los datos de entrenamiento del modelo
X_train = df_train.loc[:, df_train.columns != 'unidades_vendidas']

# Se definen los datos que pretende predecir el modelo
y_train = df_train['unidades_vendidas']

# Se define un dataframe que guarde los valores de los campos 
# 'fecha' e 'id' del dataset de evaluacion. Estos campos se
# utilizarán para el dataset de respuesta
fecha_id_test = df_test[['fecha', 'id']]

# Se definen los datos de entrenamiento que evaluarán el modelo
X_test = df_test.drop(['fecha', 'id'], axis=1)



##########################################################
# Definición de los datos de entrenamiento y de test
##########################################################

print("\n4. Algoritmo de prediccion de las unidades vendidas")

print("\n  4.1. Definición de los datos de entrenamiento y de test.")

# Se eliminan dos campos que no se usarán en el modelado
df_train = df_train.drop(['fecha', 'id'], axis=1)

# Se convierte el campo 'visitas' en entero
df_train['visitas'] = df_train['visitas'].astype(str).astype(int)
df_test['visitas'] = df_test['visitas'].astype(str).astype(int)

# Se convierte el campo 'unidades_vendidas' a entero
df_train['unidades_vendidas'] = pd.to_numeric(df_train['unidades_vendidas'])


# Se definen los datos de entrenamiento del modelo
X_train = df_train.loc[:, df_train.columns != 'unidades_vendidas']

# Se definen los datos que pretende predecir el modelo
y_train = df_train['unidades_vendidas']

# Se define un dataframe que guarde los valores de los campos 
# 'fecha' e 'id' del dataset de evaluacion. Estos campos se
# utilizarán para el dataset de respuesta
fecha_id_test = df_test[['fecha', 'id']]

# Se definen los datos de entrenamiento que evaluarán el modelo
X_test = df_test.drop(['fecha', 'id'], axis=1)


##############################################################
# Optimización de hiperparámetros
##############################################################

print("\n  4.2. Optimización de hiperparámetros")

# Se defince una funcion
def bayes_parameter_opt_lgb(X_train, y_train, init_round, opt_round, n_folds, 
                            random_seed, n_estimators, learning_rate, 
                            output_process=False):
    
    # Se preparan los datos para su evaluacion
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Función de evaluación
    def lgb_eval(feature_fraction, bagging_fraction, max_depth, lambda_l1, 
                 lambda_l2, min_split_gain, min_child_weight):
        
        params = {'task': 'train',
                  'objective': 'regression',
                  'num_leaves': 4,
                  'num_iterations': n_estimators, 
                  'learning_rate':learning_rate,
                  'early_stopping_round': 100, 
                  'metric': ['l2', 'rmse'],
                  'verbose': -1,
                 }
        
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        params['n_splits'] = 5
        
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, 
                           stratified=True, verbose_eval = 200, metrics=['rmse'])
        

        return max(cv_result['rmse-mean'])
    
    # Se definen los rangos de los hiperparámetros a optimizar 
    lgbBO = BayesianOptimization(lgb_eval, {'feature_fraction': (0.1, 0.9),
                                            'bagging_fraction': (0.8, 1),
                                            'max_depth': (5, 8),
                                            'lambda_l1': (0, 5),
                                            'lambda_l2': (0, 3),
                                            'min_split_gain': (0.001, 0.1),
                                            'min_child_weight': (5, 50)}, random_state=0)
    
    # Se inicia la optimización de los hiperparámetros
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    

    
    # Se devuelven los mejores hiperparámetros
    return lgbBO.max


opt_params = bayes_parameter_opt_lgb(X_train, y_train, init_round=15, opt_round=15, n_folds=5, 
                                     random_seed=6, n_estimators=5, learning_rate=0.05)
        


##############################################################
# Entrenamiento del modelo
##############################################################

print("\n  4.3. Entrenamiento del modelo")

# Se extraen los parámetros de la funcion optimizadora
opt_params = opt_params['params']

# Se definen algunos hiperparametros
# Si el númeo de hojas es mayor, se produce un desbordamiento de la memoria
opt_params['num_leaves'] = 4
opt_params['n_estimators'] = 10000
opt_params['max_depth'] = int(opt_params['max_depth'])
opt_params['learning_rate'] = 0.005

# Se entrena el modelo
train_data = lgb.Dataset(X_train, label=y_train)
modelo = lgb.train(opt_params, train_data)



##############################################################
# Predicción y recogida de resultados
##############################################################

print("\n  4.4. Predicción y recogida de resultados")


y_pred = modelo.predict(X_test)

df_respuesta = fecha_id_test

df_respuesta['y_pred'] = y_pred.astype(int)

np.savetxt('DataBURRITOS_UH2021.txt', df_respuesta.values,  fmt = "%s", 
delimiter="|", header="Fecha|Id|Unidades vendidas")
