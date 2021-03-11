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
#     - Roberto Flores
#     - Miguel Chacón 
#     - Álvaro Pérez Trasancos
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
    2.4. Creación del campo 'variacion'
    2.5. Imputacion del campo 'antiguedad'
    2.6. Creación del campo 'score'.
    2.7. Variables dummy para el campo 'categoria_uno'.
    2.8. Variables dummy para el campo 'categoria_dos'.

3. Tratamiento del dataset de estimacion.
    3.1. Lectura de los datos de estimacion.
    3.2. Preprocesamiento de los campos.
    3.3. Imputacion del campo 'precio'.
    3.4. Creación del campo 'variacion'
    3.5. Imputacion del campo 'antiguedad'
    3.6. Creación del campo 'score'.
    3.7. Variables dummy para el campo 'categoria_uno'.
    3.8. Variables dummy para el campo 'categoria_dos'.

4. Algoritmo de prediccion de las unidades vendidas.
    4.1. Definicion del algoritmo.
    4.2. Carga de los datasets de entrenamiento y de test.
    4.3. Exposicion de los resultados


##########################################################

''')

###########################################################
# Importacion de los modulos a emplear
###########################################################

# Manejo de estructuras de datos ordenados
import pandas as pd 

import numpy as np

# Para la comprobacion de valores nulos
import math

# Informa sobre el avance de un bucle
from tqdm import tqdm 

###########################################################
# Lectura de los datos
###########################################################


print('Lectura del archivo de modelado')

# Creacion de un dataframe con los datos originales
df = pd.read_table("Modelar_UH2021.txt", delimiter = "|", encoding='utf8', parse_dates=["fecha"])

###########################################################
# Preprocesamiento de los campos
###########################################################

print('Preprocesamiento de los campos')

# Conversion del tipo de dato del campo 'precio' de tipo string a float
df['precio'] = df['precio'].astype('str', copy=True, errors='raise')
df['precio'] = [x.replace(',', '.') for x in df['precio']]
df['precio'] = df['precio'].astype('float64', copy=True, errors='raise')

# Conversion del tipo de dato del campo 'fecha' de tipo string a time
df['fecha'] = df['fecha'].dt.strftime('%Y/%m/%d')

# Creacion del campo fechaid como union de los campos 'fecha' e 'id
df['fecha'] = df['fecha'].astype('str', copy=True, errors='raise')
df['id'] = df['id'].astype('str', copy=True, errors='raise')
df['fechaid'] = df['fecha'] + df['id'] 

# Se eliminan las filas duplicadas
df = df.drop_duplicates()

###########################################################
# Imputacion del campo 'precio' 
###########################################################

print("Imputacion del campo 'precio'")

# Se realiza un ordenamiento de los valores en función de los campos
# 'fechaid' y 'camapaña'
df_fechaid_campaña = df.sort_values(['fechaid', 'campaña']).drop_duplicates('fechaid', keep='last')

# Conversion del tipo de dato del campo 'id' de tipo string a int
df_fechaid_campaña['id'] = df_fechaid_campaña['id'].astype('int', copy=True, errors='raise')

# Conversion del tipo de dato del campo 'fecha' de tipo string a time
df_fechaid_campaña = df_fechaid_campaña[["id","fecha","precio","fechaid"]].sort_values(["id", "fecha"])

# Lista de identificadores sobre la que se iterara
id_unique = df_fechaid_campaña["id"].unique()
list_final = []

print("Avance en la imputación del campo 'precio'")

# Este bucle complementa las filas cuyo precio sea nulo, con 
# el precio de la fila anterior más próxima, del mismo id, que no lo sea
for x in tqdm(id_unique):
    
    # Se crea un dataframe que contenga solo las filas con el id que se está tratando
#     df_corte_id = df_fechaid_campaña[df_fechaid_campaña["id"]==x]

    df_corte_id = df_fechaid_campaña.loc[df_fechaid_campaña["id"].isin([x])]
                         
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


###########################################################
# Creacion del campo 'variacion' 
###########################################################

df_variacion = df[["id","fecha","precio","fechaid"]].sort_values(["id", "fecha"])

id_unique = df_variacion["id"].unique()

list_final_var = []
list_final_porc = []

print("Avance en la creacion del campo 'variacion'")

for x in tqdm(id_unique):
    
    # Creacion de un dataframe que incluye solo las filas con un id determinado
    df_corte_id = df_variacion[df_variacion["id"].isin([x])]

    # Extraccion del campo 'precio' de ese dataframe
    column_precio_id = df_corte_id["precio"]

    # Calculo de la variacion entre filas
    variacion =  column_precio_id.diff()

    # A la primera fila se le asocia la variacion 0
    variacion.fillna(0)

    # Se halla el porcentaje de variacion
    porc = (variacion/column_precio_id)*100

    # Se convierte la serie 'variacion' en una lista
    list_variacion = variacion.tolist()

    # Se convierte la serie 'porc' en una lista
    list_porc = porc.tolist()

    list_final_var = list_final_var + list_variacion
    list_final_porc = list_final_porc + list_porc

# Crear df_variacion para posterior merge
column_fechaid = df_variacion["fechaid"].tolist()
column_precio = df_variacion["precio"].tolist()


df_variacion = pd.DataFrame({'variacion' : list_final_var,
                             '%_var' : list_final_porc,
                             'precio' : column_precio,
                             'fechaid': column_fechaid})    

df = df.merge(df_variacion[["fechaid","variacion","%_var"]], how="left", on="fechaid")



###########################################################
# Imputacion del campo 'antiguedad' 
###########################################################

categorias = df['categoria_uno'].unique()
fechas = df['fecha'].unique()
columna_fecha = df.columns.get_loc('antiguedad')

print("Avance en la imputacion del campo 'antiguedad'")

for letra in tqdm(categorias):
    
    for fecha in tqdm(fechas):
        
        df_temp = df.loc[df['categoria_uno'].isin([letra]) & df['fecha'].isin([fecha])]                                                          
        antiguedad_media = df_temp['antiguedad'].mean()
        df_temp_nan_index = df_temp[df_temp['antiguedad'].isin([np.nan])].index
        df.loc[df_temp_nan_index, columna_fecha] = antiguedad_media
                
###########################################################
# Creacion del campo 'score'
###########################################################

print("Creacion del campo 'score'")

# Creacion de un nuevo campo, 'mes'
df['mes'] = df['fecha'].str[5:7]

# Creacion de un dataframe auxiliar en el que se ordenan los meses de menos a mas
# importantes para cada uno de los valores del campo 'categoria_uno'
importancia_mes = pd.DataFrame(df.groupby(['categoria_uno', 'mes']).sum()['unidades_vendidas'])

# Ordenacion del dataframe en base al criterio antes expuesto
importancia_mes = importancia_mes.sort_values(by=['categoria_uno', 'unidades_vendidas'], ascending=[False, False])

# Creacion de una lista de listas a partir del indice del dataframe
importancia_mes_index = importancia_mes.index.tolist()

# Se añade un nuevo campo al dataframe. Este campo indicara la puntuacion de cada mes, para
# cada categoria. Su valor inicial es 0 en todas las filas
df['score'] = 0

# Inicializacion de la variable que indicara la puntuacion de cada mes
score = 0

# Se itera sobre la lista antes creada. Ademas, la funcion 'tqdm' permite visualizar
# en la linea de comandos el avance del proceso
for letra, mes in tqdm(importancia_mes_index):
    
    # A medida que se recorren las categorias ya ordenadas de menos a mas importantes, es necesario
    # aumentar la puntuacion
    score += 1
    
    # Posicion en el dataframe de las filas que tienen esta 'categoria_uno' y este 'mes'
    df_indices = df.loc[df['categoria_uno'].isin([letra]) & df['mes'].isin([mes])].index
    
    # Asignacion del valor de puntuacion
    df.loc[df_indices, 'score'] = score
    
    # Una vez se llega a la puntuacion maxima, se pasara a otra letra, por lo que es necesario 
    # reiniciar la puntuacion
    if score == 12:
        score = 0

# Eliminacion del campo 'categoria_uno'
df = df.drop(['mes'], axis = 1)
        
###########################################################
# Variables dummy del campo 'categoria_uno'
###########################################################

# Creacion de un dataframe con las variables dummies extraidas para cada valor 
# del campo 'categoria_uno'
dummies_categoria_uno = pd.get_dummies(df['categoria_uno'], prefix='categoria_uno')

# Eliminacion del campo 'categoria_uno'
df = df.drop(['categoria_uno'], axis = 1)

# Union del dataframe con variables dummies al dataframe 'df'
df = pd.merge(df, dummies_categoria_uno, left_index=True, right_index=True)

###########################################################
# Variables dummy del campo 'categoria_dos'
###########################################################

# Creacion de un dataframe con las variables dummies extraidas para cada valor 
# del campo 'categoria_dos'
dummies_categoria_dos = pd.get_dummies(df['categoria_dos'], prefix='categoria_dos',dummy_na=True)

# Eliminacion del campo 'categoria_dos'
df = df.drop(['categoria_dos'], axis = 1)

# Union del dataframe con variables dummies al dataframe 'df'
df = pd.merge(df, dummies_categoria_dos, left_index=True, right_index=True)


