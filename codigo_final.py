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
# Creacion del campo 'variacion' 
###########################################################

def get_stock(df):

    #Creamos un diccionario donde almacenamos el id como clave y el número de veces que aparece como valor
    dict_id_len = df[['fecha', 'id']].groupby('id').count().to_dict()['fecha']

    #Creamos df1 donde filtramos por los siguientes campos
    df1 = df[['id', 'fecha', 'fechaid', 'estado']].sort_values(['id', 'fecha'])

    #Codificamos la variable estado para separar los casos donde hay stock(no rotura) de los que no hay (rotura, transito)
    df1['estado'] = [1 if x == 'No Rotura' else 0 for x in df1['estado']]

    #Pasamos los valores y las claves de dict_id_len para poder crear listas
    keys = list(dict_id_len.keys())
    values = list(dict_id_len.values())

    #En el siguiente bucle creamos una variable que será code que cambiará al comparar los estados o se mantendrá 
    #igual para cada id, para ello guardamos los valores en una columna
    col2 = []
    for x in range(len(keys)):

        df_2 = df1[df1['id'] == keys[x]].reset_index()
        cont = 0
        col2.append([df_2['fechaid'].iloc[0], cont])

        for y in range(1, int(values[x])):

            if df_2['estado'].iloc[y] != df_2['estado'].iloc[y-1]:
                cont = cont +1
                col2.append([df_2['fechaid'].iloc[y], cont])

            else:

                col2.append([df_2['fechaid'].iloc[y], cont])

    #Después transformamos col2 en un dataframe y lo unimos al dataframe original
    df_stock = pd.DataFrame(col2, columns = ['fechaid', 'code'])        
    df_stock['code'] = df_stock['code'].astype('int').astype('str')
    df = df.merge(df_stock, how = 'inner', on = 'fechaid')

    #Finalmente, creamos un nuevo campo en df donde unimos el campo id y el campo code que acabamos de crear
    df['idcode'] = df['id'] + df['code']

    return df

df['stock'] = get_stock(df)