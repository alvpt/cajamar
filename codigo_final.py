# CAJAMAR UNIVERSITY HACK


# Este archivo contiene la entr

# Autores:
#     - Roberto Flores
#     - Miguel Chacón 
#     - Álvaro Pérez Trasancos

# Representación: Universidad de Navarra - Sede de Postgrado
# Máster en Big Data Science
# Fecha: 08/03/2021
# Lugar: Madrid, España


# En primer lugar, se importan los módulos a utilizar durante el resto 
# del programa
import pandas as pd # Manejo de estructuras de datos ordenados
import math 


# Lectura de los datos 
df = pd.read_table("Modelar_UH2021.txt", delimiter = "|", encoding='utf8', parse_dates=["fecha"])

# Conversion del campo precio a float
df['precio'] = df['precio'].astype('str', copy=True, errors='raise')
df['precio'] = [x.replace(',', '.') for x in df['precio']]
df['precio'] = df['precio'].astype('float64', copy=True, errors='raise')

# Conversion del campo fecha a tiempo
df['fecha'] = df['fecha'].dt.strftime('%Y/%m/%d')

# Creacion del campo fechaid
df['id'] = df['id'].astype('str', copy=True, errors='raise')
df['fecha'] = df['fecha'].astype('str', copy=True, errors='raise')
df['fechaid'] = df['fecha']+df['id'] 


# Definicion de la funcion preproc
def preproc(df):
    if 'value' in df.columns:
        df = df.drop(['value'], axis = 1)
    df2 = df['fechaid'].value_counts().to_frame()
    df2 = df2.reset_index()
    df2 = df2.rename(columns = {'index' : 'fechaid', 'fechaid': 'value'})
    df = df.merge(df2, how = 'left', on = 'fechaid')
    df = df.drop_duplicates()
    df = df.sort_values(['value', 'fechaid'], ascending = False)
    return df

# Preprocesamiento del dataframe
df = preproc(df)
df3 = df.dropna(subset = ["antiguedad"])
a = list(set(df['id'].unique())-set(df3['id'].unique()))

df_extract = df[df['id'].isin(a)]
df = df.dropna(subset=['antiguedad'])
df = pd.concat([df, df_extract])

df = preproc(df)

## Borrar valores duplicados de campaña

df.drop_duplicates(inplace=True)
# Muy importante hacer short_values por fechaid + campaña

df = df.sort_values(['fechaid', 'campaña']).drop_duplicates('fechaid', keep='last')

df['id'] = df['id'].astype('int', copy=True, errors='raise')

df = df[["id","fecha","precio","fechaid"]].sort_values(["id", "fecha"])

id_unique = df["id"].unique()

list_final = []

for x in id_unique:
    
    df_corte_id = df[df["id"]==x]
    column_precio_id = df_corte_id["precio"].tolist()
   
    dict_idx = {}
    list_idx = []
    value = 0
    
    for idx,x in enumerate(column_precio_id):

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

    

    
column_precio = df["precio"].tolist()
column_id = df["id"].tolist()
column_fecha = df["fecha"].tolist()
column_fechaid = df["fechaid"].tolist()


df_precios = pd.DataFrame({'id':column_id,
                           'completado' : list_final,
                           'sin completar' : column_precio,
                            'fecha':column_fecha,
                             'fechaid':column_fechaid})

df_entrega = df.copy()

df_entrega = df_entrega.merge(df_precios[["fechaid","completado"]], how="left", on="fechaid")

print(df_entrega.head())