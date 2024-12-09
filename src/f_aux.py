import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def dame_variables_categoricas(dataset=None):
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    lista_variables_categoricas = []
    other = []
    
    for i in dataset.columns:
        if (dataset[i].dtype != float) & (dataset[i].dtype != int):
            unicos = int(len(np.unique(dataset[i].dropna(axis=0, how='all'))))
            if unicos < 100:
                lista_variables_categoricas.append(i)
            else:
                other.append(i)
    
    # Contar el número de variables categóricas
    num_categoricas = len(lista_variables_categoricas)
    
    return lista_variables_categoricas, other, num_categoricas

def dame_variables_continuas(dataset=None):
    if dataset is None:
        print(u'\nFaltan argumentos por pasar a la función')
        return 1
    
    lista_variables_continuas = []
    other = []
    
    for i in dataset.columns:
        if dataset[i].dtype in [float, int]:
            unicos = dataset[i].nunique()
            if unicos >= 10:
                lista_variables_continuas.append(i)
            else:
                other.append(i)
    
    num_continuas = len(lista_variables_continuas)
    
    return lista_variables_continuas, other, num_continuas

def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[10,8]):
    # Para obtener la correlación de Spearman, sólo cambiar el metodo por 'spearman'

    if dataset is None:
        print(u'\nHace falta pasar argumentos a la función')
        return 1
    sns.set(style="white")
    # Compute the correlation matrix
    corr = dataset.corr(method=metodo) 
    # Set self-correlation to zero to avoid distraction
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=size_figure)
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, center=0,
                square=True, linewidths=.5,  cmap ='viridis' ) #cbar_kws={"shrink": .5}
    plt.show()
    
    return 0

def get_deviation_of_mean_perc(pd_loan, list_var_cont, target, multiplier):
    """
    Devuelve el porcentaje de valores que exceden del intervalo de confianza
    :type series:
    :param multiplier:
    :return:
    """
    pd_final = pd.DataFrame()
    
    for i in list_var_cont:
        
        series_mean = pd_loan[i].mean()
        series_std = pd_loan[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = pd_loan[i].size
        
        perc_goods = pd_loan[i][(pd_loan[i] >= left) & (pd_loan[i] <= right)].size / size_s
        perc_excess = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size / size_s
        
        if perc_excess > 0:    
            pd_concat_percent = pd.DataFrame(pd_loan[target][(pd_loan[i] < left) | (pd_loan[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0, 0], 
                                         pd_concat_percent.iloc[0, 1]]
            # Verificamos si 'index' existe antes de intentar eliminarla
            if 'index' in pd_concat_percent.index:
                pd_concat_percent = pd_concat_percent.drop('index', axis=0)
            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_outlier_values'] = pd_loan[i][(pd_loan[i] < left) | (pd_loan[i] > right)].size
            pd_concat_percent['porcentaje_sum_null_values'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

def get_percent_null_values_target(pd_loan, list_var_cont, target):

    pd_final = pd.DataFrame()
    for i in list_var_cont:
        if pd_loan[i].isnull().sum() > 0:
            pd_concat_percent = pd.DataFrame(pd_loan[target][pd_loan[i].isnull()]
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0, 0], 
                                         pd_concat_percent.iloc[0, 1]]
            
            # Verificamos si 'index' existe antes de intentar eliminarla
            if 'index' in pd_concat_percent.index:
                pd_concat_percent = pd_concat_percent.drop('index', axis=0)

            pd_concat_percent['variable'] = i
            pd_concat_percent['sum_null_values'] = pd_loan[i].isnull().sum()
            pd_concat_percent['porcentaje_sum_null_values'] = pd_loan[i].isnull().sum() / pd_loan.shape[0]
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

def cramers_v(confusion_matrix):
    """ 
    calculate Cramers V statistic for categorial-categorial association.
    uses correction from Bergsma and Wicher,
    Journal of the Korean Statistical Society 42 (2013): 323-328
    
    confusion_matrix: tabla creada con pd.crosstab()
    
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def plot_categorical_feature(df, col_name, target):
    """
    Visualize a categorical variable with and without faceting on the target variable.
    - df: DataFrame containing the data
    - col_name: Name of the categorical variable to plot
    - target: The target variable for faceting
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    count_null = df[col_name].isnull().sum()
    
    # Visualización del conteo de categorías
    sns.countplot(x=df[col_name], order=sorted(df[col_name].dropna().unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(f'{col_name} - Número de nulos: {count_null}')
    plt.xticks(rotation=90)

    # Visualización de la distribución de la variable objetivo según la variable categórica
    data = df.groupby(col_name)[target].value_counts(normalize=True).to_frame('proportion').reset_index()
    data.columns = [col_name, target, 'proportion']
    
    sns.barplot(x=col_name, y='proportion', hue=target, data=data, saturation=1, ax=ax2)
    ax2.set_ylabel(f'{target} fraction')
    ax2.set_title(f'{target} distribution by {col_name}')
    ax2.set_xlabel(col_name)
    plt.xticks(rotation=90)

    plt.tight_layout()
    
def plot_continuous_feature(df, col_name, target):
    """
    Visualize a continuous variable with and without faceting on the target variable.
    - df: DataFrame containing the data
    - col_name: Name of the continuous variable to plot
    - target: The target variable for faceting
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,3), dpi=90)
    
    count_null = df[col_name].isnull().sum()

    # Visualización del histograma
    sns.histplot(df.loc[df[col_name].notnull(), col_name], kde=False, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    ax1.set_title(f'{col_name} - Número de nulos: {count_null}')

    # Visualización del boxplot para la variable continua y la variable objetivo
    sns.boxplot(x=col_name, y=target, data=df, ax=ax2, hue=target, palette="Set2")  # Se agrega hue y palette
    ax2.set_ylabel('')
    ax2.set_title(f'{col_name} by {target}')
    ax2.set_xlabel(col_name)
    
    plt.tight_layout()

    """
    Calcula las correlaciones entre las variables continuas y la variable objetivo.
    
    :param df: DataFrame que contiene los datos.
    :param target: Nombre de la columna objetivo (por ejemplo, 'TARGET').
    :param list_var_cont: Lista de nombres de las variables continuas.
    
    :return: DataFrame con las correlaciones de las variables continuas con la variable objetivo.
    """
    # Crear un diccionario para almacenar las correlaciones
    correlaciones = {}
    
    # Recorremos cada variable continua en la lista
    for var in list_var_cont:
        # Calculamos la correlación de Pearson entre la variable y el objetivo
        correlacion = df[var].corr(df[target])
        # Almacenamos la correlación en el diccionario
        correlaciones[var] = correlacion
    
    # Convertimos el diccionario en un DataFrame para mejor visualización
    correlaciones_df = pd.DataFrame(list(correlaciones.items()), columns=['Variable', 'Correlación'])
    
    # Ordenamos el DataFrame de mayor a menor según la columna 'Correlación'
    correlaciones_df = correlaciones_df.sort_values(by='Correlación', ascending=False)
    
    return correlaciones_df