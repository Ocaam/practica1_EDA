# practica1
Realización práctica 1

Alumno:

Pablo Oceja Campanero 

Mail:

pablo.oceja@cunef.edu

URL del repositorio
https://github.com/Ocaam/practica1_EDA

A lo largo de este trabajo se ha realizado un primer análisis EDA de un dataset otorgado por los docentes en los cuales se buscaba realizar un posterior modelo para una variable objetivo o 'TARGET' la cual obtenía valores binarios (0 o 1) siendo 0 = no tuvo dificultades para realizar pagos, 1 = tuvo dificultades para realizar los pagos.

Consta de 4 notebooks en los cuales se realiza, explicado a continuación de forma muy generalizada:

Notebook 1: Un primer análisis y vista general del dataset, variable objetivo, variables continuas y categóricas

Notebook 2: Visualización de como se comportan las variables con el 'TARGET' y entre sí a través de correlaciones. También se realiza el tratamiento de los outlier, NaN y similares

Notebook 3: Primero se realiza la codificación de variables, dos variables categóricas tenían más de 10 valores diferentes por los que se realiza un OneHotEncoding para todas las variables salvo para estas dos. Posterior a esto se realiza el escalado estandar de variables para poder trabajar con ellas, también un par de regularizaciones, Lasso y Ridge, para la selección de variables, me quedo con Lasso, y la prueba de modelos e hiperparámetros. Finalmente me quedo con un modelo LightGBM que me devolvía las mejores métricas a priorí, estudio sus mejores hiperparámetros para maximizar el recall como objetivo y estudio sus resultados con métricas como la curva ROC-AUC, Lift Curve, relación precision-recall... 

Notebook 4: A través de la librería SHAP se realiza el estudio de la variabilidad y como las variables afectan al modelo. Se guarda el modelo y posteriormente se pondrá en producción para otra práctica de la asignatura