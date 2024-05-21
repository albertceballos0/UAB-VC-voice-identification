Sistema de Reconocimiento e Identificación de Voz

Introducción

Este proyecto tiene como objetivo diseñar e implementar un sistema de reconocimiento e identificación de voz para cuatro personas específicas. El sistema utilizará técnicas de procesamiento de señales de audio y aprendizaje automático para reconocer y distinguir las voces de las personas autorizadas.

El proceso de desarrollo constará de varias etapas clave:

Recopilación de Datos: Se recopilarán muestras de voz de las cuatro personas de interés, abarcando una variedad de palabras y frases relevantes para el contexto de uso previsto.
Preprocesamiento de Datos: Se extraerán características acústicas relevantes de las muestras de voz, como tono, intensidad y frecuencia.
Entrenamiento del Modelo: Se utilizarán algoritmos de aprendizaje automático para entrenar un modelo de reconocimiento de voz. Durante el entrenamiento, el modelo aprenderá a asociar las características acústicas de las muestras de voz con las identidades de las cuatro personas.
Evaluación y Ajuste Fino: Se probará el sistema utilizando conjuntos de datos adicionales que no se utilizaron durante el entrenamiento, con el fin de evaluar su precisión y rendimiento en la identificación de las cuatro personas. Se realizarán ajustes en el modelo según sea necesario para mejorar su capacidad de reconocimiento y reducir posibles errores.
Datos

Recopilaremos muestras de voz de cada uno de los usuarios a identificar mediante el modelo, 50 audios del tipo "hola, qué tal?" para cada uno de los usuarios a identificar, adicionalmente añadiremos 100 muestras de audio de usuarios diferentes a identificar. Por lo tanto, contaremos con un total de 300 audios.

id	id_mensaje	audio	id_persona
int (0 -> 299)	int (01 -> 50)	str (path)	int (0 -> 5)
Instalación

Para ejecutar este proyecto, necesitarás tener instalado Python y las siguientes bibliotecas:

pip install -r requeriments.txt


Para ejecutar el proyecto, simplemente abre y ejecuta el notebook main.ipynb en Jupyter Notebook o JupyterLab. El notebook contiene todas las celdas necesarias para preprocesar los datos, entrenar el modelo y evaluar su rendimiento.

