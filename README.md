# 🚀 Proyecto Integrador: Procesamiento de Imágenes y Machine Learning

El proyecto se estructura en dos módulos principales:
1. **FotoApp (`fotoapp.py`):** Aplicación modular de procesamiento, filtrado y detección de bordes sobre imágenes digitales.
2. **Modelo Predictivo (`parte2_regresion.ipynb`):** Modelo de Regresión Lineal Múltiple para la estimación de costos médicos anuales (`insurance.csv`).

---

## 📂 Estructura del Repositorio

```text
├── imagenes/                  # Imágenes de prueba locales
├── fotoapp.py                 # Módulo principal con funciones de procesamiento de imagen
├── fotoapp_testdev.ipynb      # Notebook de ejecución y pruebas (Parte 1)
├── parte2_regresion.ipynb     # Notebook de análisis y Regresión Lineal Múltiple (Parte 2)
├── insurance.csv              # Dataset de seguros médicos (1.338 registros)
├── requirements.txt           # Dependencias del proyecto
└── README.md                  # Documentación oficial y conclusiones
 ---

📊 Diagrama de Flujo y Arquitectura
El siguiente diagrama detalla el flujo de opciones y procesos ejecutados en ambas partes del proyecto:

    A[Inicio: Proyecto Integrador FotoApp & ML] --> B[Parte 1: Módulo FotoApp fotoapp.py]
    A --> C[Parte 2: Regresión Lineal Múltiple]

    B --> B1[1. Carga local o URL y Redimensionamiento]
    B --> B2[2. Ecualización de Contraste e Histograma]
    B --> B3[3. Aplicación de 9 Filtros Pillow]
    B --> B4[4. Detección de Bordes Canny y Umbral]

    C --> C1[1. Carga de insurance.csv y EDA]
    C --> C2[2. Codificación get_dummies y Estructura Matricial X, y]
    C --> C3[3. Particionado Train 80% / Test 20%]
    C --> C4[4. Entrenamiento LinearRegression]
    C --> C5[5. Evaluación en Train y Test: R², MAE, RMSE]
    C --> C6[6. Diagnóstico de Residuos y Conclusiones]


⚙️ Instalación y Ejecución
Clonar el repositorio e instalar las dependencias:

https://github.com/danydeitu/Proyecto-FotoApp

Instalar en consola Bash:

pip install -r requirements.txt

Comandos:
