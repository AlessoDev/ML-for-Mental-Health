import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Predictor Work Interfere",
    page_icon="🧠",
    layout="wide"
)

# Variable global para el modelo
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_columns' not in st.session_state:
    st.session_state.model_columns = None

# Función para cargar el modelo pre-entrenado
@st.cache_resource
def load_trained_model():
    """Carga el modelo Random Forest pre-entrenado desde archivo joblib"""
    try:
        # Buscar archivo del modelo en diferentes ubicaciones posibles
        possible_paths = [
            'rf_optimizado_final.pkl',
            'rf_optimizado_final.joblib',
            'modelo_rf.pkl',
            'modelo_rf.joblib',
            'rf_model.pkl',
            'rf_model.joblib'
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path:
            model = joblib.load(model_path)
            st.success(f"✅ Modelo cargado exitosamente desde: {model_path}")
            return model
        else:
            return None
            
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

# Función para obtener las columnas del modelo entrenado
def get_model_columns_from_model(model):
    """Extrae las columnas del modelo entrenado"""
    if hasattr(model, 'feature_names_in_'):
        return list(model.feature_names_in_)
    else:
        # Columnas por defecto basadas en tu modelo original
        return [
            'Gender_Male', 'Gender_otros', 'family_history_Yes', 'treatment_Yes', 
            'remote_work_Yes', 'benefits_No', 'benefits_Yes', 'care_options_No', 
            'care_options_Not sure', 'care_options_Yes', 'wellness_program_No', 
            'wellness_program_Yes', 'seek_help_No', 'seek_help_Yes', 'anonymity_No', 
            'anonymity_Yes', 'mental_health_consequence_No', 'mental_health_consequence_Yes', 
            'coworkers_No', 'coworkers_Some of them', 'coworkers_Yes', 'supervisor_No', 
            'supervisor_Some of them', 'supervisor_Yes', 'mental_health_interview_No', 
            'mental_health_interview_Yes', 'phys_health_interview_No', 
            'phys_health_interview_Yes', 'mental_vs_physical_No', 'mental_vs_physical_Yes', 
            'obs_consequence_Yes'
        ]

# Función para preprocesar datos con las columnas correctas del modelo
def preprocess_data_for_model(df, model_columns, target_col='work_interfere'):
    """Preprocesa los datos aplicando one-hot encoding y alineando con columnas del modelo"""
    if target_col in df.columns:
        X = df.drop(target_col, axis=1)
        y = df[target_col]
    else:
        X = df.copy()
        y = None
    
    # Aplicar one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=False)
    
    # Alinear con las columnas del modelo
    # Agregar columnas faltantes con valor 0
    for col in model_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    
    # Mantener solo las columnas del modelo en el orden correcto
    X_encoded = X_encoded[model_columns]
    
    return X_encoded, y

# Función para hacer predicciones
def make_predictions(model, X_encoded):
    """Hace predicciones y retorna resultados con probabilidades"""
    predictions = model.predict(X_encoded)
    probabilities = model.predict_proba(X_encoded)
    
    # Obtener las clases del modelo
    classes = model.classes_
    
    results = []
    for i in range(len(predictions)):
        result = {'Prediccion': predictions[i]}
        
        # Agregar probabilidades para cada clase
        for j, class_name in enumerate(classes):
            result[class_name] = probabilities[i][j]
        
        results.append(result)
    
    return pd.DataFrame(results)

# Función para crear gráfico de radar
def create_radar_chart(categories, probabilities, title="Distribución de Probabilidades"):
    """Crea un gráfico de radar con las probabilidades de predicción"""
    
    # Crear el gráfico de radar
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=probabilities,
        theta=categories,
        fill='toself',
        name='Probabilidades',
        line=dict(color='rgb(32, 201, 151)', width=3),
        fillcolor='rgba(32, 201, 151, 0.25)',
        marker=dict(color='rgb(32, 201, 151)', size=8)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(probabilities) * 1.1],
                tickformat=".2f",
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=12)
            )
        ),
        showlegend=True,
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='white')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        width=500,
        height=500
    )
    
    return fig

# Función para crear gráficos de análisis
def create_analysis_plots(df):
    """Crea gráficos de análisis para variables categóricas"""
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    plots = []
    for col in categorical_cols:
        if col != 'work_interfere':
            value_counts = df[col].value_counts()
            
            plot_data = pd.DataFrame({
                'Categoria': value_counts.index,
                'Frecuencia': value_counts.values
            })
            
            fig = px.bar(
                plot_data,
                x='Categoria',
                y='Frecuencia',
                title=f'Distribución de {col}',
                labels={'Categoria': col, 'Frecuencia': 'Frecuencia'}
            )
            fig.update_layout(showlegend=False)
            plots.append((col, fig))
    
    return plots

# Función para obtener opciones de las variables
def get_variable_options():
    """Retorna las opciones disponibles para cada variable"""
    return {
        'Gender': ['Male', 'Female', 'otros'],
        'family_history': ['Yes', 'No'],
        'treatment': ['Yes', 'No'],
        'remote_work': ['Yes', 'No'],
        'benefits': ['Yes', 'No', "Don't know"],
        'care_options': ['Yes', 'No', 'Not sure'],
        'wellness_program': ['Yes', 'No', "Don't know"],
        'seek_help': ['Yes', 'No', "Don't know"],
        'anonymity': ['Yes', 'No', "Don't know"],
        'mental_health_consequence': ['Yes', 'No', 'Maybe'],
        'coworkers': ['Yes', 'No', 'Some of them'],
        'supervisor': ['Yes', 'No', 'Some of them'],
        'mental_health_interview': ['Yes', 'No', 'Maybe'],
        'phys_health_interview': ['Yes', 'No', 'Maybe'],
        'mental_vs_physical': ['Yes', 'No', "Don't know"],
        'obs_consequence': ['Yes', 'No']
    }

# Función para obtener preguntas y opciones en español
def get_questions_spanish():
    """Retorna las preguntas en español y opciones traducidas"""
    return {
        'Gender': {
            'question': '¿Cuál es tu género?',
            'options': {'Male': 'Masculino', 'Female': 'Femenino', 'otros': 'Otro'}
        },
        'family_history': {
            'question': '¿Tienes antecedentes familiares de enfermedades mentales?',
            'options': {'Yes': 'Sí', 'No': 'No'}
        },
        'treatment': {
            'question': '¿Has buscado tratamiento para una condición de salud mental?',
            'options': {'Yes': 'Sí', 'No': 'No'}
        },
        'remote_work': {
            'question': '¿Trabajas de forma remota (fuera de una oficina) al menos el 50% del tiempo?',
            'options': {'Yes': 'Sí', 'No': 'No'}
        },
        'benefits': {
            'question': '¿Tu empleador ofrece beneficios de salud mental?',
            'options': {'Yes': 'Sí', 'No': 'No', "Don't know": 'No lo sé'}
        },
        'care_options': {
            'question': '¿Conoces las opciones de atención de salud mental que ofrece tu empleador?',
            'options': {'Yes': 'Sí', 'No': 'No', 'Not sure': 'No estoy seguro/a'}
        },
        'wellness_program': {
            'question': '¿Tu empleador ha tratado el tema de la salud mental como parte de un programa de bienestar para empleados?',
            'options': {'Yes': 'Sí', 'No': 'No', "Don't know": 'No lo sé'}
        },
        'seek_help': {
            'question': '¿Tu empleador proporciona recursos para informarse sobre problemas de salud mental y cómo buscar ayuda?',
            'options': {'Yes': 'Sí', 'No': 'No', "Don't know": 'No lo sé'}
        },
        'anonymity': {
            'question': '¿Tu anonimato está protegido si decides usar recursos de tratamiento para salud mental o abuso de sustancias?',
            'options': {'Yes': 'Sí', 'No': 'No', "Don't know": 'No lo sé'}
        },
        'mental_health_consequence': {
            'question': '¿Crees que hablar de un problema de salud mental con tu empleador tendría consecuencias negativas?',
            'options': {'Yes': 'Sí', 'No': 'No', 'Maybe': 'Tal vez'}
        },
        'coworkers': {
            'question': '¿Estarías dispuesto/a a hablar de un problema de salud mental con tus compañeros de trabajo?',
            'options': {'Yes': 'Sí', 'No': 'No', 'Some of them': 'Con algunos de ellos'}
        },
        'supervisor': {
            'question': '¿Estarías dispuesto/a a hablar de un problema de salud mental con tu(s) supervisor(es) directo(s)?',
            'options': {'Yes': 'Sí', 'No': 'No', 'Some of them': 'Con algunos de ellos'}
        },
        'mental_health_interview': {
            'question': '¿Mencionarías un problema de salud mental durante una entrevista con un posible empleador?',
            'options': {'Yes': 'Sí', 'No': 'No', 'Maybe': 'Tal vez'}
        },
        'phys_health_interview': {
            'question': '¿Mencionarías un problema de salud física durante una entrevista con un posible empleador?',
            'options': {'Yes': 'Sí', 'No': 'No', 'Maybe': 'Tal vez'}
        },
        'mental_vs_physical': {
            'question': '¿Crees que tu empleador se toma la salud mental tan en serio como la salud física?',
            'options': {'Yes': 'Sí', 'No': 'No', "Don't know": 'No lo sé'}
        },
        'obs_consequence': {
            'question': '¿Has escuchado o visto consecuencias negativas para compañeros de trabajo con condiciones de salud mental en tu lugar de trabajo?',
            'options': {'Yes': 'Sí', 'No': 'No'}
        }
    }

# Título principal
st.title("🧠 Predictor de Work Interfere")
st.markdown("### Análisis y Predicción de Interferencia del Trabajo en la Salud Mental")

# Sidebar para seleccionar modo
st.sidebar.title("Opciones de Entrada")

# Inicializar modelo si no existe
if st.session_state.model is None:
    st.session_state.model = load_trained_model()

# Si no se pudo cargar el modelo desde archivo, permitir carga manual
if st.session_state.model is None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Cargar Modelo")
    st.sidebar.info("No se encontró modelo pre-entrenado. Sube tu archivo.")
    
    # Opción para subir archivo del modelo
    uploaded_model = st.sidebar.file_uploader(
        "Sube tu modelo entrenado (.pkl o .joblib)", 
        type=['pkl', 'joblib']
    )
    
    if uploaded_model is not None:
        try:
            st.session_state.model = joblib.load(uploaded_model)
            st.sidebar.success("✅ Modelo cargado exitosamente!")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar modelo: {str(e)}")

# Obtener columnas del modelo si existe
if st.session_state.model is not None and st.session_state.model_columns is None:
    st.session_state.model_columns = get_model_columns_from_model(st.session_state.model)

# Verificar si el modelo está disponible
model_available = st.session_state.model is not None and hasattr(st.session_state.model, 'classes_')

# Mostrar estado del modelo
if model_available:
    st.sidebar.success("✅ Modelo listo para predicciones")
else:
    st.sidebar.warning("⚠️ Modelo no disponible o no entrenado")

modo = st.sidebar.selectbox(
    "Selecciona el modo de entrada:",
    ["📊 Subir Dataset", "👤 Registro Individual", "📝 Encuesta Interactiva"]
)

# Modo 1: Subir Dataset (NO MODIFICAR - FUNCIONA CORRECTAMENTE)
if modo == "📊 Subir Dataset":
    st.header("📊 Análisis de Dataset")
    
    uploaded_file = st.file_uploader("Sube tu archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset cargado exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            
            # Mostrar muestra del dataset
            st.subheader("📋 Muestra del Dataset")
            st.dataframe(df.head())
            
            # Información del dataset
            st.subheader("ℹ️ Información del Dataset")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Filas", df.shape[0])
            with col2:
                st.metric("Columnas", df.shape[1])
            with col3:
                st.metric("Valores Nulos", df.isnull().sum().sum())
            
            # Análisis de variables categóricas
            st.subheader("📈 Análisis de Variables Categóricas")
            
            # Crear gráficos
            plots = create_analysis_plots(df)
            
            # Mostrar gráficos en grid
            cols = st.columns(2)
            for i, (col_name, fig) in enumerate(plots):
                with cols[i % 2]:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Preprocesar datos y hacer predicciones
            if st.button("🔮 Realizar Predicciones"):
                if not model_available:
                    st.error("❌ El modelo no está disponible. Por favor, carga un modelo entrenado.")
                else:
                    with st.spinner("Procesando datos y realizando predicciones..."):
                        # Separar datos con y sin work_interfere
                        if 'work_interfere' in df.columns:
                            df_with_target = df[df['work_interfere'] != 'IDK'].copy()
                            df_idk = df[df['work_interfere'] == 'IDK'].copy()
                            
                            # Entrenar modelo con datos conocidos
                            if not df_with_target.empty:
                                X_train, y_train = preprocess_data_for_model(df_with_target, st.session_state.model_columns)
                                st.session_state.model.fit(X_train, y_train)
                                
                                st.success("Modelo entrenado con datos del dataset")
                        
                        # Hacer predicciones en todo el dataset
                        X_all, _ = preprocess_data_for_model(df.drop('work_interfere', axis=1, errors='ignore'), st.session_state.model_columns)
                        results = make_predictions(st.session_state.model, X_all)
                        
                        # Mostrar resultados
                        st.subheader("🎯 Resultados de Predicciones")
                        
                        # Combinar dataset original con resultados
                        df_results = pd.concat([df.reset_index(drop=True), results], axis=1)
                        
                        # Mostrar tabla de resultados
                        st.dataframe(df_results)
                        
                        # Gráfico de distribución de predicciones
                        pred_counts = results['Prediccion'].value_counts()
                        pred_plot_data = pd.DataFrame({
                            'Work_Interfere': pred_counts.index,
                            'Cantidad': pred_counts.values
                        })
                        
                        fig_pred = px.bar(
                            pred_plot_data,
                            x='Work_Interfere',
                            y='Cantidad',
                            title='Distribución de Predicciones',
                            labels={'Work_Interfere': 'Work Interfere', 'Cantidad': 'Cantidad'}
                        )
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Opción para descargar resultados
                        csv = df_results.to_csv(index=False)
                        st.download_button(
                            label="⬇️ Descargar Resultados",
                            data=csv,
                            file_name='predicciones_work_interfere.csv',
                            mime='text/csv'
                        )
                        
        except Exception as e:
            st.error(f"Error al procesar el archivo: {str(e)}")
            st.info("Asegúrate de que el archivo CSV tenga las columnas correctas")

# Modo 2: Registro Individual (MODIFICADO CON GRÁFICO DE RADAR)
elif modo == "👤 Registro Individual":
    st.header("👤 Predicción para Registro Individual")
    
    if not model_available:
        st.error("❌ El modelo no está disponible. Por favor, carga un modelo entrenado primero.")
        st.stop()
    
    st.markdown("Ingresa los datos del registro en formato de diccionario:")
    
    # Ejemplo predeterminado
    ejemplo_default = """{
    'Gender': 'Male',
    'family_history': 'Yes',
    'treatment': 'Yes',
    'remote_work': 'No',
    'benefits': "Don't know",
    'care_options': 'Not sure',
    'wellness_program': 'No',
    'seek_help': "Don't know",
    'anonymity': "Don't know",
    'mental_health_consequence': 'Maybe',
    'coworkers': 'Some of them',
    'supervisor': 'No',
    'mental_health_interview': 'Maybe',
    'phys_health_interview': 'No',
    'mental_vs_physical': "Don't know",
    'obs_consequence': 'Yes'
}"""
    
    datos_input = st.text_area(
        "Datos del registro:",
        value=ejemplo_default,
        height=400
    )
    
    if st.button("🔮 Realizar Predicción"):
        try:
            # Convertir string a diccionario
            datos_dict = eval(datos_input)
            
            # Crear DataFrame
            df_individual = pd.DataFrame([datos_dict])
            
            # Mostrar datos ingresados
            st.subheader("📋 Datos Ingresados")
            st.dataframe(df_individual)
            
            # Preprocesar y predecir usando las columnas del modelo
            X_processed, _ = preprocess_data_for_model(df_individual, st.session_state.model_columns)
            
            # Hacer predicción
            prediction = st.session_state.model.predict(X_processed)[0]
            probabilities = st.session_state.model.predict_proba(X_processed)[0]
            classes = st.session_state.model.classes_
            
            # Mostrar resultados
            st.subheader("🎯 Resultados de la Predicción")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicción", prediction)
                
                # Tabla de probabilidades
                st.subheader("📊 Detalle de Probabilidades")
                prob_table = pd.DataFrame({
                    'Categoría': classes,
                    'Probabilidad': [f"{prob:.6f}" for prob in probabilities],
                    'Porcentaje': [f"{prob*100:.2f}%" for prob in probabilities]
                })
                st.dataframe(prob_table)
            
            with col2:
                # Crear gráfico de radar
                radar_fig = create_radar_chart(
                    categories=classes,
                    probabilities=probabilities,
                    title="Probabilidades de Work Interfere"
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error al procesar los datos: {str(e)}")
            st.info("Asegúrate de que el formato del diccionario sea correcto")

# Modo 3: Encuesta Interactiva (MODIFICADO CON GRÁFICO DE RADAR)
elif modo == "📝 Encuesta Interactiva":
    st.header("📝 Encuesta Interactiva")
    
    if not model_available:
        st.error("❌ El modelo no está disponible. Por favor, carga un modelo entrenado primero.")
        st.stop()
    
    st.markdown("Responde las siguientes preguntas para obtener tu predicción:")
    
    # Obtener preguntas y opciones en español
    questions_spanish = get_questions_spanish()
    
    # Crear formulario
    with st.form("encuesta_form"):
        st.subheader("📊 Información Personal y Laboral")
        
        col1, col2 = st.columns(2)
        
        responses = {}
        
        # Dividir preguntas en dos columnas
        questions = list(questions_spanish.keys())
        mid_point = len(questions) // 2
        
        with col1:
            for question in questions[:mid_point]:
                question_data = questions_spanish[question]
                spanish_options = list(question_data['options'].values())
                english_options = list(question_data['options'].keys())
                
                selected_spanish = st.selectbox(
                    f"**{question_data['question']}**",
                    spanish_options,
                    key=question
                )
                
                # Convertir respuesta en español de vuelta al inglés
                for eng, esp in question_data['options'].items():
                    if esp == selected_spanish:
                        responses[question] = eng
                        break
        
        with col2:
            for question in questions[mid_point:]:
                question_data = questions_spanish[question]
                spanish_options = list(question_data['options'].values())
                english_options = list(question_data['options'].keys())
                
                selected_spanish = st.selectbox(
                    f"**{question_data['question']}**",
                    spanish_options,
                    key=question
                )
                
                # Convertir respuesta en español de vuelta al inglés
                for eng, esp in question_data['options'].items():
                    if esp == selected_spanish:
                        responses[question] = eng
                        break
        
        # Botón para enviar
        submitted = st.form_submit_button("🔮 Obtener Predicción")
        
        if submitted:
            # Crear DataFrame con respuestas
            df_respuestas = pd.DataFrame([responses])
            
            # Mostrar respuestas
            st.subheader("📋 Tus Respuestas")
            st.dataframe(df_respuestas)
            
            # Preprocesar y predecir
            X_processed, _ = preprocess_data_for_model(df_respuestas, st.session_state.model_columns)
            
            # Hacer predicción
            prediction = st.session_state.model.predict(X_processed)[0]
            probabilities = st.session_state.model.predict_proba(X_processed)[0]
            classes = st.session_state.model.classes_
            
            # Mostrar resultados
            st.subheader("🎯 Tu Predicción")
            
            # Resultado principal
            st.success(f"**Predicción: {prediction}**")
            
            # Crear dos columnas para mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                # Gráfico de radar
                radar_fig = create_radar_chart(
                    categories=classes,
                    probabilities=probabilities,
                    title="Distribución de Probabilidades"
                )
                st.plotly_chart(radar_fig, use_container_width=True)
            
            with col2:
                # Tabla detallada
                st.subheader("📊 Probabilidades Detalladas")
                prob_table = pd.DataFrame({
                    'Categoría': classes,
                    'Probabilidad': [f"{prob:.6f}" for prob in probabilities],
                    'Porcentaje': [f"{prob*100:.2f}%" for prob in probabilities]
                })
                st.dataframe(prob_table)
                
                # Interpretación
                st.subheader("📝 Interpretación")
                max_prob_idx = np.argmax(probabilities)
                max_category = classes[max_prob_idx]
                confidence = probabilities[max_prob_idx] * 100
                
                st.info(f"La predicción indica que **{max_category}** es la categoría más probable con un {confidence:.1f}% de confianza.")

# Footer
st.markdown("---")
st.markdown("💡 **Nota:** Este modelo está basado en Random Forest optimizado para predecir la interferencia del trabajo en la salud mental.")

# Instrucciones para guardar modelo
with st.expander("ℹ️ Instrucciones para usar tu modelo pre-entrenado"):
    st.markdown("""
    ### 📁 Cómo usar tu modelo entrenado:
    
    **Opción 1 - Guardar modelo desde tu notebook:**
    ```python
    import joblib
    
    # Después de entrenar tu modelo rf_optimizado_final
    joblib.dump(rf_optimizado_final, 'rf_optimizado_final.pkl')
    ```
    
    **Opción 2 - Ubicación del archivo:**
    - Coloca el archivo `.pkl` o `.joblib` en la misma carpeta que este script
    - Nombres aceptados: `rf_optimizado_final.pkl`, `rf_optimizado_final.joblib`, `modelo_rf.pkl`, etc.
    
    **Opción 3 - Subir modelo:**
    - Usa la opción "Cargar Modelo" en la barra lateral
    - Sube directamente tu archivo `.pkl` o `.joblib`
    
    ### 🔧 Código para guardar tu modelo:
    ```python
    # En tu Google Colab o Jupyter Notebook:
    import joblib
    
    # Guardar el modelo entrenado
    joblib.dump(rf_optimizado_final, 'rf_optimizado_final.pkl')
    
    # Para descargarlo desde Colab:
    from google.colab import files
    files.download('rf_optimizado_final.pkl')
    ```
    """)