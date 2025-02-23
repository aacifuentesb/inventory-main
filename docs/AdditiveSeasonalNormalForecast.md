# Pronóstico de Distribución Normal con Estacionalidad

## Descripción General
El modelo de Pronóstico de Distribución Normal con Estacionalidad es una herramienta avanzada para predecir demanda que combina la precisión estadística con el reconocimiento de patrones estacionales. Este modelo es especialmente útil para productos que muestran variaciones regulares en su demanda a lo largo del tiempo.

## Fundamento Matemático

### 1. Descomposición Estacional
```
Yt = Tt + St + Rt  (Aditiva)
```
donde:
- Yt: Serie temporal original
- Tt: Componente de tendencia
- St: Componente estacional
- Rt: Componente residual

### 2. Factores Estacionales
```
FE[i] = (Base + S[i]) / Base
```
donde:
- FE[i]: Factor estacional para el período i
- Base: Media de la serie
- S[i]: Componente estacional aditivo

### 3. Generación del Pronóstico
```
P[t] = N(μ, σ) × FE[t mod s]
```
donde:
- N(μ, σ): Distribución normal con media μ y desviación estándar σ
- FE[t mod s]: Factor estacional para el período t
- s: Longitud del período estacional (4 semanas)

### 4. Intervalos de Confianza
```
IC[t] = P[t] ± t(α/2, n-1) × σ × FE[t mod s] / √n
```
donde:
- t(α/2, n-1): Valor de la distribución t
- n: Tamaño de la muestra
- α: Nivel de significancia (0.05 para IC del 95%)

## ¿Cómo Funciona?

### 1. Detección de Estacionalidad
- Analiza datos históricos semanales para identificar patrones mensuales
- Utiliza descomposición aditiva para separar la señal estacional
- Valida la significancia de los patrones encontrados

### 2. Proceso de Pronóstico
1. **Preparación de Datos**
   - Limpia y prepara los datos históricos
   - Maneja valores cero y datos faltantes
   - Verifica la calidad de los datos

2. **Análisis Estadístico**
   - Calcula la media y desviación estándar
   - Ajusta los valores según los factores estacionales
   - Genera múltiples pronósticos y selecciona el mejor

3. **Generación de Intervalos de Confianza**
   - Calcula intervalos ajustados por estacionalidad
   - Proporciona límites superior e inferior
   - Considera la incertidumbre creciente en el tiempo

