# Wine quality prediction

Modelling a Kaggle dataset of [red wine properties and quality ratings](https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009).

Fork de [icebeam7/mlops-ci](https://github.com/icebeam7/mlops-ci). Se conserva el modelo original: `RandomForestRegressor(max_depth=2, random_state=42)` y una división entrenamiento/prueba de 80/20 con semilla 42.

## Instalación

Requiere Python 3.12. Desde una terminal:

```bash
git clone https://github.com/ClaudiaCurso/mlops-ci.git
cd mlops-ci
python -m venv .venv
```

Activa el entorno:

- Windows (CMD): `.venv\Scripts\activate.bat`
- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- Linux/macOS: `source .venv/bin/activate`

Después, desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python train.py
```

Las dependencias directas están fijadas a las versiones utilizadas para validar el cambio. No se incluye un bloqueo completo de dependencias transitivas.

## Entrenamiento

`python train.py` lee `wine_quality.csv` y escribe `metrics.txt`, `feature_importance.png` y `residuals.png` en el directorio actual (sobrescribe archivos con esos nombres). Las métricas son R² multiplicado por 100; pueden ser negativas.

Importar `train` no ejecuta el entrenamiento. Para invocarlo desde Python:

```python
from train import train
scores = train(data_path="wine_quality.csv", output_dir="results")
```

La función devuelve `train_score` y `test_score`. Se conserva el ruido visual aleatorio del gráfico original de residuos: las imágenes pueden variar entre ejecuciones; ese ruido no interviene en las métricas.

## Qué valida la prueba

Una prueba de humo automatizada con `unittest` comprueba que:

- Una importación en un proceso nuevo funciona sin el CSV en el directorio actual y no crea archivos allí.
- El entrenamiento con el CSV incluido termina y devuelve dos métricas finitas, no mayores de 100.
- `metrics.txt` contiene las métricas devueltas con el formato esperado.
- Ambos gráficos se generan y pueden leerse como imágenes PNG.

Los resultados de la prueba se escriben en un directorio temporal que se elimina al terminar. Esta prueba detecta fallos básicos de importación, entrenamiento y generación de resultados. No establece una calidad predictiva mínima ni valida todas las entradas posibles.

## Integración continua

`.github/workflows/tests.yml` instala las mismas dependencias y ejecuta el mismo comando de prueba con Python 3.12 en cada `push` y `pull_request`, sin filtros de rama. Usa `Agg` para generar gráficos sin interfaz gráfica. GitHub Actions debe estar habilitado en el fork; el resultado aparece en la pestaña **Actions** y en las comprobaciones del pull request.

Se sustituyeron los archivos vacíos `.github` y `workflows` por el directorio válido `.github/workflows/`. No se añaden despliegues ni cambios al modelo.
