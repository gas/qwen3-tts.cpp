# Q3VP (Qwen3 Voice Profile) Format Specification

El formato `.q3vp` (Qwen3 Voice Profile) es un formato binario diseñado específicamente para almacenar perfiles de voz pre-computados para el motor Qwen3-TTS de inferencia en C++. Al pre-procesar el audio de referencia offline, se ahorra impacto de cómputo en el tiempo de inferencia y se permite la reutilización de características consistentes del locutor en múltiples sesiones o instancias de servidor.

## 1. ¿Cómo generar un archivo `.q3vp`?

El repositorio incluye un script nativo en Python que aprovecha el paquete oficial de transformers `qwen_tts` para extraer internamente los embeddings fonéticos de manera idéntica a la implementación original del modelo creador de Qwen.

### Comando básico de generación:

```bash
# Activar el entorno virtual o entorno de Python que contenga qwen_tts y audioread
source ~/.envs/torch_rocm702/bin/activate

# Ejecutar el script (soporta WAV, de ser posible ya resampleado a 24000 Hz)
python3 scripts/extract_voice_profile.py \
    -m /ruta/al/modelo/Qwen3-TTS \
    -i /ruta/al/audio_de_referencia.wav \
    -o /ruta/al/salida_perfil.q3vp
```

El script genera rápidamente el archivo `.q3vp` en disco conteniendo tanto el *speaker embedding* (X-Vector base de timbre) genérico y los *acoustic codes* (códigos discretos dependientes del In-Context Learning).

---

## 2. Estructura Interna del Archivo (Binario)

El archivo `.q3vp` utiliza una estructura binaria secuencial, en alineamiento *little-endian*, empaquetada herméticamente en tres bloques lógicos: la cabecera (metadata), el *speaker embedding* (características fijas), y los códigos acústicos (secuencia tokenizada).

### A. Cabecera (Metadata Inicial)
| Offset | Tamaño (Bytes) | Valor | Descripción |
|---|---|---|---|
| `0x00` | 4 | `char[4]` | **Magic Number**: Siempre la cadena `"Q3VP"`. |
| `0x04` | 4 | `uint32_t` | **Version**: Versión del formato de iteración. Actualmente `1`. |
| `0x08` | 4 | `uint32_t` | **Sample Rate**: Frecuencia de muestreo original del decoder. Normalmente `24000`. |
| `0x0C` | 4 | `uint32_t` | **Speaker Dim**: Dimensión tensorial del X-Vector. En la familia Qwen3 es `2048`. |
| `0x10` | 4 | `uint32_t` | **Num Frames**: Número de tramas de tiempo extraídas de tokens acústicos de In-Context. |
| `0x14` | 4 | `uint32_t` | **Num Quantizers**: Cantidad total de cuantizadores residuales por frame (Codebooks). En Qwen3 es `16`. |

### B. Payload 1: Speaker Embedding (Generador de Timbre "X-Vector")
- **Tamaño Fijo:** `Speaker Dim * 4` (Para una dimensión de 2048 son típicamente **8192 bytes**).
- **Tipo de Dato:** Array secuencial en C++ std::vector de `float` (32-bit de coma flotante).
- **Utilidad:** Modela pasivamente la forma de onda, ecualización, tono y matices de cuerda vocal de la persona locutora abstraída totalmente de su contenido hablado (lenguaje, ritmo o ruido). 

### C. Payload 2: Acoustic Codes (Códigos de In-Context Learning de Prosodia ICL)
- **Tamaño Variable:** Dependiente del largo del audio de entrada (`Num Frames * Num Quantizers * 4` bytes).
- **Tipo de Dato:** Construcción multidimensional aplanada a un array `int32_t`.
- **Utilidad:** Consiste en la tokenización precisa del audio de referencia cuantizada por los *codebooks*. El Transformer Autoregresivo usará estos códigos insertándolos nativamente antes de los tokens a generar como contexto directo de los segundos previos, forzando a adaptarse a la pronunciación, la velocidad silábica, la prosodia y cadencia del audio en la síntesis que le sucederá.

---

## 3. Utilización como Condicionamiento de Inferencia en C++

Existen dos estrategias para inyectar este perfil compilado durante la síntesis.

### Método 1: Inferencia Estándar Full ICL (Timbre + Prosodia Estricta) 
Al cargar todo el perfil (`Payload 1` y `Payload 2`), el modelo tratará de comportarse como una continuación de tu audio de referencia. Requerirá indispensablemente del texto exacto usado en tu sample referencial a manera de alineación.

```bash
./build/qwen3-tts-cli -m ./models \
    -r mi_perfil_exportado.q3vp \
    -p "Transcript. El texto idéntico y muy preciso dictado en dicho audio." \
    -t "Hola. Soy el Transformer continuando mi clonación a partir de aquel punto."
```

> ⚠️  **Cuidado con Inestabilidad ICL:** 
La familia de modelos Qwen3 sufre de sensibilidad arquitectónica inherente en la inyección de `Payload 2`. Si el transcript (`-p`) está desacompasado con el texto real del audio, o hay discrepancias en la tokenización, el salto al inicio del `-t` generará graves distorsiones. Un artefacto clásico documentado en estos modelos es que los primeros tokens post-ICL alcen su tono enormemente de forma sintética para luego tender a irse estabilizando paulatinamente bajando en *pitch* a medida que el audio fluye.


### Método 2: Inferencia Puramente de Timbre (Modo `X-Vector Only` o `--x-vector-only`) 
Usando este modificador flag, desactivaremos y le instruiremos nativamente a `qwen3_tts_c_api` saltarse la lectura e inclusión de carga computacional del `Payload 2` del documento `.q3vp`. Se abstraerá sólo el `Speaker Embedding` y generará la cadena de inicio a fin desde cero.

```bash
./build/qwen3-tts-cli -m ./models \
    -r mi_perfil_exportado.q3vp \
    -x \
    -t "Hola. Noto de inmediato como el sistema clona perfectamente mi voz, aunque escoja yo las cadencias de prosodia al azar."
```

Dentro del servidor REST JSON:
```json
{
  "texts": ["Texto a sintetizar desde la API."],
  "voice_ref": "mi_perfil_exportado.q3vp",
  "x_vector_only": true
}
```

**Ventajas Definitivas de `-x`:**
- **Estabilidad Acústica Absoluta:** Al desahogar al transformador generativo de predecir la correlación con la secuencia pre-tokenizada, **hemos comprobado** experimentalmente que los fallos extremos de tono ascendentes en Qwen3 al arrancar desaparecen.
- **Independencia del Texto (No Requiere -p):** Puesto que se ha podado la información del audio, no hay contexto alineable, por ende *no obliga* a pre-transcribir o requerir el parámetro que alimentaba la alineación.
- **Eficacia Extrema de Latencia:** Evita todo tipo de cálculos de embedding contextual anterior a la síntesis haciendo del factor Tiempo Para El Primer Primer Token (TTFT) significativamente más veloz en perfiles de grandes minutos de duración.
