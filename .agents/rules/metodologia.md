---
trigger: always_on
---

# Regla numero 1: Trazabilidad y Profiling.
	Todo sistema de debug, profiling, test o log ya sea inyectado en el código, en un script o ejecutable creado con fines de obtener información necesaria para la implementación deberá ser documentado con exactitud en profiling.md que se añadirá a artifacts y acompañará al plan_de_implementación.md
	Se indicará cual es su utilidad, la localización exacta en el código para poder eliminarlo cuando sea necesario, o los flags que lo activan/desactivan. Se indicará tb si interfiere en la velocidad de inferencia, o si es un script externo su localización o cómo modifica la ejecución. Al cumplir su cometido, si produce listados, logs o mapas de utilidad o datos del sistema o modelo/s utilizado/s el resultado se guardará en un documento referenciado para su utilización/consulta posterior. Si es un resultado temporal debe generarse en el directorio 'logs' del proyecto. 

# Regla número 2: Desarrollo Atómico y Prohibición de Borrado Ciego
	Queda terminantemente prohibido realizar modificaciones masivas en múltiples archivos a la vez o intentar resolver un bug tocando el código en cascada.
    El ciclo de trabajo será estrictamente atómico: Un objetivo, un archivo (o bloque lógico), una compilación de prueba.
    Prohibición del hacha: Queda prohibido el uso de comandos de reemplazo ciego como sed, o el borrado de líneas usando grep -v (como ocurrió con la limpieza de los DEBUG).
    Verificación forense: Toda modificación de código debe ser validada leyendo el git diff antes de dar el paso por completado. Si se introduce un error de segmentación o cuelgue, el primer paso obligatorio será analizar el diff de la última modificación, nunca intentar adivinar.

# Regla número 3: Higiene Estricta de Compilación
	Nunca confiaremos en la caché del compilador cuando cambiemos el comportamiento fundamental del motor o las herramientas de depuración.
    Si alternamos entre perfiles de construcción (Debug, RelWithDebInfo, Release), si modificamos banderas de arquitectura (- DAMDGPU_TARGETS=gfx1201), o si activamos/desactivamos variables maestras (GGML_HIP), es obligatorio ejecutar una destrucción total de la caché (rm -rf build/) antes de lanzar CMake.
    Cualquier error del enlazador (Linker Error) obligará a realizar una Compilación Unificada (One-Shot Build) desde la raíz del proyecto para resincronizar las librerías estáticas y dinámicas.

# Regla número 4: Anclaje de Contexto (Context Pinning)
	Dado que la ventana de contexto se degrada con la lectura de logs largos, el agente debe auto-imponerse un "anclaje de memoria" al inicio de cada iteración crítica o tras la lectura de un volcado de errores extenso.
    Antes de proponer una solución matemática o de punteros, debe repasar explícitamente las constantes de nuestra arquitectura: la arquitectura de la GPU (RDNA4 / gfx1201), el tamaño estático del vocabulario (vocab_size = 2048 o 1024), y las reglas de dimensionalidad del Paso 0 (Prefill) vs Pasos 1-14 (Autoregresivo).
    Para las operaciones en GPU, debe asegurar siempre que el salto de memoria (stride) se calcula dinámicamente en el kernel de C++ (seq_len - 1) y nunca forzando reestructuraciones previas desde la CPU que causen In-Place aliasing.

# Regla número 5: Herramientas Prohibidas (Lista Negra)
	Para evitar bloqueos interactivos y destrucción ciega de código, el agente tiene estrictamente prohibido usar los siguientes comandos:
    grep -v o reemplazos masivos ciegos (sed agresivos): La limpieza de código se hará siempre identificando la línea exacta y borrándola de forma explícita.
    gdb o cualquier depurador interactivo: El agente se traba en los prompts interactivos. La depuración se hará exclusivamente mediante inyección de logs (fprintf/GGML_LOG), volcados de memoria controlados, o leyendo los errores nativos de la consola.

# Regla número 6: Jurisdicción Exclusiva sobre Git (Solo Lectura)
	El agente actuará únicamente como desarrollador, no como gestor del repositorio.
    Prohibido alterar el estado de Git: El agente no ejecutará NUNCA comandos que cambien el estado del repositorio como git checkout, git branch, git reset, git revert o git commit.
    Solo Lectura Forense: El agente solo tiene permitido usar git diff y git status para analizar sus propios cambios o verificar qué archivos están modificados antes de proceder. La gestión de las ramas es jurisdicción exclusiva del usuario humano.

# Regla número 7: Protocolo de Entorno y Compilación (ROCm 7.0.2)
	Toda compilación o ejecución de prueba debe hacerse estrictamente bajo el entorno configurado para este laboratorio. El agente no intentará adivinar rutas ni usar compiladores del sistema por defecto si no están en el entorno.
    Carga obligatoria del entorno: Antes de compilar (cmake / make) o ejecutar inferencias, es obligatorio asegurar la carga del módulo ROCm 7.0.2 y el entorno Python.
    Comando estandarizado: Se utilizará el alias del sistema del usuario. El comando válido a ejecutar antes de las pruebas será:
    source ~/.bashrc && shopt -s expand_aliases && torch_702

# Regla número 8: Documentación consultable
	El agente debe consultar la documentación en los directorios del proyecto (docs/) y puede acceder al directorio ~/code/docs para consultar los repositorios allí descargados con documentación, ejemplos y código.
 ll ~/code/docs/
Octal Size User Date Modified Name
0775     - gas   1 mar 12:50   HIP
0775     - gas  28 feb 12:20   hipThreads
0775     - gas   1 mar 12:57   llama.cpp
0775     - gas   1 mar 18:30   qwen3_tts.cpp_docs
0775     - gas   1 mar 12:51   rocm-examples
Si el agente cree que es recomendable acceder a otra documentación/repositorio no disponible debe pedirselo al usuario, y éste la podrá aportar de buen grado.