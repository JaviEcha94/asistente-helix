# 🏦 Helix IA

Asistente fintech inteligente construido con LLMs — proyecto de portfolio que integra
**historial de conversación**, **búsqueda semántica (RAG)** sobre una base de FAQs,
**function calling** sobre datos de cliente, y **acceso a datos financieros reales en
tiempo real** (cotización del dólar vía API, búsqueda web para otros activos), con
logging de auditoría, soporte multi-idioma (ES/EN/PT), y derivación automática a un
asesor humano cuando corresponde.

> **Helix** es una fintech digital **ficticia**, creada para este proyecto de portfolio.
> Cualquier parecido con productos financieros reales es solo ilustrativo.

🚀 **Probalo en vivo:** https://asistente-helix.streamlit.app

---

## El problema que resuelve

Un asistente de atención al cliente para una fintech necesita poder:

1. Mantener el **contexto** de toda la conversación
2. Responder preguntas sobre productos usando **lenguaje natural del cliente**, no las
   palabras exactas de un manual
3. Consultar **datos reales** del cliente (saldo, movimientos) en lugar de inventarlos
4. Acceder a **datos del mercado en tiempo real** (cotización del dólar, activos) sin
   alucinar cifras
5. Saber **cuándo no puede resolver algo** y derivar a un humano
6. Atender en **varios idiomas**

Este proyecto implementa las seis capacidades en un único sistema, con datos simulados
para lo específico de Helix y datos reales para lo que cambia día a día.

---

## Arquitectura

```
Usuario escribe un mensaje (ES / EN / PT)
        │
        ▼
┌────────────────────────────────────────────┐
│ 1. Búsqueda semántica en FAQs               │
│    (sentence-transformers, local)            │
│    → si hay match fuerte (≥0.6), se inyecta  │
│      como contexto adicional                 │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ 2. LLM (Groq · Llama 3.3 70B) + historial    │
│    + function calling (5 tools):             │
│    - consultar_saldo                         │
│    - listar_movimientos                      │
│    - buscar_producto (catálogo Helix)        │
│    - obtener_cotizacion_dolar (API real)     │
│    - buscar_informacion_actual (web search)  │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ 3. Detección de derivación                   │
│    → frases de incapacidad explícita         │
│      → deriva al asesor humano               │
└────────────────────────────────────────────┘
        │
        ▼
┌────────────────────────────────────────────┐
│ 4. Log de auditoría (JSON)                   │
│    timestamp, FAQ match, tools usadas,       │
│    derivación, errores, respuesta            │
└────────────────────────────────────────────┘
```

---

## Stack

| Componente | Tecnología | Por qué |
|---|---|---|
| LLM | [Groq](https://groq.com) — `llama-3.3-70b-versatile` | API gratuita, compatible con el SDK de OpenAI |
| Embeddings / búsqueda semántica | `sentence-transformers` (`all-MiniLM-L6-v2`) | 100% local, sin API key, sin costo |
| Cotización del dólar | [dolarapi.com](https://dolarapi.com) | API pública argentina, gratuita, sin key, datos confiables (oficial/blue/MEP/CCL/cripto/tarjeta) |
| Búsqueda web (otros activos/noticias) | `ddgs` (DuckDuckGo) | Gratuita, sin API key |
| Interfaz | [Streamlit](https://streamlit.io) | Convierte el script en una app web con URL pública |
| Idiomas | Diccionario de traducciones ES/EN/PT | UI + instrucción de idioma al LLM, sin servicios externos de traducción |
| Logging | JSON | Auditoría de cada interacción |

---

## Cómo correr localmente

```bash
git clone https://github.com/JaviEcha94/asistente-helix.git
cd asistente-helix
pip install -r requirements.txt

# Crear .streamlit/secrets.toml con:
# GROQ_API_KEY = "tu_api_key_de_groq"

streamlit run app.py
```

Conseguí una API Key gratuita en [console.groq.com/keys](https://console.groq.com/keys).

---

## Funcionalidades

- **Chat con memoria**: mantiene contexto de toda la sesión
- **Botones de preguntas sugeridas**: un click envía la consulta, sin escribir
- **Selector de idioma** 🇦🇷 / 🇺🇸 / 🇧🇷: traduce la interfaz completa y el idioma de respuesta del modelo (cambiar de idioma reinicia la conversación, a propósito)
- **Datos en tiempo real**: cotización del dólar (API dedicada) y búsqueda web para otros activos/noticias
- **Sección "Sobre este proyecto"**: atribución y links a portfolio y LinkedIn
- **Manejo de errores robusto**: si una llamada a la API falla, la app no se rompe — responde con un mensaje y limpia el historial para continuar la conversación

---

## Ejemplos de conversación

### FAQ + memoria + function calling (cliente C001)

> **Cliente:** Hola, soy Carlos. Tengo $2.000.000 para invertir, ¿cuánto rinde el FCI de renta fija?
> **AsesorAI:** El FCI de renta fija ofrece un rendimiento promedio de 9.2% mensual...
>
> **Cliente:** ¿Cuál es mi saldo?
> **AsesorAI:** Tu saldo actual es de $1.500.000 ARS y $8.000 USD...

### Datos en tiempo real (API del dólar)

> **Cliente:** ¿Cuál es la cotización del dólar hoy?
> **AsesorAI:** Dólar Oficial: compra $1405 / venta $1455. Dólar Blue: compra $1430 /
> venta $1450. Dólar MEP: compra $1453.7 / venta $1455.3... (datos reales, actualizados
> al momento de la consulta, vía dolarapi.com)

### Derivación real (cliente C002)

> **Cliente:** Quiero hacer un reclamo formal: me cobraron una comisión indebida hace 6 meses.
> **AsesorAI:** Lo siento, pero no puedo ayudarte con reclamos formales. Te recomiendo
> que te comuniques con tu asesor de cuenta...

---

## Hallazgos y decisiones de diseño

Este proyecto documenta el proceso real de iteración, no solo el resultado final.

### 1. Búsqueda semántica — el corpus importa más que el modelo

La primera versión indexaba *respuestas* (texto formal de manual) y obtenía similitudes
de 0.43-0.51 — insuficiente para un umbral confiable. La solución fue indexar
*preguntas/variantes* (cómo pregunta un cliente real) en vez de respuestas —
**asymmetric semantic search**, un patrón estándar en sistemas RAG. Resultado:
similitudes de 0.87-0.98, 5/5 correctas.

### 2. Heurística de derivación — falso positivo corregido

Detectar derivación buscando "asesor de cuenta" en el texto generaba falsos positivos:
el modelo ofrece contacto con el asesor como cortesía/upsell incluso cuando *sí*
resolvió la consulta. La heurística se ajustó para detectar solo frases de incapacidad
explícita ("no puedo ayudarte con...", "está fuera de mi alcance").

### 3. Modelos "reasoning" rompen el parseo JSON

Modelos como Qwen3 anteponen un bloque `<think>...</think>` en inglés antes de la
respuesta — si se intenta `json.loads()` directo sobre eso, falla. Se optó por
`llama-3.3-70b-versatile`, que no genera este bloque.

### 4. ⚠️ Hallazgo crítico: alucinación numérica con activos externos

Al preguntar por la cotización de un CEDEAR específico (SPY 500), el modelo **inventó
un valor** ("$723,00 por unidad") sin llamar a ninguna función ni tener ninguna fuente.
Al preguntársele de dónde obtuvo ese dato, **admitió la alucinación**: *"no obtuve el
valor de $723,00 de ninguna fuente confiable"*.

**Causa:** la instrucción original ("nunca inventes tasas, números o condiciones de
productos") estaba implícitamente acotada a "productos *de Helix*" — el modelo no la
aplicó a activos externos que "cree conocer" de su entrenamiento.

**Fix:** una regla anti-alucinación explícita y universal en el system prompt — *"nunca
generes un número, precio o cotización por tu cuenta, ni de productos de Helix ni de
activos externos, aunque creas conocer el valor aproximado. Si no tenés un dato
verificado, decilo explícitamente."*

**Limitación honesta:** ningún prompt es 100% infalible. La mitigación robusta en
producción sería validar programáticamente que cualquier cifra numérica en la
respuesta provenga literalmente de un resultado de función — fuera del alcance de este
proyecto, pero es el siguiente paso lógico para un sistema real.

### 5. Confiabilidad desigual de los datos financieros abiertos en Argentina

El **dólar** tiene una API pública, gratuita y confiable (`dolarapi.com`) con
cotizaciones oficial/blue/MEP/CCL/cripto/tarjeta actualizadas. Los **precios de CEDEARs
individuales**, en cambio, no tienen un equivalente gratuito y confiable sin
registro — se depende de búsqueda web genérica, sesgada hacia fuentes como BYMA, sin
garantía de precisión. Esto refleja la madurez desigual de la infraestructura de datos
abiertos en distintos segmentos del mercado financiero argentino, y es una limitación
real a tener en cuenta en cualquier integración similar.

### 6. Particularidades de la API de Groq con function calling

- Un mensaje `assistant` con `tool_calls` y `content: null` es rechazado por Groq
  (`BadRequestError`) — hay que enviar `content: ""` en su lugar.
- Un tool con `"parameters": {"type": "object", "properties": {}}` (sin parámetros) es
  rechazado — hace falta al menos una propiedad declarada (aunque sea opcional) y
  `"required": []` explícito.
- Cuando el modelo llama a una función sin pasarle argumentos, Groq puede devolver
  `arguments: "null"` en vez de `"{}"` — `json.loads("null")` da `None`, y `**None`
  rompe. Fix: `json.loads(...) or {}`.

Estas particularidades no están siempre documentadas y se descubren iterando — son
justamente el tipo de detalle que demuestra trabajo real con la API, no solo seguir un
tutorial.

---

## Estructura del repo

```
asistente-helix/
├── app.py                    # App de Streamlit (deploy)
├── asistente_helix.ipynb     # Notebook de desarrollo y pruebas (Ej1-5)
├── requirements.txt
├── conversaciones_log.json   # Ejemplo de log de auditoría generado
├── .streamlit/
│   └── config.toml           # Tema visual (colores light/dark)
└── README.md
```

---

## Autor

**Javier Echalecu** — Banca & Fintech | AI Automation
[Portfolio](https://app.notion.com/p/Portfolio-Javier-Echalecu-37a3e7f37f9881e881b6fa18d5005ef5?source=copy_link) · [LinkedIn](https://linkedin.com/in/javier-nicolas-echalecu/) · [GitHub](https://github.com/JaviEcha94)
