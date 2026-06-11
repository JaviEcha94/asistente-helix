# 🏦 AsesorAI Helix

Asistente fintech inteligente construido con LLMs — proyecto de portfolio que integra
**historial de conversación**, **búsqueda semántica (RAG)** sobre una base de FAQs, y
**function calling** sobre datos de cliente, con logging de auditoría y derivación
automática a un asesor humano cuando corresponde.

> **Helix** es una fintech digital **ficticia**, creada para este proyecto de portfolio.
> Cualquier parecido con productos financieros reales es solo ilustrativo.

🚀 **Probalo en vivo:** `[link de Streamlit Cloud — agregar después del deploy]`

---

## El problema que resuelve

Un asistente de atención al cliente para una fintech necesita poder:

1. Mantener el **contexto** de toda la conversación (no pedirle al cliente que repita información)
2. Responder preguntas sobre productos usando **lenguaje natural del cliente**, no solo
   las palabras exactas de un manual
3. Consultar **datos reales** del cliente (saldo, movimientos) en lugar de inventarlos
4. Saber **cuándo no puede resolver algo** y derivar a un humano

Este proyecto implementa las cuatro capacidades en un único sistema, con datos simulados.

---

## Arquitectura

```
Usuario escribe un mensaje
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
│    + function calling                        │
│    → puede llamar consultar_saldo,           │
│      listar_movimientos, buscar_producto     │
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
│    derivación, respuesta                     │
└────────────────────────────────────────────┘
```

---

## Stack

| Componente | Tecnología | Por qué |
|---|---|---|
| LLM | [Groq](https://groq.com) — `llama-3.3-70b-versatile` | API gratuita, compatible con el SDK de OpenAI (`base_url` + modelo es lo único que cambia) |
| Embeddings / búsqueda semántica | `sentence-transformers` (`all-MiniLM-L6-v2`) | 100% local, sin API key, sin costo |
| Interfaz | [Streamlit](https://streamlit.io) | Convierte el script en una app web con URL pública |
| Logging | JSON | Auditoría de cada interacción |

---

## Cómo correr localmente

```bash
git clone https://github.com/TU_USUARIO/asistente-helix.git
cd asistente-helix
pip install -r requirements.txt

# Crear .streamlit/secrets.toml con:
# GROQ_API_KEY = "tu_api_key_de_groq"

streamlit run app.py
```

Conseguí una API Key gratuita en [console.groq.com/keys](https://console.groq.com/keys).

---

## Ejemplos de conversación

### Conversación 1 — FAQ + memoria + function calling (cliente C001)

> **Cliente:** Hola, soy Carlos. Tengo $2.000.000 para invertir, ¿cuánto rinde el FCI de renta fija?
> **AsesorAI:** El FCI de renta fija ofrece un rendimiento promedio de 9.2% mensual...
>
> **Cliente:** Buenísimo. Por otro lado, ¿cuál es mi saldo actual?
> **AsesorAI:** Tu saldo actual es de $1.500.000 ARS y $8.000 USD...
>
> **Cliente:** ¿Y si en cambio necesito algo con liquidez inmediata?
> **AsesorAI:** Te recomiendo nuestro FCI Money Market, con rescate el mismo día hábil...

### Conversación 2 — Derivación real (cliente C002)

> **Cliente:** Quiero hacer un reclamo formal: me cobraron una comisión indebida hace 6 meses.
> **AsesorAI:** Lo siento, pero no puedo ayudarte con reclamos formales. Te recomiendo
> que te comuniques con tu asesor de cuenta...

---

## Hallazgos y decisiones de diseño

Este proyecto documenta el proceso real de iteración, no solo el resultado final:

**1. Búsqueda semántica — el corpus importa más que el modelo.**
La primera versión indexaba *respuestas* (texto formal de manual) y obtenía similitudes
de 0.43-0.51 — insuficiente para un umbral confiable, y en un caso devolvió una FAQ
incorrecta con alta confianza. La solución fue indexar *preguntas/variantes* (cómo
pregunta un cliente real) en vez de respuestas — **asymmetric semantic search**, un
patrón estándar en sistemas RAG. Resultado: similitudes de 0.87-0.98, 5/5 correctas.

**2. Heurística de derivación — falso positivo corregido.**
Detectar derivación buscando "asesor de cuenta" en el texto generaba falsos positivos:
el modelo ofrece contacto con el asesor como cortesía/upsell incluso cuando *sí* resolvió
la consulta. La heurística se ajustó para detectar solo frases de incapacidad explícita
("no puedo ayudarte con...", "está fuera de mi alcance").

**3. Modelos "reasoning" rompen el parseo JSON.**
Modelos como Qwen3 anteponen un bloque `<think>...</think>` en inglés antes de la
respuesta — si se intenta `json.loads()` directo sobre eso, falla. Se optó por
`llama-3.3-70b-versatile`, que no genera este bloque.

**4. Limitación conocida:** cuando una FAQ matchea, el modelo a veces igual llama a
`buscar_producto` para el mismo dato — una llamada redundante. Mejora futura: instruir
al modelo para que reconozca cuándo el contexto FAQ ya es suficiente.

---

## Estructura del repo

```
asistente-helix/
├── app.py                    # App de Streamlit (deploy)
├── asistente_helix.ipynb     # Notebook de desarrollo y pruebas (Ej1-5)
├── requirements.txt
├── conversaciones_log.json   # Ejemplo de log de auditoría generado
└── README.md
```

---

## Autor

**Javier Echalecu** — Banca & Fintech | AI Automation
[LinkedIn](#) · [GitHub](#)
