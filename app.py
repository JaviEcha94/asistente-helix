import streamlit as st
import json
import numpy as np
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# ================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ================================================================
st.set_page_config(
    page_title="Helix IA",
    page_icon="🏦",
    layout="centered"
)

# ================================================================
# SETUP — recursos pesados se cachean (cargan una sola vez)
# ================================================================
@st.cache_resource
def cargar_modelo_embeddings():
    return SentenceTransformer('all-MiniLM-L6-v2')

modelo_embeddings = cargar_modelo_embeddings()

client = OpenAI(
    api_key=st.secrets["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

# ================================================================
# BASE DE FAQs (búsqueda semántica asimétrica — Ej4)
# ================================================================
faq_database = [
    {
        "respuesta": "El plazo fijo Helix Premium ofrece una tasa de 118% TNA para montos mayores a $500.000.",
        "variantes": ["tasa del plazo fijo", "cuánto rinde un plazo fijo en Helix", "interés si pongo plata a plazo fijo"]
    },
    {
        "respuesta": "El FCI de renta fija tiene un rendimiento promedio de 9.2% mensual y rescate en 24-48hs.",
        "variantes": ["rendimiento del FCI de renta fija", "cuánto gano si dejo mi plata quieta en un fondo", "rentabilidad mensual del fondo"]
    },
    {
        "respuesta": "El FCI Money Market permite rescate inmediato el mismo día hábil, ideal para liquidez.",
        "variantes": ["necesito la plata ya no puedo esperar", "fondo con liquidez inmediata", "quiero retirar mi dinero el mismo día"]
    },
    {
        "respuesta": "Para abrir una cuenta Helix Black necesitás un patrimonio mínimo de $5.000.000 o ingresos de $500.000 mensuales.",
        "variantes": ["requisitos para la cuenta Helix Black", "cuánto necesito para ser cliente premium"]
    },
    {
        "respuesta": "Podés operar dólar MEP y cable a través de la mesa de cambio de Helix.",
        "variantes": ["cómo compro dólares en Helix", "opciones para dolarizarme"]
    },
    {
        "respuesta": "La tarjeta Mastercard Black no tiene costo de mantenimiento para clientes Helix Black.",
        "variantes": ["la tarjeta black tiene costo de mantenimiento", "cuánto pago por la tarjeta premium"]
    },
    {
        "respuesta": "El horario de atención del equipo de soporte es de lunes a viernes de 9 a 18hs.",
        "variantes": ["a qué hora puedo llamar si tengo un problema", "horario del soporte al cliente"]
    },
    {
        "respuesta": "Para cancelar un plazo fijo antes de la fecha de vencimiento, contactá a tu asesor — puede haber penalidades.",
        "variantes": ["puedo retirar el plazo fijo antes de tiempo", "qué pasa si cancelo el plazo fijo antes"]
    },
    {
        "respuesta": "Helix no cobra comisión por transferencias entre cuentas propias.",
        "variantes": ["hay algún costo por mover dinero entre mis cuentas", "cobran comisión por transferir entre mis cuentas"]
    },
    {
        "respuesta": "Los fondos comunes de inversión no tienen garantía del Estado, a diferencia de los plazos fijos bancarios tradicionales.",
        "variantes": ["el gobierno me devuelve la plata si el fondo quiebra", "los fondos comunes están garantizados por el estado"]
    },
    {
        "respuesta": "Podés consultar el estado de tus inversiones las 24 horas desde la app de Helix.",
        "variantes": ["puedo ver mis inversiones desde el celular", "la app funciona las 24 horas"]
    },
    {
        "respuesta": "Los CEDEARs son certificados que representan acciones de empresas extranjeras (ej: Apple, Tesla, Amazon, Microsoft) y cotizan en pesos en el mercado argentino, permitiendo invertir en el exterior sin necesidad de girar dólares fuera del país. Helix cobra una comisión de 0.5% por operación, sin monto mínimo.",
        "variantes": [
            "qué son los cedears",
            "cómo compro acciones de empresas extranjeras desde Argentina",
            "puedo invertir en Apple o Tesla desde acá",
            "quiero invertir en cedears"
        ]
    },
    {
        "respuesta": "Para armar una cartera diversificada de CEDEARs, Helix recomienda distribuir el monto entre 6-8 empresas de distintos sectores: tecnología (Apple, Microsoft, Nvidia), consumo (Coca-Cola, McDonald's), salud (Pfizer, Johnson & Johnson) y energía (Exxon). Esto reduce el riesgo de concentración en un solo sector o empresa.",
        "variantes": [
            "quiero invertir en una cartera de cedears que me recomiendan",
            "cómo armo una cartera diversificada de acciones extranjeras",
            "en qué empresas me conviene invertir si quiero diversificar",
            "tengo dólares para invertir en cedears, cómo los reparto"
        ]
    },
    {
        "respuesta": "Helix ofrece acceso a bonos soberanos argentinos en dólares (como AL30 y GD30), cuyo rendimiento depende de la cotización de mercado. Son instrumentos con riesgo país y potencial de revalorización, recomendados para perfiles que buscan exposición a renta fija en dólares con mayor volatilidad que un plazo fijo.",
        "variantes": [
            "qué bonos soberanos tienen disponibles",
            "conviene invertir en bonos argentinos en dólares",
            "qué es el AL30 o el GD30",
            "quiero comprar bonos del estado argentino"
        ]
    },
    {
        "respuesta": "A través de CEDEARs también podés acceder a ETFs internacionales que replican índices como el S&P 500 (SPY) o el Nasdaq 100 (QQQ), lo que te da diversificación instantánea en cientos de empresas con una sola operación, ideal si buscás exposición amplia sin elegir acciones individuales.",
        "variantes": [
            "puedo invertir en el sp500 desde Argentina",
            "qué etfs están disponibles para invertir",
            "cómo diversifico en muchas empresas con una sola inversión",
            "quiero invertir en el nasdaq"
        ]
    },
    {
        "respuesta": "Las Obligaciones Negociables (ON) son bonos corporativos emitidos por empresas argentinas en dólares, con cupones de interés periódicos. El cupón promedio en Helix es de 7% anual a 2-5 años, y representan una alternativa a los bonos soberanos con riesgo crediticio de la empresa emisora en lugar del riesgo país.",
        "variantes": [
            "qué son las obligaciones negociables",
            "conviene invertir en on de empresas argentinas",
            "qué diferencia hay entre un bono del estado y uno de una empresa"
        ]
    }
]

todas_variantes, indice_respuesta = [], []
for entry in faq_database:
    for variante in entry["variantes"]:
        todas_variantes.append(variante)
        indice_respuesta.append(entry["respuesta"])


@st.cache_resource
def calcular_embeddings_faqs():
    return modelo_embeddings.encode(todas_variantes)

embeddings_variantes = calcular_embeddings_faqs()


def similitud_coseno(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def buscar_faq(consulta, umbral=0.6):
    """Busca la FAQ más relevante. Devuelve (respuesta, score) o (None, score)."""
    embedding_consulta = modelo_embeddings.encode(consulta)
    similitudes = [
        (similitud_coseno(embedding_consulta, emb), indice_respuesta[i])
        for i, emb in enumerate(embeddings_variantes)
    ]
    similitudes.sort(key=lambda x: x[0], reverse=True)
    score, respuesta = similitudes[0]
    if score >= umbral:
        return respuesta, score
    return None, score


# ================================================================
# FUNCIONES FINANCIERAS + TOOLS (function calling — Ej3)
# ================================================================
def consultar_saldo(cliente_id: str) -> str:
    saldos = {
        "C001": {"saldo_ars": 1500000, "saldo_usd": 8000},
        "C002": {"saldo_ars": 320000, "saldo_usd": 1200},
    }
    datos = saldos.get(cliente_id)
    if not datos:
        return f"No se encontró el cliente {cliente_id}"
    return f"Saldo cliente {cliente_id}: ${datos['saldo_ars']:,} ARS y USD {datos['saldo_usd']:,}"


def listar_movimientos(cliente_id: str, cantidad: int = 3) -> str:
    movimientos = {
        "C001": [
            "Transferencia recibida: +$200.000 (10/06/2026)",
            "Pago de servicios: -$45.000 (08/06/2026)",
            "Compra USD: -$300.000 / +USD 500 (05/06/2026)",
        ],
        "C002": [
            "Acreditación de sueldo: +$320.000 (01/06/2026)",
            "Transferencia enviada: -$50.000 (03/06/2026)",
        ],
    }
    movs = movimientos.get(cliente_id, [])
    if not movs:
        return f"No hay movimientos para el cliente {cliente_id}"
    return "\n".join(movs[:cantidad])


def buscar_producto(nombre_producto: str) -> str:
    catalogo = {
        "plazo fijo": "Plazo Fijo Helix Premium: 118% TNA, plazos de 30 a 365 días.",
        "fci renta fija": "FCI Renta Fija Helix: rendimiento ~9.2% mensual, rescate en 24-48hs.",
        "fci money market": "FCI Money Market Helix: rescate inmediato, ideal para liquidez.",
        "fci": "Helix ofrece dos FCI: Renta Fija (rendimiento ~9.2% mensual, rescate 24-48hs) y "
               "Money Market (rescate inmediato, ideal para liquidez).",
        "fondo": "Helix ofrece dos FCI: Renta Fija (rendimiento ~9.2% mensual, rescate 24-48hs) y "
                 "Money Market (rescate inmediato, ideal para liquidez).",
        "cedears": "CEDEARs Helix: acceso a acciones de empresas extranjeras (Apple, Tesla, Amazon, "
                   "Microsoft, Nvidia, Coca-Cola, entre otras). Comisión de 0.5% por operación, sin "
                   "monto mínimo. Cotizan en pesos.",
        "bonos": "Bonos soberanos Helix: AL30, GD30 y otros, en dólares. Rendimiento sujeto a "
                 "cotización de mercado y riesgo país.",
        "obligaciones negociables": "Obligaciones Negociables Helix: bonos corporativos en USD, "
                                     "cupón promedio 7% anual, plazo 2-5 años.",
        "on": "Obligaciones Negociables Helix: bonos corporativos en USD, cupón promedio 7% anual, "
              "plazo 2-5 años.",
        "etf": "A través de CEDEARs, Helix da acceso a ETFs como SPY (S&P 500) y QQQ (Nasdaq 100), "
               "con comisión de 0.5% por operación.",
        "acciones": "CEDEARs Helix: acceso a acciones de empresas extranjeras (Apple, Tesla, Amazon, "
                    "Microsoft, Nvidia, Coca-Cola, entre otras). Comisión de 0.5% por operación, sin "
                    "monto mínimo. Cotizan en pesos.",
    }
    nombre_lower = nombre_producto.lower()
    if nombre_lower in catalogo:
        return catalogo[nombre_lower]
    for key, value in catalogo.items():
        if key in nombre_lower or nombre_lower in key:
            return value
    return "Producto no encontrado en el catálogo."


funciones_disponibles = {
    "consultar_saldo": consultar_saldo,
    "listar_movimientos": listar_movimientos,
    "buscar_producto": buscar_producto,
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "consultar_saldo",
            "description": "Consulta el saldo disponible (ARS y USD) de un cliente por su ID. Usar SOLO cuando el cliente pide su saldo específico, no para preguntas generales sobre productos.",
            "parameters": {
                "type": "object",
                "properties": {"cliente_id": {"type": "string", "description": "ID del cliente, formato 'C001'."}},
                "required": ["cliente_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "listar_movimientos",
            "description": "Devuelve los últimos movimientos de la cuenta de un cliente. Usar SOLO cuando el cliente pide ver sus movimientos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cliente_id": {"type": "string", "description": "ID del cliente, formato 'C001'."},
                    "cantidad": {"type": "integer", "description": "Cantidad de movimientos. Default 3."}
                },
                "required": ["cliente_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "buscar_producto",
            "description": "Busca condiciones EXACTAS y actualizadas de un producto financiero de Helix (tasas, plazos, montos). Usar cuando el cliente pide datos concretos de un producto específico.",
            "parameters": {
                "type": "object",
                "properties": {"nombre_producto": {"type": "string", "description": "Nombre del producto, ej: 'plazo fijo', 'fci'."}},
                "required": ["nombre_producto"]
            }
        }
    }
]


# ================================================================
# ASISTENTE HELIX — orquestador (FAQ + historial + function calling + log)
# ================================================================
def asistente_helix(mensaje_usuario, historial, cliente_id="C001", log_sesion=None):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "cliente_id": cliente_id,
        "mensaje_usuario": mensaje_usuario,
        "faq_match": None,
        "faq_score": None,
        "tools_llamadas": [],
        "derivado": False,
    }

    # 1. Búsqueda semántica en FAQs — RAG simple
    faq_respuesta, faq_score = buscar_faq(mensaje_usuario)
    registro["faq_score"] = round(float(faq_score), 3)

    contexto_faq = ""
    if faq_respuesta:
        registro["faq_match"] = faq_respuesta
        contexto_faq = f"\n\n[Info de la base de conocimiento Helix: {faq_respuesta}]"

    # 2. System prompt (solo la primera vez)
    if not historial:
        historial.append({
            "role": "system",
            "content": f"""Sos AsesorAI, el asistente virtual de Helix, una fintech digital premium argentina.
            Helix es una fintech, no un banco tradicional: nunca uses esa palabra.
            El cliente que está consultando tiene el ID '{cliente_id}'.
            Sos profesional, conciso y orientado a soluciones.
            Si la consulta está fuera de tu alcance (temas legales, fiscales, reclamos formales,
            o no tenés información suficiente), decilo y ofrecé derivar al asesor de cuenta.
            Nunca inventes tasas, números o condiciones de productos."""
        })

    historial.append({"role": "user", "content": mensaje_usuario + contexto_faq})

    # 3. Request con tools (function calling)
    respuesta = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=historial,
        tools=tools,
        tool_choice="auto",
        temperature=0.4
    )
    mensaje_modelo = respuesta.choices[0].message

    if mensaje_modelo.tool_calls:
        historial.append(mensaje_modelo)
        for tool_call in mensaje_modelo.tool_calls:
            nombre_funcion = tool_call.function.name
            argumentos = json.loads(tool_call.function.arguments)
            registro["tools_llamadas"].append({"funcion": nombre_funcion, "argumentos": argumentos})

            resultado = funciones_disponibles[nombre_funcion](**argumentos)
            historial.append({
                "role": "tool", "tool_call_id": tool_call.id,
                "name": nombre_funcion, "content": resultado
            })

        respuesta_final = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=historial, temperature=0.4
        )
        texto_respuesta = respuesta_final.choices[0].message.content
    else:
        texto_respuesta = mensaje_modelo.content

    historial.append({"role": "assistant", "content": texto_respuesta})

    # 4. Detección de derivación real (frases de incapacidad explícita)
    frases_derivacion_real = [
        "no puedo ayudarte con", "no puedo resolver", "está fuera de mi alcance",
        "no tengo información suficiente", "no puedo asistirte con"
    ]
    registro["derivado"] = any(p in texto_respuesta.lower() for p in frases_derivacion_real)
    registro["respuesta_asistente"] = texto_respuesta

    if log_sesion is not None:
        log_sesion.append(registro)

    return texto_respuesta


# ================================================================
# INTERFAZ DE CHAT (Streamlit)
# ================================================================
st.title("🏦 Helix IA")
st.caption("Asistente financiero — proyecto de portfolio | Helix es una fintech digital ficticia")
st.divider()

if "historial" not in st.session_state:
    st.session_state.historial = []
if "log_sesion" not in st.session_state:
    st.session_state.log_sesion = []

# Mostrar historial (sin el system prompt ni el contexto FAQ inyectado)
for msg in st.session_state.historial:
    if msg["role"] == "user":
        contenido = msg["content"].split("\n\n[Info de la base de conocimiento Helix:")[0]
        with st.chat_message("user"):
            st.write(contenido)
    elif msg["role"] == "assistant":
        with st.chat_message("assistant"):
            st.write(msg["content"])

# Input del usuario
if prompt := st.chat_input("💬 Escribí tu consulta financiera..."):
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 Procesando tu consulta..."):
            respuesta = asistente_helix(
                prompt,
                st.session_state.historial,
                cliente_id="C001",
                log_sesion=st.session_state.log_sesion
            )
        st.write(respuesta)

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.write("**Cliente de prueba:** C001 (Carlos)")
    st.write("**Probá preguntar:**")
    st.write("• ¿Cuánto rinde el FCI de renta fija?")
    st.write("• ¿Cuál es mi saldo?")
    st.write("• Quiero invertir en CEDEARs, ¿qué me recomendás?")
    st.write("• Necesito liquidez inmediata, ¿qué me conviene?")
    st.divider()
    if st.button("🗑️ Reiniciar conversación"):
        st.session_state.historial = []
        st.session_state.log_sesion = []
        st.rerun()
