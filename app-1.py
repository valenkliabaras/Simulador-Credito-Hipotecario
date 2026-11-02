"""
app.py — Simulador de Crédito Hipotecario UVA (Streamlit)

Página con dos pestañas:
  1) Simulador: formulario que dice si te conviene el crédito y qué banco te conviene.
  2) Asesor IA (chat): chat con Gemini, que usa nuestras propias funciones para calcular.

Usa las mismas funciones que el notebook (calcular_cuota, comparar_bancos, evaluar_credito,
buscar_opciones, responder_asesor), reunidas en un solo archivo para publicarlas con Streamlit.
"""

import os
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from google import genai
from google.genai import types

st.set_page_config(page_title="Simulador de Crédito Hipotecario UVA", page_icon="🏠")

# =====================================================
# DATOS DE REFERENCIA (iguales a los del notebook)
# =====================================================

# Precios en USD — fuente: informes públicos de Zonaprop
precios_zona = {
    "Capital":    {1: 108337, 2: 130391, 3: 179389},
    "Zona Norte": {1: 40*2381, 2: 55*2381, 3: 70*2381},
    "Zona Sur":   {1: 40*1576, 2: 55*1576, 3: 70*1576},
}

# Condiciones de cada banco — relevamiento manual, septiembre/octubre 2026
BANCOS = {
    "Banco Nación":      {"tasa_anual": 0.067, "plazo_meses": 360, "financiacion_maxima": 0.75},
    "Banco Hipotecario": {"tasa_anual": 0.075, "plazo_meses": 180, "financiacion_maxima": 0.80},
    "BBVA":              {"tasa_anual": 0.075, "plazo_meses": 360, "financiacion_maxima": 0.80},
    "Banco Galicia":     {"tasa_anual": 0.075, "plazo_meses": 240, "financiacion_maxima": 0.70},
    "Santander":         {"tasa_anual": 0.095, "plazo_meses": 240, "financiacion_maxima": 0.70},
}

LIMITE_CUOTA_INGRESO = 25     # los bancos exigen cuota <= 25% del ingreso
MODELO_IA = "gemini-3.5-flash"
ARCHIVO_CONSULTAS = "consultas_hipotecario.csv"

# La clave de Gemini NO va escrita en el código: se guarda como "Secret" en Streamlit Cloud
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


# =====================================================
# FUNCIONES DE CÁLCULO (las mismas del notebook)
# =====================================================

def obtener_dolar_oficial():
    """Trae el dólar oficial (venta) desde DolarAPI. Si falla, usa un valor de referencia."""
    try:
        respuesta = requests.get("https://dolarapi.com/v1/dolares/oficial", timeout=5)
        respuesta.raise_for_status()
        datos = respuesta.json()
        return datos["venta"], datos["fechaActualizacion"]
    except Exception:
        return 1450, "sin conexión (valor de referencia)"


def calcular_cuota(monto_ars, tasa_anual, plazo_meses):
    """Cuota mensual con sistema de amortización francés."""
    tasa_mensual = tasa_anual / 12
    cuota = monto_ars * tasa_mensual / (1 - (1 + tasa_mensual) ** (-plazo_meses))
    return cuota


def comparar_bancos(precio_usd, ahorro_usd, ingreso_mensual_ars, tipo_de_cambio):
    """Calcula la cuota en cada banco. Primero los que alcanzan con el ahorro, de menor a mayor cuota."""
    filas = []

    for nombre_banco, condiciones in BANCOS.items():
        ahorro_minimo = precio_usd * (1 - condiciones["financiacion_maxima"])
        alcanza = ahorro_usd >= ahorro_minimo

        if alcanza:
            monto_usd = max(precio_usd - ahorro_usd, 0)
            monto_ars = monto_usd * tipo_de_cambio
            cuota = calcular_cuota(monto_ars, condiciones["tasa_anual"], condiciones["plazo_meses"])
            ratio = cuota / ingreso_mensual_ars * 100
        else:
            cuota = None
            ratio = None

        filas.append({
            "banco": nombre_banco,
            "tasa_anual": condiciones["tasa_anual"],
            "plazo_anios": condiciones["plazo_meses"] // 12,
            "ahorro_minimo_usd": round(ahorro_minimo, 2),
            "cuota_mensual_ars": None if cuota is None else round(cuota, 2),
            "ratio_cuota_ingreso": None if ratio is None else round(ratio, 2),
            "alcanza_ahorro": alcanza,
        })

    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values(by=["alcanza_ahorro", "cuota_mensual_ars"], ascending=[False, True])
    return tabla.reset_index(drop=True)


def evaluar_credito(nombre, edad, zona, tipologia, ingreso_mensual_ars, ahorro_usd, tipo_de_cambio):
    """Elige el banco más conveniente y arma el semáforo (VIABLE / AJUSTADO / NO RECOMENDABLE)."""
    precio_usd = precios_zona[zona][tipologia]
    tabla = comparar_bancos(precio_usd, ahorro_usd, ingreso_mensual_ars, tipo_de_cambio)
    bancos_posibles = tabla[tabla["alcanza_ahorro"]]

    ahorro_minimo = tabla["ahorro_minimo_usd"].min()
    ahorro_faltante = max(ahorro_minimo - ahorro_usd, 0)

    if len(bancos_posibles) == 0:
        resultado = "NO ALCANZA EL AHORRO MÍNIMO"
        banco = None
        cuota = None
        ratio = None
    else:
        mejor = bancos_posibles.iloc[0]
        banco = mejor["banco"]
        cuota = mejor["cuota_mensual_ars"]
        ratio = mejor["ratio_cuota_ingreso"]

        if ratio <= LIMITE_CUOTA_INGRESO:
            resultado = "VIABLE"
        elif ratio <= LIMITE_CUOTA_INGRESO + 15:
            resultado = "AJUSTADO"
        else:
            resultado = "NO RECOMENDABLE"

    return {
        "nombre": nombre,
        "edad": edad,
        "zona": zona,
        "tipologia": tipologia,
        "ingreso_mensual_ars": ingreso_mensual_ars,
        "ahorro_usd": ahorro_usd,
        "precio_propiedad_usd": precio_usd,
        "ahorro_minimo_usd": round(ahorro_minimo, 2),
        "ahorro_faltante_usd": round(ahorro_faltante, 2),
        "banco_recomendado": banco,
        "cuota_mensual_ars": cuota,
        "ratio_cuota_ingreso": ratio,
        "resultado": resultado,
    }


# =====================================================
# ASESOR IA (CHAT CON HERRAMIENTA) — igual al del notebook
# =====================================================

def datos_para_la_ia():
    """
    Arma un texto con los datos reales del notebook para pasárselo a Gemini junto con la pregunta.
    Así la IA contesta con nuestros datos y no con cosas que se le ocurran.
    """
    tipo_de_cambio, fecha_dolar = obtener_dolar_oficial()

    texto = f"DÓLAR OFICIAL HOY: $ {tipo_de_cambio}\n\n"

    texto += "BANCOS (relevamiento septiembre/octubre 2026):\n"
    for nombre_banco, cond in BANCOS.items():
        # Cuota mensual por cada u$s 100.000 financiados, calculada por nosotros (no por la IA)
        cuota_100k = calcular_cuota(100000 * tipo_de_cambio, cond["tasa_anual"], cond["plazo_meses"])
        ahorro_pct = 100 - cond["financiacion_maxima"] * 100
        texto += (f"- {nombre_banco}: tasa {cond['tasa_anual']*100:.1f}% anual, "
                  f"plazo {cond['plazo_meses']//12} años, "
                  f"financia hasta el {cond['financiacion_maxima']*100:.0f}% del valor "
                  f"(hay que tener ahorrado al menos el {ahorro_pct:.0f}%), "
                  f"cuota por cada u$s 100.000 financiados: $ {cuota_100k:,.0f}\n")

    texto += "\nPRECIOS PROMEDIO DE DEPARTAMENTOS (USD):\n"
    for zona, tipologias in precios_zona.items():
        texto += f"- {zona}: 1 amb. u$s {tipologias[1]:,}, 2 amb. u$s {tipologias[2]:,}, 3 amb. u$s {tipologias[3]:,}\n"

    texto += f"""
REGLAS DEL SIMULADOR:
- Los bancos piden que la cuota no supere el {LIMITE_CUOTA_INGRESO}% del ingreso mensual.
- Semáforo: VIABLE si la cuota llega hasta el {LIMITE_CUOTA_INGRESO}% del ingreso, AJUSTADO hasta el {LIMITE_CUOTA_INGRESO + 15}%, NO RECOMENDABLE por encima.
- Zonas: Capital (CABA), Zona Norte (GBA norte), Zona Sur (GBA sur). Tipologías: 1, 2 o 3 ambientes.

CONCEPTOS:
- UVA: unidad que se actualiza con la inflación. En un crédito UVA la cuota se calcula en UVA, por eso la cuota en pesos va subiendo con la inflación. La cuota que muestra el simulador es la inicial.
- Ahorro mínimo (anticipo): la parte del valor de la propiedad que el banco no financia y pone la persona.
- Sistema francés: la cuota es la misma durante todo el plazo (medida en UVA).
"""
    return texto


def buscar_opciones(ingreso_mensual_ars: float, ahorro_usd: float, zona: str = "", tipologia: int = 0) -> dict:
    """
    Busca qué departamentos puede comprar una persona con un crédito hipotecario UVA.
    Evalúa cada combinación de zona (Capital, Zona Norte, Zona Sur) y tipología (1, 2 o 3 ambientes)
    y devuelve, para cada una, el banco más conveniente, la cuota estimada y el resultado.

    Args:
        ingreso_mensual_ars: ingreso mensual de la persona, en pesos.
        ahorro_usd: ahorro disponible de la persona, en dólares.
        zona: opcional. "Capital", "Zona Norte" o "Zona Sur". Si va vacío, evalúa todas las zonas.
        tipologia: opcional. 1, 2 o 3 ambientes. Si es 0, evalúa todas las tipologías.
    """
    # Esta es la función que la IA puede "usar": adentro llama a nuestro evaluar_credito()
    tipo_de_cambio, fecha_dolar = obtener_dolar_oficial()

    opciones = []
    for zona_i in precios_zona:
        for tipologia_i in [1, 2, 3]:
            if zona and zona != zona_i:
                continue
            if tipologia and int(tipologia) != tipologia_i:
                continue

            r = evaluar_credito("consulta", 0, zona_i, tipologia_i, ingreso_mensual_ars, ahorro_usd, tipo_de_cambio)
            opciones.append({
                "zona": zona_i,
                "ambientes": tipologia_i,
                "precio_usd": r["precio_propiedad_usd"],
                "resultado": r["resultado"],
                "banco": r["banco_recomendado"],
                "cuota_ars": None if r["cuota_mensual_ars"] is None else round(float(r["cuota_mensual_ars"])),
                "cuota_sobre_ingreso_pct": None if r["ratio_cuota_ingreso"] is None else round(float(r["ratio_cuota_ingreso"]), 1),
                "ahorro_que_falta_usd": r["ahorro_faltante_usd"],
            })

    if len(opciones) == 0:
        return {"error": "No encontré esa zona o tipología. Zonas válidas: Capital, Zona Norte, Zona Sur."}

    # Ordenamos: primero las mejores opciones
    orden = {"VIABLE": 0, "AJUSTADO": 1, "NO RECOMENDABLE": 2, "NO ALCANZA EL AHORRO MÍNIMO": 3}
    opciones = sorted(opciones, key=lambda o: orden[o["resultado"]])
    return {"dolar_usado": tipo_de_cambio, "opciones": opciones}


def responder_asesor(mensaje, historial):
    """
    Responde un mensaje del chat.
    'historial' son los mensajes anteriores (en la app los guarda Streamlit): cada uno es un
    diccionario con "role" (user o assistant) y "content" (el texto).
    La IA puede usar la función buscar_opciones() cuando tiene el ingreso y el ahorro de la persona.
    Si Gemini no responde (sin clave, sin internet, límite gratuito), devuelve los datos en texto.
    """
    # Armamos el texto de la conversación con los últimos mensajes
    conversacion = ""
    for turno in historial[-8:]:
        if isinstance(turno["content"], str):
            rol = "Usuario" if turno["role"] == "user" else "Asesor"
            conversacion += f"{rol}: {turno['content']}\n"

    datos_ia = datos_para_la_ia()

    prompt = f"""
Sos un asesor de créditos hipotecarios UVA en Argentina. Respondé en español rioplatense (vos),
con tono cercano y claro, en no más de 6 líneas.

CÓMO TRABAJAR:
1. Usá SOLO los datos de abajo. Si te piden algo que no está, decí que no lo tenés. No inventes números.
2. Si la persona quiere saber si puede comprar o si le conviene el crédito, necesitás dos datos:
   su ingreso mensual (en pesos) y su ahorro (en dólares). Si falta alguno, preguntáselo; no lo supongas.
   Preguntale también, como algo opcional, si ya tiene en mente una zona (Capital, Zona Norte o Zona Sur)
   o una cantidad de ambientes; si no tiene, buscás en todas.
3. Cuando tengas ingreso y ahorro, usá la herramienta buscar_opciones. NO hagas cuentas vos:
   usá lo que devuelve la herramienta.
4. Con el resultado, decile qué puede comprar: nombrá las mejores opciones (zona, ambientes, banco y cuota).
   Si ninguna le sirve, decile qué le falta (más ahorro o más ingreso) o qué alternativa cercana probar.
5. Esto es una simulación educativa, no una oferta de ningún banco. La cuota es la inicial:
   en un crédito UVA sube con la inflación.

{datos_ia}
CONVERSACIÓN HASTA AHORA:
{conversacion}
MENSAJE NUEVO: {mensaje}
"""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        # Abrimos un chat con Gemini y le damos nuestra función como herramienta
        chat_ia = client.chats.create(
            model=MODELO_IA,
            config=types.GenerateContentConfig(tools=[buscar_opciones]),
        )
        respuesta = chat_ia.send_message(prompt)
        return respuesta.text
    except Exception:
        return "[Asesor IA no disponible en este momento] Estos son los datos que tengo:\n" + datos_ia


# =====================================================
# GUARDADO DE CONSULTAS (CSV)
# =====================================================

def guardar_consulta(resultado):
    """Agrega la consulta como una fila nueva al CSV (lo crea si no existe)."""
    fila = pd.DataFrame([resultado])
    fila["fecha_hora"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fila.to_csv(ARCHIVO_CONSULTAS, mode="a", index=False, header=not os.path.exists(ARCHIVO_CONSULTAS))


# =====================================================
# PÁGINA (STREAMLIT)
# =====================================================

st.title("🏠 Simulador de Crédito Hipotecario UVA")

tab_simulador, tab_chat, tab_panel = st.tabs(["Simulador", "Asesor IA (chat)", "Panel de consultas"])

# ---------- PESTAÑA 1: SIMULADOR ----------
with tab_simulador:
    st.caption("Las consultas se guardan con fines académicos. Podés usar un nombre ficticio.")
    st.write("Completá tus datos para saber si te conviene tomar el crédito hoy y qué banco te conviene.")

    col1, col2 = st.columns(2)
    nombre = col1.text_input("Nombre")
    edad = col2.number_input("Edad", min_value=18, max_value=99, value=30)
    zona = col1.selectbox("Zona", list(precios_zona.keys()))
    tipologia = col2.selectbox("Tipología (ambientes)", [1, 2, 3], index=1)
    ingreso = col1.number_input("Ingreso mensual (ARS)", min_value=0, step=100000)
    ahorro = col2.number_input("Ahorro disponible (USD)", min_value=0, step=1000)

    if st.button("Simular crédito", type="primary"):
        # Validación: si falta algún dato, avisamos y no calculamos
        if not nombre or ingreso <= 0:
            st.warning("⚠️ Completá tu nombre y un ingreso mayor a 0.")
        else:
            tipo_de_cambio, fecha_dolar = obtener_dolar_oficial()   # dólar en vivo en cada consulta

            resultado = evaluar_credito(
                nombre=nombre, edad=int(edad), zona=zona, tipologia=int(tipologia),
                ingreso_mensual_ars=float(ingreso), ahorro_usd=float(ahorro),
                tipo_de_cambio=tipo_de_cambio,
            )
            guardar_consulta(resultado)

            if resultado["resultado"] == "NO ALCANZA EL AHORRO MÍNIMO":
                st.error(f"⛔ {nombre}, te falta ahorro. Necesitás al menos u$s {resultado['ahorro_minimo_usd']:,.0f} "
                         f"y te faltan u$s {resultado['ahorro_faltante_usd']:,.0f}.")
            else:
                texto = (f"{nombre}, tu crédito es **{resultado['resultado']}**.\n\n"
                         f"Banco más conveniente: **{resultado['banco_recomendado']}**\n\n"
                         f"Cuota estimada: **$ {resultado['cuota_mensual_ars']:,.0f}** "
                         f"({resultado['ratio_cuota_ingreso']}% de tu ingreso; los bancos piden {LIMITE_CUOTA_INGRESO}% o menos)")
                if resultado["resultado"] == "VIABLE":
                    st.success(texto)
                elif resultado["resultado"] == "AJUSTADO":
                    st.warning(texto)
                else:
                    st.error(texto)

            st.caption(f"Dólar oficial usado: $ {tipo_de_cambio}")

            st.subheader("Comparativo entre bancos")
            tabla = comparar_bancos(precios_zona[zona][int(tipologia)], float(ahorro), float(ingreso), tipo_de_cambio)
            st.dataframe(tabla)

# ---------- PESTAÑA 2: CHAT CON EL ASESOR ----------
with tab_chat:
    st.write("Preguntale sobre bancos, tasas y precios, o contale cuánto ahorraste y cuánto ganás y te dice qué podés comprar.")

    # st.session_state guarda la conversación mientras la persona usa la página
    if "mensajes" not in st.session_state:
        st.session_state.mensajes = []

    # Preguntas frecuentes: botones que se tocan y listo
    st.caption("Preguntas frecuentes:")
    preguntas_frecuentes = [
        "¿Qué banco tiene la tasa más baja?",
        "¿Qué banco tiene la cuota más baja?",
        "Tengo 25.000 dólares ahorrados, ¿puedo comprarme algo?",
        "¿Cuánto ahorro necesito para un 2 ambientes en Zona Norte?",
        "¿Qué es un crédito UVA y cómo funciona?",
        "¿Qué significa que la cuota no pase el 25% del ingreso?",
    ]

    pregunta = None
    for i, pregunta_frecuente in enumerate(preguntas_frecuentes):
        if st.button(pregunta_frecuente, key=f"pf_{i}"):
            pregunta = pregunta_frecuente

    # Mostramos la conversación hasta ahora
    for mensaje in st.session_state.mensajes:
        with st.chat_message(mensaje["role"]):
            st.write(mensaje["content"])

    # Caja para escribir una pregunta nueva
    escrito = st.chat_input("Escribí tu pregunta...")
    if escrito:
        pregunta = escrito

    if pregunta:
        with st.chat_message("user"):
            st.write(pregunta)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                respuesta = responder_asesor(pregunta, st.session_state.mensajes)
            st.write(respuesta)

        # Guardamos los dos mensajes en el historial
        st.session_state.mensajes.append({"role": "user", "content": pregunta})
        st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

# ---------- PESTAÑA 3: PANEL DE CONSULTAS (solo para mí, con contraseña) ----------
with tab_panel:
    # La contraseña NO va escrita en el código: se guarda como "Secret" en Streamlit Cloud
    try:
        PANEL_PASSWORD = st.secrets["PANEL_PASSWORD"]
    except Exception:
        PANEL_PASSWORD = os.environ.get("PANEL_PASSWORD")

    if not PANEL_PASSWORD:
        st.info("El panel no está configurado todavía (falta el Secret PANEL_PASSWORD).")
    else:
        clave = st.text_input("Contraseña del panel", type="password")

        if clave != PANEL_PASSWORD:
            st.write("Ingresá la contraseña para ver las consultas guardadas.")
        elif not os.path.exists(ARCHIVO_CONSULTAS):
            st.info("Todavía no hay consultas guardadas.")
        else:
            df = pd.read_csv(ARCHIVO_CONSULTAS)

            # Indicadores principales
            col1, col2, col3 = st.columns(3)
            col1.metric("Consultas totales", len(df))
            col2.metric("Edad promedio", f"{df['edad'].mean():.1f} años")
            col3.metric("% VIABLE", f"{(df['resultado'] == 'VIABLE').mean() * 100:.0f}%")

            # Gráficos con value_counts(), igual que en el notebook
            st.subheader("Consultas por zona")
            st.bar_chart(df["zona"].value_counts())

            st.subheader("Resultados")
            st.bar_chart(df["resultado"].value_counts())

            st.subheader("Últimas consultas")
            st.dataframe(df.tail(20))

            # Botón para bajar todo el historial y llevarlo a Excel o Power BI
            st.download_button(
                "Descargar todas las consultas (CSV)",
                data=df.to_csv(index=False),
                file_name="consultas_hipotecario.csv",
                mime="text/csv",
            )
