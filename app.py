
import os
import io
import json
import glob
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fpdf import FPDF

from api_client import APIClient
from ui_components import plot_signal, plot_signal_with_saliency, plot_probabilities

BACKEND_URL = os.environ.get("BACKEND_URL", "https://q-mtl-xai-backend-clinico.onrender.com")
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
XAI_DIR = os.path.join(ASSETS_DIR, "xai")

# Configuración para bloquear el zoom y el movimiento en los gráficos
PLOTLY_CONFIG = {
    'displayModeBar': False, # Oculta la barra de herramientas superior derecha
    'scrollZoom': False,     # Bloquea el zoom con la rueda del ratón
}

st.set_page_config(page_title="Q-MTL-XAI Clinical", page_icon="🫀", layout="wide")

# ============================================================
# CACHÉ DE LLAMADAS A LA API Y CÁLCULOS PESADOS
# ============================================================
@st.cache_data(ttl=30, show_spinner=False)
def _cached_list_studies(patient_id: str, token: str, _client):
    return _client.list_studies(patient_id)

@st.cache_data(ttl=60, show_spinner=False)
def _cached_get_signal(study_id: str, token: str, _client):
    return _client.get_study_signal(study_id)

@st.cache_data(ttl=300, show_spinner=False)
def _get_historical_vitals_cached(studies_data: tuple, token: str, _client):
    """
    Versión cacheada de _get_historical_vitals. 
    Recibe una tupla hasheable de (id, fecha_str, prediccion) para no re-descargar ni recalcular en cada clic.
    """
    historial = []
    for study_id, fecha_str, prediccion in studies_data:
        try:
            sig_data = _cached_get_signal(study_id, token, _client)
            sig = np.array(sig_data)
            bpm, ritmo = _calculate_bpm(sig)
            if isinstance(bpm, int):
                historial.append({
                    "Fecha": pd.to_datetime(fecha_str).tz_localize(None),
                    "BPM": bpm,
                    "Ritmo": ritmo,
                    "Predicción": prediccion.capitalize(),
                    "ID": study_id
                })
        except Exception:
            pass
    return pd.DataFrame(historial).sort_values("Fecha") if historial else pd.DataFrame()

# ============================================================
# HELPERS
# ============================================================
def _calculate_bpm(signal, fs=250):
    """Calcula el BPM promedio y evalúa el ritmo, ignorando PVCs aislados."""
    if signal is None:
        return None, "—"
    if signal.ndim == 2:
        sig = signal[1] if signal.shape[0] > 1 else signal[0]
    else:
        sig = signal
    sig = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-8)
    peaks, _ = find_peaks(sig, distance=int(0.3 * fs), height=0.4)
    if len(peaks) < 2:
        return None, "—"
    
    rr_intervals = np.diff(peaks) / fs
    
    med_rr = np.median(rr_intervals)
    valid_rr = rr_intervals[(rr_intervals > 0.6 * med_rr) & (rr_intervals < 1.5 * med_rr)]
    
    if len(valid_rr) == 0:
        valid_rr = rr_intervals
        
    mean_rr = np.mean(valid_rr)
    bpm = int(60 / mean_rr)
    std_rr = np.std(valid_rr)
    
    ritmo = "Regular" if std_rr < 0.15 else "Irregular"
    return bpm, ritmo

def _clean_text(text):
    if not text:
        return ""
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U',
        'ñ': 'n', 'Ñ': 'N',
        '📈': '', '📉': '', '⚠️': '', '🧠': '', '✅': '', '🔴': '', '🟡': '', '🔵': '',
        '→': '->', '←': '<-', '•': '-', '*': ''
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode('ascii', 'ignore').decode('ascii')

def _generate_pdf(patient, study, signal, saliency, bpm):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_fill_color(31, 119, 180)
    pdf.rect(0, 0, 210, 30, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", size=18, style='B')
    pdf.cell(0, 15, txt="Q-MTL-XAI - Reporte Clinico", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    pdf.set_font("Helvetica", size=12, style='B')
    pdf.cell(200, 8, txt=_clean_text(f"Paciente: {patient['full_name']}"), ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(200, 6, txt=_clean_text(f"DNI: {patient['dni']} | Fecha: {study['study_date'][:19]}"), ln=True)
    pdf.ln(5)

    pred = study['ai_prediction'].lower()
    if pred == "normal":
        r, g, b = 40, 167, 69
    elif pred in ["ischemia", "pvc", "spb"]:
        r, g, b = 220, 53, 69
    else:
        r, g, b = 255, 193, 7
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", size=14, style='B')
    pdf.cell(0, 12, txt=_clean_text(f"Prediccion: {study['ai_prediction'].upper()} (Confianza: {study['ai_confidence']:.2%})"), ln=True, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    pdf.set_font("Helvetica", size=10, style='B')
    pdf.cell(60, 8, txt="Calidad de Senal (SQI):", border=1, align='C')
    pdf.cell(60, 8, txt="Frecuencia Cardiaca (BPM):", border=1, align='C')
    pdf.cell(60, 8, txt="Validacion Medica:", border=1, align='C', ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(60, 8, txt=_clean_text(study['sqi_score']), border=1, align='C')
    pdf.cell(60, 8, txt=_clean_text(str(bpm) if bpm else "No calculable"), border=1, align='C')
    pdf.cell(60, 8, txt=_clean_text(study.get('medical_validation', 'Pendiente')), border=1, align='C', ln=True)
    pdf.ln(5)

    top_ts = study.get('top_timesteps', [])
    if isinstance(top_ts, str):
        try: top_ts = json.loads(top_ts)
        except: top_ts = []
    if top_ts:
        pdf.set_font("Helvetica", size=9, style='I')
        ts_str = ", ".join([str(t) for t in top_ts[:5]])
        pdf.cell(200, 5, txt=_clean_text(f"Top instantes de atencion XAI (muestras): {ts_str}"), ln=True)
        pdf.ln(3)

    notas = study.get('medical_notes', '')
    if notas:
        pdf.set_font("Helvetica", size=10, style='B')
        pdf.cell(200, 6, txt="Notas Clinicas:", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 6, txt=_clean_text(notas))
        pdf.ln(5)

    sig_1d = signal[1] if hasattr(signal, 'ndim') and signal.ndim == 2 and signal.shape[0] > 1 else signal

    buf1 = io.BytesIO()
    fig1, ax1 = plt.subplots(figsize=(8, 2.5))
    ax1.plot(sig_1d, color='#1f77b4')
    ax1.set_title("Senal ECG Original")
    ax1.set_xlabel("Muestras")
    ax1.set_ylabel("Amplitud")
    ax1.grid(True, alpha=0.3)
    max_val = float(np.max(sig_1d))
    min_val = float(np.min(sig_1d))
    ax1.text(0.01, 0.95, f"Max: {max_val:.2f} | Min: {min_val:.2f}", transform=ax1.transAxes, fontsize=8, verticalalignment='top')
    fig1.tight_layout()
    fig1.savefig(buf1, format='png', dpi=120)
    plt.close(fig1)
    buf1.seek(0)
    pdf.image(buf1, x=15, y=pdf.get_y(), w=180)
    pdf.ln(50)

    if saliency:
        buf2 = io.BytesIO()
        fig2, ax2 = plt.subplots(figsize=(8, 2.5))
        ax2.plot(sig_1d, color='#1f77b4', label="ECG")
        sal_arr = np.array(saliency)
        y_min, y_max = float(np.min(sig_1d)), float(np.max(sig_1d))
        sal_scaled = y_min + sal_arr * (y_max - y_min) * 0.9
        ax2.fill_between(range(len(sig_1d)), y_min, sal_scaled, color='#ff6464', alpha=0.3, label="Importancia XAI")
        max_xai_idx = int(np.argmax(sal_arr))
        ax2.axvline(x=max_xai_idx, color='red', linestyle='--', alpha=0.7)
        ax2.text(max_xai_idx, y_max * 0.85, f' Max Atencion: t={max_xai_idx/250:.2f}s', color='red', fontsize=8, verticalalignment='bottom')
        ax2.set_title("Senal Original + Importancia XAI")
        ax2.set_xlabel("Muestras")
        ax2.set_ylabel("Amplitud")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right")
        fig2.tight_layout()
        fig2.savefig(buf2, format='png', dpi=120)
        plt.close(fig2)
        buf2.seek(0)
        pdf.image(buf2, x=15, y=pdf.get_y(), w=180)
        pdf.ln(50)

    buf3 = io.BytesIO()
    fig3, ax3 = plt.subplots(figsize=(6, 2))
    try:
        clases = ["Normal", "SPB", "Isquemia", "PVC", "Paced"]
        valores = [0.1] * len(clases)
        if pred == "normal": valores[0] = study['ai_confidence']
        elif pred == "spb": valores[1] = study['ai_confidence']
        elif pred == "ischemia": valores[2] = study['ai_confidence']
        elif pred == "pvc": valores[3] = study['ai_confidence']
        elif pred == "paced": valores[4] = study['ai_confidence']
        colors = ['#1f77b4' if v == max(valores) else '#aec7e8' for v in valores]
        bars = ax3.bar(clases, valores, color=colors)
        for bar in bars:
            height = bar.get_height()
            ax3.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        ax3.set_title("Probabilidades por Clase")
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, axis='y', alpha=0.3)
        fig3.tight_layout()
        fig3.savefig(buf3, format='png', dpi=120)
        plt.close(fig3)
        buf3.seek(0)
        pdf.image(buf3, x=35, y=pdf.get_y(), w=140)
    except:
        pass

    if study.get('llm_explanation'):
        pdf.add_page()
        pdf.set_fill_color(240, 242, 246)
        pdf.rect(10, 10, 190, 280, 'F')
        pdf.set_font("Helvetica", size=14, style='B')
        pdf.set_text_color(31, 119, 180)
        pdf.cell(200, 10, txt="Interpretacion Asistida (IA Generativa):", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)
        pdf.set_font("Helvetica", size=10)
        texto_llm = _clean_text(study['llm_explanation'])
        texto_llm = texto_llm.replace(" - ", chr(10) + "- ")
        pdf.multi_cell(0, 6, txt=texto_llm)
        pdf.ln(10)
        pdf.set_font("Helvetica", size=8, style='I')
        pdf.set_text_color(100, 100, 100)
        pdf.multi_cell(0, 5, txt="Este reporte fue generado por un sistema de soporte a la decision clinica. No reemplaza el juicio de un profesional de la salud.")

    return bytes(pdf.output(dest='S'))

def _plot_gauge(prediction, confidence):
    if prediction.lower() == "normal":
        color = "#28a745"
    elif prediction.lower() in ["ischemia", "pvc", "spb"]:
        color = "#dc3545"
    else:
        color = "#ffc107"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': f"Riesgo: {prediction.upper()}", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 80], 'color': '#fff3e0'},
                {'range': [80, 100], 'color': '#e8f5e9'}
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def _get_comparison(patient_id, current_result, client):
    try:
        studies = _cached_list_studies(patient_id, client.token, client)
        if not studies:
            return None
        anterior = studies[0]
        pred_actual = current_result["prediction"].lower()
        conf_actual = current_result["confidence"]
        pred_anterior = anterior["ai_prediction"].lower()
        conf_anterior = anterior["ai_confidence"]
        diff = (conf_actual - conf_anterior) * 100
        if pred_actual == pred_anterior:
            if diff > 0:
                return f"📈 La confianza en **{pred_actual.upper()}** ha **aumentado un {abs(diff):.1f}%** respecto al estudio del {anterior['study_date'][:10]}."
            else:
                return f"📉 La confianza en **{pred_actual.upper()}** ha **disminuido un {abs(diff):.1f}%** respecto al estudio del {anterior['study_date'][:10]}."
        else:
            return f"⚠️ La predicción ha **cambiado** de **{pred_anterior.upper()}** a **{pred_actual.upper()}** respecto al estudio del {anterior['study_date'][:10]}."
    except:
        return None

# ============================================================
# LOGIN
# ============================================================
def show_login():
    st.markdown('<div style="font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;">🫀 Q-MTL-XAI Clinical</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; color: #666; margin-bottom: 2rem;">Sistema clínico con XAI + LLM</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab_login, tab_reg = st.tabs(["🔐 Iniciar sesión", "📝 Registrarse"])
        with tab_login:
            email = st.text_input("Email", key="l_email")
            pwd = st.text_input("Contraseña", type="password", key="l_pwd")
            if st.button("Ingresar", type="primary", use_container_width=True):
                try:
                    data = client.login(email, pwd)
                    st.session_state.logged_in = True
                    st.session_state.doctor = data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        with tab_reg:
            e2 = st.text_input("Email", key="r_email")
            p2 = st.text_input("Contraseña", type="password", key="r_pwd")
            n2 = st.text_input("Nombre completo", key="r_name")
            l2 = st.text_input("Colegiatura (opcional)", key="r_lic")
            if st.button("Registrarme", type="primary", use_container_width=True):
                try:
                    data = client.register(e2, p2, n2, l2 or None)
                    st.session_state.logged_in = True
                    st.session_state.doctor = data
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")

# ============================================================
# FRAGMENT: RESUMEN (Aislado para evitar reruns globales)
# ============================================================
@st.fragment
def render_resumen(patient, client):
    st.markdown("""
    <style>
    .kpi-card-custom {
        background: #ffffff; border-radius: 15px; padding: 20px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05); border: 1px solid #e0e0e0;
        height: 100%; display: flex; flex-direction: column; justify-content: center;
        margin-bottom: 20px;
    }
    .kpi-card-custom .kpi-title { font-size: 0.95rem; color: #666; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
    .kpi-card-custom .kpi-value { font-size: 2.5rem; font-weight: 800; margin: 0; line-height: 1.1; }
    </style>
    """, unsafe_allow_html=True)

    try:
        studies = _cached_list_studies(patient["id"], client.token, client)
        studies_validos = [s for s in studies if s.get("medical_validation") != "Anulado"]
    except Exception as e:
        st.error(f"Error cargando estudios: {e}")
        return

    # ================= FILA 1: KPIs =================
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="kpi-card-custom" style="background-color: #f4f8ff;">
            <div class="kpi-title">Total estudios</div>
            <div class="kpi-value" style="color: #1565c0;">{len(studies_validos)}</div>
        </div>
        """, unsafe_allow_html=True)
    if studies_validos:
        with c2:
            st.markdown(f"""
            <div class="kpi-card-custom" style="background-color: #f4fff4;">
                <div class="kpi-title">Última predicción</div>
                <div class="kpi-value" style="color: #2e7d32;">{studies_validos[0]['ai_prediction'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-card-custom" style="background-color: #fffaf4;">
                <div class="kpi-title">Confianza</div>
                <div class="kpi-value" style="color: #f57f17;">{studies_validos[0]['ai_confidence']:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        with c2:
            st.markdown('<div class="kpi-card-custom" style="background-color: #f4fff4;"><div class="kpi-title">Última predicción</div><div class="kpi-value" style="color: #2e7d32;">—</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="kpi-card-custom" style="background-color: #fffaf4;"><div class="kpi-title">Confianza</div><div class="kpi-value" style="color: #f57f17;">—</div></div>', unsafe_allow_html=True)

    if not studies_validos:
        st.info("Sin estudios válidos registrados.")
        return

    # ================= FILA 2: Columnas =================
    col_izq, col_der = st.columns([1, 1.6])
    
    with col_izq:
        fig_donut = _plot_donut_from_studies(studies_validos)
        if fig_donut: 
            st.plotly_chart(fig_donut, use_container_width=True, key="resumen_donut")
        
        fig_sqi = _plot_sqi_distribution(studies_validos)
        if fig_sqi: 
            st.plotly_chart(fig_sqi, use_container_width=True, key="resumen_sqi")

        st.markdown("---")
        with st.container(key="history-container"):
            st.markdown('<h4 style="text-align: center; text-transform: uppercase; margin-bottom: 15px;">HISTORIAL RECIENTE</h4>', unsafe_allow_html=True)
            df_hist = pd.DataFrame([{
                "Fecha": s["study_date"][:10],
                "Predicción": s["ai_prediction"].upper(),
                "Confianza": f"{s['ai_confidence']:.2%}",
                "SQI": s["sqi_score"],
                "Validación": f"{VAL_COLOR_MAP.get(s.get('medical_validation','Pendiente'), '⚪')} {s.get('medical_validation','Pendiente')}"
            } for s in studies_validos])
            st.dataframe(df_hist, use_container_width=True, height=300)

            fecha_csv = datetime.now().strftime("%Y%m%d_%H%M")
            csv_bytes = df_hist.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar historial (CSV)",
                data=csv_bytes,
                file_name=f"historial_{patient['dni']}_{fecha_csv}.csv",
                mime="text/csv",
                key="dl_historial_csv",
                use_container_width=True,
            )

    with col_der:
        st.markdown("#### 🫀 Evolución de Signos Vitales (BPM)")
        
        # Usamos la tupla cacheada para evitar re-descargas
        studies_tuple = tuple((s["id"], s["study_date"], s["ai_prediction"]) for s in studies_validos)
        with st.spinner("Calculando tendencia de BPM..."):
            df_vitals = _get_historical_vitals_cached(studies_tuple, client.token, client)
            
            if not df_vitals.empty:
                df_vitals = df_vitals.reset_index(drop=True)
                df_vitals["Idx"] = df_vitals.index
                
                fig_bpm = go.Figure()
                fig_bpm.add_trace(go.Scatter(
                    x=df_vitals["Idx"], y=df_vitals["BPM"],
                    mode='lines', name='Tendencia',
                    line=dict(color='rgba(200, 200, 200, 0.6)', width=2, dash='dot'),
                    hoverinfo='skip'
                ))
                
                df_reg = df_vitals[df_vitals["Ritmo"] == "Regular"]
                if not df_reg.empty:
                    fig_bpm.add_trace(go.Scatter(
                        x=df_reg["Idx"], y=df_reg["BPM"],
                        mode='markers', name='Ritmo Regular',
                        marker=dict(size=10, color='#1f77b4', symbol='circle', line=dict(width=1, color='white')),
                        customdata=df_reg[["Fecha", "Predicción"]],
                        hovertemplate="<b>Fecha:</b> %{customdata[0]|%Y-%m-%d %H:%M}<br><b>BPM:</b> %{y}<br><b>Ritmo:</b> Regular<br><b>Predicción:</b> %{customdata[1]}<extra></extra>"
                    ))
                    
                df_irr = df_vitals[df_vitals["Ritmo"] == "Irregular"]
                if not df_irr.empty:
                    fig_bpm.add_trace(go.Scatter(
                        x=df_irr["Idx"], y=df_irr["BPM"],
                        mode='markers', name='Ritmo Irregular',
                        marker=dict(size=12, color='#dc3545', symbol='diamond', line=dict(width=1, color='white')),
                        customdata=df_irr[["Fecha", "Predicción"]],
                        hovertemplate="<b>Fecha:</b> %{customdata[0]|%Y-%m-%d %H:%M}<br><b>BPM:</b> %{y}<br><b>Ritmo:</b> Irregular<br><b>Predicción:</b> %{customdata[1]}<extra></extra>"
                    ))

                fig_bpm.add_hrect(y0=0, y1=60, fillcolor="rgba(255, 193, 7, 0.1)", line_width=0, layer="below")
                fig_bpm.add_hrect(y0=100, y1=200, fillcolor="rgba(220, 53, 69, 0.1)", line_width=0, layer="below")
                
                fig_bpm.update_layout(
                    height=280, margin=dict(l=20, r=20, t=30, b=20),
                    plot_bgcolor='#ffffff', paper_bgcolor='#ffffff',
                    xaxis=dict(title="", showgrid=False, showticklabels=False, fixedrange=True),
                    yaxis=dict(title="BPM", showgrid=True, gridcolor='#f0f0f0', range=[20, 180], fixedrange=True),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                    dragmode=False
                )
                st.plotly_chart(fig_bpm, use_container_width=True, key="resumen_bpm", config=PLOTLY_CONFIG)
                st.caption("Círculo Azul: Ritmo Regular | Rombo Rojo: Ritmo Irregular.")
            else:
                st.info("No hay suficientes señales para calcular la tendencia de BPM.")

        st.markdown("---")
        
        df_conf = pd.DataFrame([{
            "Fecha": pd.to_datetime(s["study_date"]).tz_localize(None),
            "Predicción": s["ai_prediction"].capitalize(),
            "Confianza": s["ai_confidence"] * 100,
            "SQI": s["sqi_score"]
        } for s in studies_validos]).sort_values("Fecha")

        if not df_conf.empty:
            st.markdown("#### 📈 Evolución de la Confianza por Tipo")
            for pred in df_conf["Predicción"].unique():
                df_pred = df_conf[df_conf["Predicción"] == pred].reset_index(drop=True)
                df_pred["Idx"] = df_pred.index
                color = COLOR_MAP.get(pred.lower(), "#1f77b4")
                
                with st.container(border=True):
                    fig_c = go.Figure()
                    fig_c.add_trace(go.Scatter(
                        x=df_pred["Idx"], y=df_pred["Confianza"],
                        mode='lines+markers',
                        line=dict(color=color),
                        marker=dict(color=color),
                        customdata=df_pred[["Fecha", "SQI"]],
                        hovertemplate="<b>Fecha:</b> %{customdata[0]|%Y-%m-%d %H:%M}<br><b>Confianza:</b> %{y:.1f}%<br><b>SQI:</b> %{customdata[1]}<extra></extra>"
                    ))
                    fig_c.update_layout(
                        height=180, margin=dict(l=10, r=10, t=30, b=10),
                        title=dict(text=f"<b>{pred}</b>", font=dict(size=14, color=color), x=0.5, xanchor='center'),
                        dragmode=False,
                        xaxis=dict(title="", showgrid=False, showticklabels=False, fixedrange=True),
                        yaxis=dict(range=[0, 105], title=None, fixedrange=True)
                    )
                    st.plotly_chart(fig_c, use_container_width=True, key=f"evol_{pred}", config=PLOTLY_CONFIG)

# ============================================================
# FRAGMENT: HISTORIAL (Ya estaba aislado, se mantiene)
# ============================================================
@st.fragment
def render_historial(patient, client):
    try:
        studies = _cached_list_studies(patient["id"], client.token, client)
    except Exception:
        studies = []
    if not studies:
        st.info("Sin estudios registrados para este paciente.")
        return
    if "filtros_version" not in st.session_state:
        st.session_state["filtros_version"] = 0
    v = st.session_state["filtros_version"]
    fechas = [pd.to_datetime(s["study_date"]).date() for s in studies]
    min_fecha, max_fecha = min(fechas), max(fechas)
    tipos = list(set([s["ai_prediction"] for s in studies]))
    estados = list(set([s.get("medical_validation", "Pendiente") for s in studies]))
    sqis = list(set([s.get("sqi_score", "Fair") for s in studies]))
    with st.container(key="filter-container"):
        with st.form("filtros_historial", clear_on_submit=False):
            st.markdown("#### 🔍 Filtrar estudios")
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            with col_f1:
                fecha_inicio = st.date_input("Desde", min_value=min_fecha, value=min_fecha, key=f"f_inicio_{v}")
                fecha_fin = st.date_input("Hasta", min_value=min_fecha, value=max_fecha, key=f"f_fin_{v}")
            with col_f2:
                filtro_tipos = st.multiselect("Predicción", tipos, default=tipos, key=f"f_tipos_{v}")
            with col_f3:
                filtro_estados = st.multiselect("Estado", estados, default=estados, key=f"f_estados_{v}")
            with col_f4:
                filtro_sqi = st.multiselect("SQI", sqis, default=sqis, key=f"f_sqi_{v}")
            mostrar_anulados = st.checkbox("Mostrar estudios anulados", key=f"f_anulados_{v}")
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                st.form_submit_button("🔍 Aplicar filtros", use_container_width=True)
            with col_btn2:
                if st.form_submit_button("🧹 Limpiar filtros", use_container_width=True):
                    st.session_state["filtros_version"] += 1
                    st.rerun(scope="fragment")
    studies_filtrados = []
    for s in studies:
        fecha_s = pd.to_datetime(s["study_date"]).date()
        if fecha_s < fecha_inicio or fecha_s > fecha_fin: continue
        if s["ai_prediction"] not in filtro_tipos: continue
        if s.get("medical_validation", "Pendiente") not in filtro_estados: continue
        if s.get("sqi_score", "Fair") not in filtro_sqi: continue
        if not mostrar_anulados and s.get("medical_validation") == "Anulado": continue
        studies_filtrados.append(s)
    if not studies_filtrados:
        st.info("No hay estudios que coincidan con los filtros seleccionados.")
        return
    for s in studies_filtrados:
        val = s.get("medical_validation", "Pendiente")
        pred = s['ai_prediction'].lower()
        color_emoji = VAL_COLOR_MAP.get(val, "⚪")
        border_color_pred = COLOR_MAP.get(pred, "#ccc")
        bg_color_pred = BG_COLOR_MAP.get(pred, "#f5f5f5")
        header = f"{color_emoji} {s['study_date'][:19]} — {s['ai_prediction']} ({s['ai_confidence']:.2%}) — SQI: {s['sqi_score']}"
        st.markdown(f"""
        <div class="study-card-header" style="background-color: {bg_color_pred}; border-left-color: {border_color_pred};">
            {header}
        </div>
        """, unsafe_allow_html=True)
        with st.expander("Ver detalles del estudio", expanded=False):
            load_key = f"loaded_{s['id']}"
            if not st.session_state.get(load_key):
                if st.button("📥 Cargar señal y análisis XAI", key=f"load_btn_{s['id']}", use_container_width=True):
                    st.session_state[load_key] = True
                    st.rerun(scope="fragment")
                st.caption("Haz clic para cargar la señal y el análisis. Esto evita cargar todas las gráficas de golpe.")
                continue
            col_del, _ = st.columns([1, 4])
            with col_del:
                with st.popover("🗑️ Anular"):
                    st.write("¿Por qué anulas este estudio?")
                    motivo = st.selectbox("Motivo", ["Paciente equivocado", "Archivo corrupto", "Error de carga", "Otro"], key=f"mot_{s['id']}")
                    detalle = st.text_input("Detalles (opcional)", key=f"det_{s['id']}")
                    if st.button("Confirmar Anulación", type="primary", key=f"btn_void_{s['id']}"):
                        reason_full = f"{motivo}: {detalle}" if detalle else motivo
                        client.void_study(s['id'], reason_full)
                        st.session_state.pop(load_key, None)
                        st.rerun()
            st.write(f"**ID:** {s['id']}")
            st.write(f"**Notas actuales:** {s.get('medical_notes') or '—'}")
            sig_array = None
            sal = None
            if s.get("signal_file_path"):
                try:
                    with st.spinner("Cargando señal original..."):
                        signal_data = _cached_get_signal(s['id'], client.token, client)
                        sig_array = np.array(signal_data)
                        if s.get("saliency_data"):
                            sal = s["saliency_data"]
                            if isinstance(sal, str): sal = json.loads(sal)
                            tab_cruda, tab_xai = st.tabs(["📈 Señal Original", "🧠 Señal + XAI"])
                            with tab_cruda:
                                st.plotly_chart(plot_signal(sig_array, "Señal ECG Original"), use_container_width=True, key=f"signal_raw_{s['id']}")
                            with tab_xai:
                                st.plotly_chart(plot_signal_with_saliency(sig_array, sal, "Señal Original + Importancia XAI"), use_container_width=True, key=f"signal_xai_{s['id']}")
                        else:
                            st.plotly_chart(plot_signal(sig_array, "Señal Original"), use_container_width=True, key=f"signal_raw_{s['id']}")
                except Exception as e:
                    st.warning(f"No se pudo cargar la gráfica de la señal original: {e}")
                    if s.get("saliency_data"):
                        sal = s["saliency_data"]
                        if isinstance(sal, str): sal = json.loads(sal)
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=sal, mode="lines", fill="tozeroy", line=dict(color="#ff6464", width=1)))
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), plot_bgcolor='#fcf4f4', paper_bgcolor='#fcf4f4')
                        st.plotly_chart(fig, use_container_width=True, key=f"sal_{s['id']}")
            if s.get("llm_explanation"):
                st.markdown("### 🧠 Interpretación asistida")
                st.markdown(s["llm_explanation"])
            else:
                if st.button(f"🧠 Pedir interpretación", key=f"exp_{s['id']}"):
                    client.explain_saved(s['id'])
                    st.rerun(scope="fragment")
            st.markdown("---")
            col_pdf, _ = st.columns([2, 3])
            with col_pdf:
                if st.button("📥 Generar Reporte PDF", key=f"gen_pdf_{s['id']}"):
                    if sig_array is not None:
                        with st.spinner("Generando PDF..."):
                            bpm_str, _ = _calculate_bpm(sig_array)
                            pdf_bytes = _generate_pdf(patient, s, sig_array, sal, bpm_str)
                            fecha_pdf = datetime.now().strftime("%Y%m%d_%H%M")
                            st.download_button(
                                label="⬇️ Descargar Reporte",
                                data=pdf_bytes,
                                file_name=f"reporte_{patient['dni']}_{s['id'][:8]}_{fecha_pdf}.pdf",
                                mime="application/pdf",
                                key=f"dl_pdf_{s['id']}"
                            )
                    else:
                        st.error("No se puede generar el PDF porque la señal no está cargada.")
            st.markdown("---")
            st.markdown("**Editar validación médica:**")
            edit_val = st.radio("Estado", ["Pendiente", "Confirmado", "Dudoso", "Descartado"],
                                index=["Pendiente", "Confirmado", "Dudoso", "Descartado"].index(val) if val in ["Pendiente", "Confirmado", "Dudoso", "Descartado"] else 0,
                                horizontal=True, key=f"val_{s['id']}")
            edit_notes = st.text_area("Notas clínicas", value=s.get('medical_notes') or "", key=f"notes_{s['id']}")
            if st.button("💾 Guardar cambios", key=f"save_{s['id']}"):
                client.update_validation(s['id'], edit_val, edit_notes)
                st.success("Cambios guardados")
                st.rerun(scope="fragment")

# ============================================================
# NUEVO ECG
# ============================================================
def show_nuevo_ecg(patient, client):
    st.write("**Subir nuevo ECG**")
    uploaded = st.file_uploader("Archivo .npy", type=["npy"], key="new_upload")
    if uploaded is not None:
        try:
            sig = np.load(uploaded, allow_pickle=True)
            detected_task = None
            if sig.shape == (12, 2500) or sig.shape == (30000,): detected_task = "isc"
            elif sig.shape == (250,): detected_task = "latido"
            elif sig.shape == (750,): detected_task = "ventana"
            if detected_task is None:
                st.error(f"❌ Formato no reconocido: {sig.shape}. Se espera (250,), (750,) o (12, 2500).")
            else:
                task_names = {"latido": "LATIDO", "ventana": "VENTANA", "isc": "ISQUEMIA"}
                st.markdown(f"""
                <div class="task-card">
                    <h3 style="margin: 0; color: #28a745;">✅ Tarea detectada: {task_names[detected_task]}</h3>
                    <p style="margin: 0; color: #555;">Dimensiones: {sig.shape}</p>
                </div>
                """, unsafe_allow_html=True)
                st.plotly_chart(plot_signal(sig, "Señal cargada"), use_container_width=True, key="preview_uploaded_signal")
                if st.button("🔬 Analizar ECG (Vista Previa)", type="primary", key="btn_analyze"):
                    with st.spinner("Analizando…"):
                        try:
                            result = client.predict_only(sig, task=detected_task)
                            st.session_state["preview_result"] = result
                            st.session_state["preview_sig"] = sig
                            st.session_state["preview_task"] = detected_task
                        except Exception as e:
                            st.error(f"Error al analizar: {e}")
        except Exception as e:
            st.error(f"Error cargando archivo: {e}")
    result = st.session_state.get("preview_result")
    if result:
        st.markdown("---")
        st.subheader("🔬 Resultado del Análisis")
        col_gauge, col_vitals = st.columns([1, 1])
        with col_gauge:
            fig_gauge = _plot_gauge(result["prediction"], result["confidence"])
            st.plotly_chart(fig_gauge, use_container_width=True, key="prev_gauge")
        with col_vitals:
            st.markdown("#### 🫀 Signos Vitales Derivados")
            bpm_str, ritmo = _calculate_bpm(st.session_state.get("preview_sig"))
            if result["sqi"] == "Poor":
                st.warning("⚠️ **BPM no calculable** por baja calidad de señal (SQI: Poor).")
            else:
                st.metric(label="Frecuencia Cardíaca Estimada", value=f"{bpm_str} BPM")
                st.metric(label="Ritmo", value=ritmo)
            st.metric(label="Calidad de Señal (SQI)", value=result["sqi"])
        comp_msg = _get_comparison(patient["id"], result, client)
        if comp_msg:
            st.info(comp_msg)
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicción", result["prediction"])
        c2.metric("Confianza", f"{result['confidence']:.2%}")
        c3.metric("SQI", result["sqi"])
        last_sig = st.session_state.get("preview_sig")
        last_task = st.session_state.get("preview_task", "isc")
        if last_sig is not None:
            display_sig = last_sig[1] if last_sig.ndim == 2 and last_sig.shape[0] > 1 else (last_sig[0] if last_sig.ndim == 2 else last_sig)
            st.plotly_chart(plot_signal_with_saliency(display_sig, result["saliency"]), use_container_width=True, key="prev_sal")
        st.plotly_chart(plot_probabilities(result["probabilities"], task=last_task), use_container_width=True, key="prev_probs")
        if st.button("🧠 Pedir interpretación al asistente (borrador)", key="btn_prev_explain"):
            with st.spinner("Consultando…"):
                exp = client.explain(result["prediction"], result["confidence"], result["sqi"], result["top_timesteps"], task=last_task)
                st.session_state["preview_explanation"] = exp
        explanation = st.session_state.get("preview_explanation")
        if explanation:
            st.markdown("### 🧠 Interpretación asistida (Borrador)")
            st.markdown(explanation["explicacion"])
        st.markdown("---")
        st.markdown("### 📝 Cierre Clínico y Guardado")
        col_val, col_notes = st.columns([1, 2])
        with col_val:
            final_val = st.radio("Validación", ["Pendiente", "Confirmado", "Dudoso", "Descartado"], index=0)
        with col_notes:
            final_notes = st.text_area("Notas clínicas (opcional)")
        if st.button("💾 Guardar en Historial del Paciente", type="primary", use_container_width=True):
            with st.spinner("Guardando…"):
                try:
                    saved = client.save_analyzed_study(patient["id"], last_sig, last_task, final_val, final_notes)
                    st.success("✅ Estudio guardado exitosamente")
                    if explanation:
                        client.explain_saved(saved["study_id"])
                    for k in ["preview_result", "preview_sig", "preview_task", "preview_explanation"]:
                        st.session_state.pop(k, None)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error al guardar: {e}")

# ============================================================
# MODO CLÍNICO (Layout de 2 Columnas con fragments)
# ============================================================
def show_clinical():
    st.header("🩺 Modo Clínico")
    st.subheader("🔍 Buscar paciente")
    col1, col2 = st.columns([3, 1])
    with col1:
        dni = st.text_input("DNI del paciente", key="search_dni")
    with col2:
        st.write(""); st.write("")
        buscar = st.button("Buscar", use_container_width=True)
    if buscar and dni:
        try:
            p = client.search_patient(dni)
            st.session_state.current_patient = p
            if not p:
                st.warning(f"No existe paciente con DNI {dni}")
        except Exception as e:
            st.error(f"Error: {e}")
    if st.session_state.get("current_patient") is None and dni:
        with st.expander("➕ Registrar nuevo paciente", expanded=True):
            r_dni = st.text_input("DNI", value=dni, key="new_dni")
            r_nm = st.text_input("Nombre completo", key="new_nm")
            r_bt = st.text_input("Fecha nacimiento (YYYY-MM-DD)", key="new_bt")
            r_gn = st.selectbox("Sexo", ["M", "F", "Otro"], key="new_gn")
            if st.button("Registrar paciente", type="primary"):
                try:
                    p = client.create_patient(r_dni, r_nm, r_bt or None, r_gn)
                    st.session_state.current_patient = p
                    st.success(f"✅ Paciente {p['full_name']} registrado")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
    patient = st.session_state.get("current_patient")
    if not patient:
        return
    
    st.markdown(f"""
    <div class="patient-header">
        <h2 style="margin: 0; color: #0d5a8a;">👤 {patient['full_name']}</h2>
        <p style="margin: 0; color: #555;">DNI: {patient['dni']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab_resumen, tab_historial, tab_nuevo = st.tabs(["📊 Resumen", "📅 Historial", "➕ Nuevo ECG"])
    
    with tab_resumen:
        render_resumen(patient, client)
    with tab_historial:
        render_historial(patient, client)
    with tab_nuevo:
        show_nuevo_ecg(patient, client)

# ============================================================
# MODO EDUCATIVO
# ============================================================
def show_educational():
    st.header("🎓 Modo Educativo")
    tab_info, tab_gal, tab_sb = st.tabs(["ℹ️ Acerca del modelo", "🖼️ Galería XAI", "🧪 Sandbox"])
    with tab_info:
        st.markdown("""
        - **Arquitectura**: MultiTaskEncoder (CNN 1D + BiLSTM + Attention)
        - **Parámetros**: 725,932
        - **Tasks**: latido (4), ventana (4), isquemia (2)
        - **SQI**: RandomForest sobre 8 features
        """)
        mpath = os.path.join(ASSETS_DIR, "encoder_metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as f:
                m = json.load(f)
            for head, val in m.items():
                st.write(f"- **{head}**: F1 macro = {val.get('f1_macro', 0):.4f}")
    with tab_gal:
        dpath = os.path.join(ASSETS_DIR, "xai_descriptions.json")
        desc = {}
        if os.path.exists(dpath):
            with open(dpath) as f:
                desc = json.load(f)
        if os.path.exists(XAI_DIR):
            for img in sorted(glob.glob(os.path.join(XAI_DIR, "*.png"))):
                fname = os.path.basename(img)
                st.markdown(f"**{fname.replace('.png','').replace('_',' ').title()}**")
                st.image(img, use_column_width=True)
                st.caption(desc.get(fname, ""))
                st.markdown("---")
    with tab_sb:
        st.subheader("Sandbox — Análisis libre para aprendizaje")
        st.info("Sube cualquier archivo ECG. El sistema detectará automáticamente su tipo. **No se guarda nada en la base de datos.**")
        uploaded = st.file_uploader("Archivo .npy", type=["npy"], key="sb_up")
        if uploaded is not None:
            try:
                sig = np.load(uploaded, allow_pickle=True)
                detected_task = None
                if sig.shape == (12, 2500) or sig.shape == (30000,): detected_task = "isc"
                elif sig.shape == (250,): detected_task = "latido"
                elif sig.shape == (750,): detected_task = "ventana"
                if detected_task is None:
                    st.error(f"❌ Formato no reconocido: {sig.shape}. Se espera (250,), (750,) o (12, 2500).")
                else:
                    task_names = {"latido": "LATIDO", "ventana": "VENTANA", "isc": "ISQUEMIA"}
                    st.markdown(f"""
                    <div class="task-card" style="border-left-color: #17a2b8;">
                        <h3 style="margin: 0; color: #17a2b8;">🔍 Tarea detectada: {task_names[detected_task]}</h3>
                        <p style="margin: 0; color: #555;">Dimensiones: {sig.shape}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.plotly_chart(plot_signal(sig, "Señal cargada"), use_container_width=True, key="sb_uploaded_signal")
                    if st.button("🔬 Analizar ECG (sin guardar)", type="primary", key="sb_btn"):
                        with st.spinner("Analizando…"):
                            try:
                                res = client.predict_only(sig, task=detected_task)
                                st.session_state["sb_result"] = res
                                st.session_state["sb_sig"] = sig
                                st.session_state["sb_task"] = detected_task
                            except Exception as e:
                                st.error(f"Error al analizar: {e}")
            except Exception as e:
                st.error(f"Error cargando archivo: {e}")
        res = st.session_state.get("sb_result")
        if res:
            st.markdown("---")
            st.subheader("🔬 Resultado del Análisis")
            pred = res["prediction"].lower()
            conf = res["confidence"]
            if pred == "normal" and conf >= 0.65:
                badge_color = "#28a745"; badge_text = "✅ Sin hallazgos relevantes"
            elif pred == "normal" and conf < 0.65:
                badge_color = "#ffc107"; badge_text = "🟡 Confianza baja. Revisar con cautela."
            else:
                badge_color = "#dc3545"; badge_text = f"🔴 Hallazgo detectado: {pred.upper()}"
            st.markdown(f"""
            <div style="background-color: {badge_color}20; border: 1px solid {badge_color}; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin: 0; color: {badge_color}; text-align: center;">{badge_text}</h3>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns([1, 1])
            with c1:
                fig_gauge = _plot_gauge(res["prediction"], res["confidence"])
                st.plotly_chart(fig_gauge, use_container_width=True, key="sb_gauge")
            with c2:
                st.markdown("#### 🫀 Signos Vitales Derivados")
                bpm_str, ritmo = _calculate_bpm(st.session_state.get("sb_sig"))
                if res["sqi"] == "Poor":
                    st.warning("⚠️ **BPM no calculable** (SQI: Poor).")
                else:
                    st.metric(label="Frecuencia Cardíaca Estimada", value=f"{bpm_str} BPM")
                    st.metric(label="Ritmo", value=ritmo)
                st.metric(label="Calidad de Señal (SQI)", value=res["sqi"])
            sig_actual = st.session_state.get("sb_sig")
            task_actual = st.session_state.get("sb_task", "isc")
            if sig_actual is not None:
                disp = sig_actual[1] if sig_actual.ndim == 2 and sig_actual.shape[0] > 1 else (sig_actual[0] if sig_actual.ndim == 2 else sig_actual)
                st.plotly_chart(plot_signal_with_saliency(disp, res["saliency"], "Señal + Importancia XAI"), use_container_width=True, key="sb_saliency_chart")
                st.caption("💡 **¿Cómo leer esto?** El área roja resalta los instantes del ECG que más influyeron en la decisión del modelo.")
            st.plotly_chart(plot_probabilities(res["probabilities"], task=task_actual), use_container_width=True, key="sb_probs_chart")
            st.caption("💡 **¿Cómo leer esto?** Muestra la probabilidad que el modelo asigna a cada clase. La barra más alta es la que el modelo elige como respuesta final.")
            if st.button("🧠 Pedir interpretación al asistente", key="sb_explain"):
                with st.spinner("Consultando…"):
                    exp = client.explain(res["prediction"], res["confidence"], res["sqi"], res["top_timesteps"], task=task_actual)
                    st.markdown("### 🧠 Interpretación asistida")
                    st.markdown(exp["explicacion"])
                    st.caption(f"Fuente: {exp['source']}")

# ============================================================
# ROUTER + CONFIG GLOBAL
# ============================================================
if "client" not in st.session_state:
    st.session_state.client = APIClient(BACKEND_URL)
    st.session_state.logged_in = False
    st.session_state.doctor = None
client = st.session_state.client

COLOR_MAP = {"normal": "#28a745", "spb": "#8A2BE2", "ischemia": "#dc3545", "pvc": "#fd7e14", "paced": "#1f77b4"}
BG_COLOR_MAP = {"normal": "#e8f5e9", "ischemia": "#ffebee", "spb": "#f3e5f5", "pvc": "#fff3e0", "paced": "#e3f2fd"}
SQI_COLOR_MAP = {"Good": "#28a745", "Fair": "#ffc107", "Poor": "#dc3545"}
VAL_COLOR_MAP = {"Confirmado": "🟢", "Dudoso": "🟡", "Descartado": "🔴", "Pendiente": "🔵", "Anulado": "⚫"}

def _plot_donut_from_studies(studies):
    if not studies: return None
    from collections import Counter
    preds = [s["ai_prediction"] for s in studies]
    counts = Counter(preds)
    labels = list(counts.keys()); values = list(counts.values())
    colors = [COLOR_MAP.get(label.lower(), "#aec7e8") for label in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.55, marker=dict(colors=colors))])
    fig.update_layout(title=dict(text="DISTRIBUCIÓN DE CLASES", x=0.5, xanchor='center'), height=320,
                      margin=dict(l=20, r=20, t=70, b=20), plot_bgcolor='#f4f9f4', paper_bgcolor='#f4f9f4')
    return fig

def _plot_sqi_distribution(studies):
    if not studies: return None
    sqi_counts = pd.Series([s.get("sqi_score", "Fair") for s in studies]).value_counts()
    labels = list(sqi_counts.index); values = list(sqi_counts.values)
    colors = [SQI_COLOR_MAP.get(label, "#aec7e8") for label in labels]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.55, marker=dict(colors=colors))])
    fig.update_layout(title=dict(text="CALIDAD DE SEÑAL (SQI)", x=0.5, xanchor='center'), height=320,
                      margin=dict(l=20, r=20, t=70, b=20), plot_bgcolor='#f9f4f4', paper_bgcolor='#f9f4f4')
    return fig

if not st.session_state.logged_in:
    show_login()
else:
    doctor = st.session_state.doctor
    with st.sidebar:
        st.markdown(f"**👨‍⚕️ Dr. {doctor.get('full_name', '')}**")
        st.caption(f"ID: {doctor.get('doctor_id', '')[:8]}…")
        st.markdown("---")
        mode = st.radio("Modo", ["🩺 Modo Clínico", "🎓 Modo Educativo"])
        st.markdown("---")
        if st.button("🚪 Cerrar sesión", use_container_width=True):
            for k in ["logged_in", "doctor", "current_patient", "last_result",
                      "last_sig", "last_task", "pending_signal", "pending_task",
                      "preview_result", "preview_sig", "preview_task", "preview_explanation",
                      "sb_result", "sb_sig", "sb_task"]:
                st.session_state.pop(k, None)
            st.rerun()
    if mode == "🩺 Modo Clínico":
        show_clinical()
    else:
        show_educational()
    st.markdown("---")
    st.caption(f"Q-MTL-XAI v3.5 | Backend: {BACKEND_URL}")
