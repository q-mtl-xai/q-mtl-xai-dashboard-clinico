
import plotly.graph_objects as go
import numpy as np

CLASS_NAMES = {
    "latido": ["normal", "paced", "PVC", "SPB"],
    "ventana": ["normal", "paced", "PVC", "SPB"],
    "isc": ["normal", "ischemia"],
}

def plot_signal(signal, title="Señal ECG"):
    signal = np.asarray(signal)
    fig = go.Figure()
    if signal.ndim == 1:
        fig.add_trace(go.Scatter(y=signal, mode="lines",
                                 line=dict(color="#1f77b4", width=1.4),
                                 name="ECG"))
    else:
        for i in range(signal.shape[0]):
            fig.add_trace(go.Scatter(y=signal[i], mode="lines",
                                     line=dict(width=1),
                                     name=f"Lead {i+1}",
                                     visible=(i == 0)))
        fig.update_layout(updatemenus=[dict(
            type="buttons",
            buttons=[dict(label=f"Lead {i+1}", method="update",
                          args=[{"visible": [j == i for j in range(signal.shape[0])]}])
                     for i in range(signal.shape[0])],
        )])
    fig.update_layout(title=title, xaxis_title="Muestras", yaxis_title="Amplitud",
                      height=350, template="plotly_white",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_signal_with_saliency(signal, saliency, title="ECG + XAI"):
    signal = np.asarray(signal)
    sal = np.asarray(saliency)

    if signal.ndim == 2:
        signal_1d = signal[1] if signal.shape[0] > 1 else signal[0]
    else:
        signal_1d = signal

    if len(sal) != len(signal_1d):
        x_old = np.linspace(0, 1, len(sal))
        x_new = np.linspace(0, 1, len(signal_1d))
        sal = np.interp(x_new, x_old, sal)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal_1d, mode="lines",
                             line=dict(color="#1f77b4", width=1.4),
                             name="ECG"))

    y_min, y_max = float(signal_1d.min()), float(signal_1d.max())
    sal_scaled = y_min + sal * (y_max - y_min) * 0.9
    fig.add_trace(go.Scatter(y=sal_scaled, mode="lines", fill="tozeroy",
                             line=dict(color="rgba(255,100,100,0.3)", width=0),
                             fillcolor="rgba(255,100,100,0.25)",
                             name="Importancia XAI"))
    fig.update_layout(title=title, xaxis_title="Muestras", yaxis_title="Amplitud",
                      height=400, template="plotly_white",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_probabilities(probs, task="isc"):
    classes = CLASS_NAMES.get(task, list(probs.keys()))
    values = [probs.get(c, 0) for c in classes]
    colors = ["#1f77b4" if v == max(values) else "#aec7e8" for v in values]
    fig = go.Figure(data=[go.Bar(x=classes, y=values, marker_color=colors,
                                 text=[f"{v:.3f}" for v in values],
                                 textposition="auto")])
    fig.update_layout(title="Probabilidades por clase",
                      yaxis_title="Probabilidad", yaxis_range=[0, 1],
                      height=300, template="plotly_white",
                      margin=dict(l=20, r=20, t=40, b=20))
    return fig
