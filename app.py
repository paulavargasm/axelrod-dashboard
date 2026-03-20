# ============================================================
#  Axelrod & Hamilton (1981) — Dashboard Interactivo
#  Técnicas Computacionales Avanzadas · Dash / Plotly
# ============================================================

import json, io, base64, zlib, random, time
import numpy as np
import pandas as pd
from scipy import stats

import dash
from dash import dcc, html, Input, Output, State, ctx, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Load pre-calculated data ──────────────────────────────
with open("dashboard_data.json") as f:
    DATA = json.load(f)

df_classic  = pd.DataFrame(DATA["classic"]["detailed"])
df_real     = pd.DataFrame(DATA["real"]["detailed"])
mat_classic = pd.DataFrame(DATA["classic"]["matrix"])
mat_real    = pd.DataFrame(DATA["real"]["matrix"])
rank_classic= pd.Series(DATA["classic"]["ranking"]).sort_values(ascending=False)
rank_real   = pd.Series(DATA["real"]["ranking"]).sort_values(ascending=False)
STRATS      = list(rank_classic.index)

# ── Color palette (colorblind-friendly — Wong 2011) ───────
CB = ["#0072B2","#E69F00","#009E73","#CC79A7",
      "#56B4E9","#D55E00","#F0E442","#000000",
      "#44AA99","#117733","#332288","#AA4499",
      "#88CCEE","#DDCC77","#999933"]
STRAT_COLOR = {s: CB[i % len(CB)] for i, s in enumerate(STRATS)}

DARK_BG   = "#0f1117"
CARD_BG   = "#1a1d27"
BORDER    = "#2e3248"
TEXT_PRI  = "#e8eaf6"
TEXT_SEC  = "#9fa8da"
ACCENT    = "#7986cb"
ACCENT2   = "#80cbc4"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=TEXT_PRI, size=12),
    margin=dict(t=40, b=40, l=50, r=20),
    xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
)

# ── Strategy descriptions ──────────────────────────────────
STRAT_INFO = {
    "TitForTat":       {"emoji":"🤝","short":"Coopera primero, luego imita al oponente.","prop":"Nice · Retaliatory · Forgiving · Clear","code":"return 'C' if not history else opponent[-1]"},
    "GrimTrigger":     {"emoji":"💀","short":"Coopera hasta el primer engaño; luego defecta para siempre.","prop":"Nice · Retaliatory · Unforgiving","code":"if 'D' in opponent: triggered=True\nreturn 'D' if triggered else 'C'"},
    "Pavlov":          {"emoji":"🔔","short":"Win-Stay, Lose-Shift: repite si ganó, cambia si perdió.","prop":"Nice · Self-correcting","code":"return last_move if payoff in (R,T) else switch"},
    "AlwaysDefect":    {"emoji":"😈","short":"Traiciona en cada ronda sin importar nada.","prop":"Not nice · Exploitative","code":"return 'D'"},
    "AlwaysCooperate": {"emoji":"😇","short":"Coopera en cada ronda sin importar nada.","prop":"Nice · Exploitable","code":"return 'C'"},
    "TwoTitsForTat":   {"emoji":"🐢","short":"Solo castiga si el oponente traiciona dos veces seguidas.","prop":"Nice · Lenient · Forgiving","code":"return 'D' if opp[-1]=='D' and opp[-2]=='D' else 'C'"},
    "Random":          {"emoji":"🎲","short":"Coopera o traiciona con probabilidad 50/50.","prop":"Stochastic · Unpredictable","code":"return 'C' if random() < 0.5 else 'D'"},
    "Joss":            {"emoji":"🎭","short":"TFT pero con 10% de traición aleatoria.","prop":"Near-nice · Subtly deceptive","code":"if random() < 0.1: return 'D'\nreturn opponent[-1]"},
    "Gradual":         {"emoji":"📈","short":"Castiga con traiciones escalantes y luego se reconcilia.","prop":"Nice · Escalating · Forgiving","code":"punish_n_times(defects_seen)\nthen cooperate×2"},
    "Adaptive":        {"emoji":"🧠","short":"Prior bayesiano sobre cooperación del oponente; maximiza payoff esperado.","prop":"Bayesian · Rational","code":"p = alpha/(alpha+beta)\nreturn 'C' if E[C] >= E[D] else 'D'"},
    "EvolvedNN":       {"emoji":"🤖","short":"Red neuronal con pesos fijos entrenados; decide basándose en 5 features.","prop":"Modern · Feature-based","code":"h = tanh(W1 @ x + b1)\ny = sigmoid(W2 @ h + b2)\nreturn 'C' if y >= 0.5"},
    "PSOPlayer":       {"emoji":"🐦","short":"Enjambre de partículas que estima la mejor probabilidad de cooperar.","prop":"Modern · Adaptive · Swarm","code":"maximize E[payoff] via PSO\nreturn 'C' if best_pos > rand()"},
    "Memory3":         {"emoji":"🧩","short":"Coopera si el oponente cooperó en ≥2 de las últimas 3 rondas.","prop":"Moderate memory · Forgiving","code":"return 'C' if opp[-3:].count('C') >= 2"},
    "Friedman":        {"emoji":"⚖️","short":"Idéntico a GrimTrigger — castigo permanente ante la primera traición.","prop":"Nice · Fully unforgiving","code":"if 'D' in opponent: forever_D=True"},
    "Tester":          {"emoji":"🔬","short":"Empieza traicionando; si hay represalia cambia a TFT, si no explota.","prop":"Probing · Adaptive","code":"D, C, C... si retaliation → TFT\nelse → exploit"},
}

# ── Helpers ───────────────────────────────────────────────
def card(children, style=None):
    base = dict(background=CARD_BG, border=f"1px solid {BORDER}",
                borderRadius="10px", padding="20px", marginBottom="16px")
    if style: base.update(style)
    return html.Div(children, style=base)

def section_title(txt, sub=None):
    return html.Div([
        html.H4(txt, style=dict(color=TEXT_PRI, fontWeight=600, marginBottom="2px")),
        html.P(sub, style=dict(color=TEXT_SEC, fontSize="13px", marginTop=0)) if sub else None
    ], style=dict(marginBottom="16px"))

def badge(txt, color=ACCENT):
    return html.Span(txt, style=dict(
        background=color+"22", color=color, border=f"1px solid {color}44",
        borderRadius="20px", padding="2px 10px", fontSize="11px", fontWeight=600,
        marginRight="6px", display="inline-block"))

def tooltip_icon(tip_id, text):
    return html.Span([
        html.Span("?", id=tip_id, style=dict(
            display="inline-flex", alignItems="center", justifyContent="center",
            width="16px", height="16px", borderRadius="50%", fontSize="10px",
            background=BORDER, color=TEXT_SEC, cursor="help", marginLeft="6px")),
        dbc.Tooltip(text, target=tip_id, placement="top")
    ])

# ── Live tournament simulation ────────────────────────────
class ReproducibleRNG:
    def __init__(self, seed):
        self.gen = np.random.Generator(np.random.PCG64(seed))
    def random(self): return float(self.gen.random())

def stable_seed(a, b, rep, base=42):
    return zlib.crc32(f"{a}|{b}|{rep}|{base}".encode()) & 0xFFFFFFFF

def make_strategy(name, rng, T, R, P, S):
    """Factory: returns a dict representing a strategy instance."""
    return {"name": name, "history": [], "opp": [], "rng": rng,
            "T": T, "R": R, "P": P, "S": S,
            "triggered": False, "defects": 0, "punish": 0, "calm": 0,
            "alpha": 1.0, "beta": 1.0, "mode": "test"}

def get_move(s):
    name, h, o = s["name"], s["history"], s["opp"]
    T, R, P, S = s["T"], s["R"], s["P"], s["S"]
    rng = s["rng"]
    if name == "TitForTat":   return 'C' if not o else o[-1]
    if name == "GrimTrigger":
        if not s["triggered"] and 'D' in o: s["triggered"] = True
        return 'D' if s["triggered"] else 'C'
    if name == "Pavlov":
        if not h: return 'C'
        lm, lo = h[-1], o[-1]
        pay = R if (lm=='C' and lo=='C') else T if (lm=='D' and lo=='C') else S if (lm=='C') else P
        return lm if pay in (R, T) else ('D' if lm=='C' else 'C')
    if name == "AlwaysDefect":    return 'D'
    if name == "AlwaysCooperate": return 'C'
    if name == "TwoTitsForTat":
        if len(o) < 2: return 'C'
        return 'D' if o[-1]=='D' and o[-2]=='D' else 'C'
    if name == "Random":    return 'C' if rng.random() < 0.5 else 'D'
    if name == "Joss":
        intended = 'C' if not o else o[-1]
        return 'D' if intended=='C' and rng.random() < 0.1 else intended
    if name == "Gradual":
        if s["punish"] > 0:
            s["punish"] -= 1
            if s["punish"] == 0: s["calm"] = 2
            return 'D'
        if s["calm"] > 0: s["calm"] -= 1; return 'C'
        if o and o[-1]=='D': s["defects"]+=1; s["punish"]=s["defects"]-1; return 'D'
        return 'C'
    if name == "Adaptive":
        p = s["alpha"]/(s["alpha"]+s["beta"])
        return 'C' if p*R+(1-p)*S >= p*T+(1-p)*P else 'D'
    if name == "Memory3":
        if len(o) < 3: return 'C'
        return 'C' if o[-3:].count('C') >= 2 else 'D'
    if name == "Friedman":
        if not s["triggered"] and 'D' in o: s["triggered"] = True
        return 'D' if s["triggered"] else 'C'
    if name == "Tester":
        if not h: return 'D'
        if s["mode"]=='test':
            s["mode"] = 'tft' if 'D' in o[:2] else 'exploit'
        if s["mode"]=='tft': return o[-1] if o else 'C'
        return 'C' if h[-1]=='D' else 'D'
    if name in ("EvolvedNN","PSOPlayer","Memory3"):
        po = o.count('C')/len(o) if o else 0.5
        return 'C' if po >= 0.5 else 'D'
    return 'C'

def update_adaptive(s, my, opp_move):
    s["history"].append(my); s["opp"].append(opp_move)
    if s["name"] == "Adaptive":
        if opp_move=='C': s["alpha"]+=1
        else: s["beta"]+=1

def get_payoff(m1, m2, T, R, P, S):
    if m1=='C' and m2=='C': return R,R
    if m1=='C' and m2=='D': return S,T
    if m1=='D' and m2=='C': return T,S
    return P,P

def run_live_tournament(selected_strats, rounds, repeats, w, T, R, P, S):
    names = selected_strats
    rows = []
    for nA in names:
        for nB in names:
            for rep in range(repeats):
                seed = stable_seed(nA, nB, rep)
                rng = ReproducibleRNG(seed)
                A = make_strategy(nA, rng, T, R, P, S)
                B = make_strategy(nB, rng, T, R, P, S)
                sA = sB = coopA = coopB = 0
                for _ in range(rounds):
                    mA = get_move(A); mB = get_move(B)
                    if rng.random() < w: mA = 'D' if mA=='C' else 'C'
                    if rng.random() < w: mB = 'D' if mB=='C' else 'C'
                    pA, pB = get_payoff(mA, mB, T, R, P, S)
                    sA+=pA; sB+=pB; coopA+=(mA=='C'); coopB+=(mB=='C')
                    update_adaptive(A, mA, mB); update_adaptive(B, mB, mA)
                rows.append({"Strategy_A":nA,"Strategy_B":nB,"rep":rep,
                              "Score_A":sA,"Score_B":sB,
                              "CoopRate_A":coopA/rounds,"CoopRate_B":coopB/rounds})
    df = pd.DataFrame(rows)
    matrix = df.groupby(["Strategy_A","Strategy_B"])["Score_A"].mean().unstack()
    ranking = df.groupby("Strategy_A")["Score_A"].mean().sort_values(ascending=False)
    return df, matrix, ranking

# ── Chart builders ────────────────────────────────────────
def build_ranking_bar(ranking, title=""):
    colors = [STRAT_COLOR.get(s, ACCENT) for s in ranking.index]
    fig = go.Figure(go.Bar(
        x=ranking.values, y=ranking.index, orientation='h',
        marker_color=colors, text=[f"{v:.1f}" for v in ranking.values],
        textposition='outside', textfont=dict(size=11, color=TEXT_SEC),
        hovertemplate="<b>%{y}</b><br>Score promedio: %{x:.2f}<extra></extra>"
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=14, color=TEXT_PRI)),
                      yaxis=dict(autorange="reversed", gridcolor=BORDER),
                      xaxis=dict(gridcolor=BORDER), height=420)
    return fig

def build_heatmap(matrix, title=""):
    strats = list(matrix.index)
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=strats, y=strats,
        colorscale="Viridis", showscale=True,
        text=[[f"{v:.0f}" for v in row] for row in matrix.values],
        texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b> vs <b>%{x}</b><br>Score: %{z:.1f}<extra></extra>",
        colorbar=dict(thickness=12, tickfont=dict(color=TEXT_SEC))
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=500,
                      title=dict(text=title, font=dict(size=14, color=TEXT_PRI)),
                      xaxis=dict(tickangle=-40, tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"),
                      yaxis=dict(tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"))
    return fig

def build_boxplot(df, title=""):
    fig = go.Figure()
    order = df.groupby("Strategy_A")["Score_A"].mean().sort_values(ascending=False).index
    for s in order:
        vals = df[df["Strategy_A"]==s]["Score_A"]
        fig.add_trace(go.Box(y=vals, name=s, marker_color=STRAT_COLOR.get(s,ACCENT),
                              line_color=STRAT_COLOR.get(s,ACCENT), fillcolor=STRAT_COLOR.get(s,ACCENT)+"44",
                              showlegend=False,
                              hovertemplate=f"<b>{s}</b><br>Score: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=420,
                      title=dict(text=title, font=dict(size=14, color=TEXT_PRI)),
                      xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
                      yaxis_title="Score")
    return fig

def build_coop_scatter(df, title=""):
    agg = df.groupby("Strategy_A").agg(
        Score=("Score_A","mean"), CoopRate=("CoopRate_A","mean")).reset_index()
    fig = px.scatter(agg, x="CoopRate", y="Score", text="Strategy_A",
                     color="Strategy_A", color_discrete_map=STRAT_COLOR, size_max=14)
    fig.update_traces(textposition="top center", textfont=dict(size=9, color=TEXT_SEC),
                      marker=dict(size=12))
    fig.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                      title=dict(text=title, font=dict(size=14, color=TEXT_PRI)),
                      xaxis_title="Tasa de cooperación promedio",
                      yaxis_title="Score promedio")
    return fig

def build_tariff_history(code):
    sector_name = {"72":"Acero","84":"Maquinaria","85":"Eléctricos"}[str(code)]
    fig = go.Figure()
    colors_map = {"AHS":ACCENT2,"BND":"#ef5350","MFN":"#ffa726"}
    labels = {"AHS":"AHS (cooperación)","BND":"BND (traición)","MFN":"MFN (nación más favorecida)"}
    for partner, label, dash_style in [("usa_mex","USA→MEX","solid"),("mex_usa","MEX→USA","dash")]:
        for dt in ["AHS","BND","MFN"]:
            key = f"{code}_{partner}_{dt}"
            d = DATA["real"]["hist_tariffs"].get(key,{})
            if not d: continue
            years = sorted(d.keys(), key=int)
            vals  = [d[y] for y in years]
            fig.add_trace(go.Scatter(
                x=[int(y) for y in years], y=vals, mode="lines+markers",
                name=f"{label} — {labels[dt]}",
                line=dict(color=colors_map[dt], dash=dash_style, width=2),
                marker=dict(size=5),
                hovertemplate=f"<b>{label} {dt}</b><br>Año: %{{x}}<br>Arancel: %{{y:.2f}}%<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=360,
                      title=dict(text=f"Evolución arancelaria — {sector_name}", font=dict(size=14,color=TEXT_PRI)),
                      xaxis_title="Año", yaxis_title="Arancel ponderado (%)",
                      legend=dict(font=dict(size=10)))
    return fig

def build_impact_bar():
    codes = ["72","84","85"]
    nombres = [DATA["real"]["impact"][c]["nombre"] for c in codes]
    loss_mex = [DATA["real"]["impact"][c]["loss_mex_bUSD"] for c in codes]
    loss_usa = [DATA["real"]["impact"][c]["loss_usa_bUSD"] for c in codes]
    fig = go.Figure([
        go.Bar(name="Daño a México (mil millones USD)", x=nombres, y=loss_mex,
               marker_color="#ef5350", hovertemplate="Daño MEX: $%{y:.2f}B<extra></extra>"),
        go.Bar(name="Daño a EE.UU. (mil millones USD)", x=nombres, y=loss_usa,
               marker_color=ACCENT, hovertemplate="Daño USA: $%{y:.2f}B<extra></extra>"),
    ])
    fig.update_layout(**PLOTLY_LAYOUT, barmode="group", height=360,
                      title=dict(text="Impacto económico estimado — escenario guerra arancelaria", font=dict(size=14,color=TEXT_PRI)),
                      yaxis_title="Miles de millones USD",
                      legend=dict(orientation="h", y=-0.25))
    return fig

def build_rng_histogram():
    samples = DATA["rng_tests"]["sample_200"]
    all_samples = DATA["rng_tests"]["sample_200"]
    fig = go.Figure(go.Histogram(
        x=all_samples, nbinsx=20, marker_color=ACCENT, opacity=0.8,
        hovertemplate="Bin: %{x:.2f}<br>Conteo: %{y}<extra></extra>"))
    fig.add_hline(y=len(all_samples)/20, line_dash="dash", line_color="#ffa726",
                  annotation_text="Esperado (uniforme)", annotation_font_color="#ffa726")
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      title=dict(text="Histograma RNG (200 muestras)", font=dict(size=13,color=TEXT_PRI)),
                      xaxis_title="Valor", yaxis_title="Frecuencia")
    return fig

def build_rng_scatter():
    s = DATA["rng_tests"]["sample_200"]
    fig = go.Figure(go.Scatter(
        x=s[:-1], y=s[1:], mode="markers",
        marker=dict(size=4, color=ACCENT2, opacity=0.6),
        hovertemplate="x(n): %{x:.4f}<br>x(n+1): %{y:.4f}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=300,
                      title=dict(text="Diagrama de dispersión lag-1", font=dict(size=13,color=TEXT_PRI)),
                      xaxis_title="x(n)", yaxis_title="x(n+1)")
    return fig

# ── App layout ────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY],
                suppress_callback_exceptions=True, title="Axelrod & Hamilton — Dashboard")

app.index_string = '''
<!DOCTYPE html>
<html>
<head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  * { box-sizing: border-box; }
  body { background: #0f1117; color: #e8eaf6; font-family: "Inter", sans-serif; margin: 0; }
  ::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #1a1d27; }
  ::-webkit-scrollbar-thumb { background: #2e3248; border-radius: 3px; }
  .nav-tab { border: none !important; background: transparent !important;
             color: #9fa8da !important; font-size: 13px; padding: 10px 18px;
             border-bottom: 2px solid transparent !important; cursor: pointer; transition: all .2s; }
  .nav-tab:hover { color: #e8eaf6 !important; }
  .nav-tab--selected { color: #7986cb !important; border-bottom: 2px solid #7986cb !important; font-weight: 600; }
  .sub-tab { font-size: 12px !important; padding: 7px 14px !important; }
  .metric-card { background: #1a1d27; border: 1px solid #2e3248; border-radius: 8px;
                 padding: 14px 18px; text-align: center; }
  .metric-val  { font-size: 22px; font-weight: 700; color: #7986cb; }
  .metric-lbl  { font-size: 11px; color: #9fa8da; margin-top: 2px; }
  .code-block  { background: #12141c; border: 1px solid #2e3248; border-radius: 6px;
                 padding: 12px 16px; font-family: "Fira Code", monospace; font-size: 12px;
                 color: #80cbc4; white-space: pre-wrap; overflow-x: auto; }
  .strat-card  { background: #1a1d27; border: 1px solid #2e3248; border-radius: 8px;
                 padding: 14px; cursor: pointer; transition: all .15s; margin-bottom: 8px; }
  .strat-card:hover { border-color: #7986cb; background: #1f2235; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px;
          font-weight: 600; margin-right: 4px; }
  .progress-bar-container { background: #2e3248; border-radius: 4px; height: 8px; margin: 8px 0; }
  .tooltip-inner { background: #1a1d27; color: #e8eaf6; border: 1px solid #2e3248; font-size: 12px; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body>
</html>
'''

NAVBAR = html.Div([
    html.Div([
        html.Span("🎮", style=dict(fontSize="20px", marginRight="10px")),
        html.Span("Axelrod & Hamilton", style=dict(fontWeight=700, fontSize="16px", color=TEXT_PRI)),
        html.Span("Evolución de la Cooperación", style=dict(fontSize="12px", color=TEXT_SEC, marginLeft="10px")),
    ], style=dict(display="flex", alignItems="center", padding="12px 24px",
                  borderBottom=f"1px solid {BORDER}", background=CARD_BG)),
], style=dict(position="sticky", top=0, zIndex=100))

MAIN_TABS = dcc.Tabs(id="main-tabs", value="tab-paper", children=[
    dcc.Tab(label="📄  Paper", value="tab-paper", className="nav-tab", selected_className="nav-tab--selected"),
    dcc.Tab(label="🔬  Metodología", value="tab-method", className="nav-tab", selected_className="nav-tab--selected"),
    dcc.Tab(label="🏆  Torneo Clásico", value="tab-classic", className="nav-tab", selected_className="nav-tab--selected"),
    dcc.Tab(label="🌎  Caso MEX–USA", value="tab-real", className="nav-tab", selected_className="nav-tab--selected"),
    dcc.Tab(label="⚡  Simulación", value="tab-sim", className="nav-tab", selected_className="nav-tab--selected"),
    dcc.Tab(label="📊  Análisis", value="tab-analysis", className="nav-tab", selected_className="nav-tab--selected"),
    dcc.Tab(label="🔧  Anexos", value="tab-annexes", className="nav-tab", selected_className="nav-tab--selected"),
], style=dict(background=CARD_BG, borderBottom=f"1px solid {BORDER}", padding="0 16px"))

app.layout = html.Div([
    NAVBAR,
    html.Div([MAIN_TABS,
              html.Div(id="main-content", style=dict(padding="20px 24px"))
              ], style=dict(maxWidth="1400px", margin="0 auto"))
], style=dict(background=DARK_BG, minHeight="100vh"))


# ═══════════════════════════════════════════════════════════
# TAB: PAPER
# ═══════════════════════════════════════════════════════════
def render_paper():
    sub_tabs = dcc.Tabs(id="paper-sub", value="ps-overview", children=[
        dcc.Tab(label="Resumen", value="ps-overview", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="Hallazgos", value="ps-findings", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="Anexo matemático", value="ps-math", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
    ], style=dict(marginBottom="16px", background="transparent"))
    return html.Div([sub_tabs, html.Div(id="paper-sub-content")])

def render_paper_overview():
    return html.Div([
        card([
            html.Div([
                html.Div([
                    html.H2("The Evolution of Cooperation", style=dict(color=TEXT_PRI, fontWeight=700, marginBottom="4px")),
                    html.P("Robert Axelrod & William D. Hamilton · Science, Vol. 211 · 1981",
                           style=dict(color=TEXT_SEC, fontSize="13px", marginBottom="16px")),
                    html.P("""En 1981, Axelrod y Hamilton se preguntaron algo que parecía paradójico: 
                    ¿cómo puede emerger la cooperación entre individuos egoístas, sin autoridad central y sin comunicación directa?
                    La respuesta la encontraron en un escenario sorprendentemente simple: el Dilema del Prisionero Iterado (IPD).""",
                           style=dict(color=TEXT_PRI, lineHeight="1.8", fontSize="14px")),
                    html.P("""El experimento central fue un torneo de computadora: diferentes estrategias 
                    compitieron en partidas repetidas. Cada par jugó 200 rondas. La estrategia que acumuló más 
                    puntos ganó. Participaron 14 estrategias de expertos en teoría de juegos, psicología, 
                    economía y ciencias políticas.""",
                           style=dict(color=TEXT_PRI, lineHeight="1.8", fontSize="14px")),
                ], style=dict(flex=1)),
                html.Div([
                    html.Div([html.Div("📅", style=dict(fontSize="24px")),
                              html.Div("1981", className="metric-val"),
                              html.Div("Año de publicación", className="metric-lbl")], className="metric-card", style=dict(marginBottom="10px")),
                    html.Div([html.Div("🏆", style=dict(fontSize="24px")),
                              html.Div("TFT", className="metric-val"),
                              html.Div("Estrategia ganadora", className="metric-lbl")], className="metric-card", style=dict(marginBottom="10px")),
                    html.Div([html.Div("📊", style=dict(fontSize="24px")),
                              html.Div("14→63K", className="metric-val"),
                              html.Div("Citas académicas", className="metric-lbl")], className="metric-card"),
                ], style=dict(minWidth="160px", marginLeft="24px")),
            ], style=dict(display="flex", alignItems="flex-start")),
        ]),
        card([
            section_title("El Dilema del Prisionero", "La estructura de pagos que lo hace todo interesante"),
            html.Div([
                html.Div([
                    html.P("Dos jugadores eligen simultáneamente Cooperar (C) o Traicionar (D).", style=dict(color=TEXT_PRI, fontSize="14px")),
                    html.P("La condición T > R > P > S garantiza que traicionar es siempre la estrategia dominante en una sola ronda — pero en repetidas rondas, la cooperación puede estabilizarse.", style=dict(color=TEXT_SEC, fontSize="13px")),
                    html.Div([
                        html.Span("T = 5", style=dict(color="#ef5350", fontWeight=600, marginRight="16px")),
                        html.Span("R = 3", style=dict(color=ACCENT2, fontWeight=600, marginRight="16px")),
                        html.Span("P = 1", style=dict(color="#ffa726", fontWeight=600, marginRight="16px")),
                        html.Span("S = 0", style=dict(color=TEXT_SEC, fontWeight=600)),
                    ], style=dict(marginTop="12px", fontSize="15px")),
                ], style=dict(flex=1, paddingRight="24px")),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([html.Th(""), html.Th("Oponente: C", style=dict(color=ACCENT2)), html.Th("Oponente: D", style=dict(color="#ef5350"))]),
                                   style=dict(borderBottom=f"1px solid {BORDER}")),
                        html.Tbody([
                            html.Tr([html.Td("Yo: C"), html.Td("R, R = 3, 3", style=dict(color=ACCENT2, fontWeight=600)), html.Td("S, T = 0, 5", style=dict(color="#ffa726"))]),
                            html.Tr([html.Td("Yo: D"), html.Td("T, S = 5, 0", style=dict(color="#ef5350", fontWeight=600)), html.Td("P, P = 1, 1", style=dict(color=TEXT_SEC))]),
                        ])
                    ], style=dict(borderCollapse="collapse", fontSize="13px", color=TEXT_PRI, width="100%")),
                ], style=dict(background="#12141c", padding="16px", borderRadius="8px", minWidth="280px")),
            ], style=dict(display="flex", alignItems="flex-start")),
        ]),
    ])

def render_paper_findings():
    findings = [
        ("🤝", "Cooperación emergente", "La cooperación puede surgir y mantenerse entre agentes puramente egoístas, sin comunicación, sin autoridad central y sin altruismo. Solo se necesitan interacciones repetidas."),
        ("📏", "Tit-for-Tat ganó el torneo", "La estrategia más simple — coopera primero, luego imita — derrotó a 13 estrategias elaboradas. Ganó por ser amable (nice), provocable (retaliatory), perdonadora (forgiving) y clara (clear)."),
        ("🔁", "El futuro importa", "La cooperación es estable cuando la probabilidad de futuras interacciones (δ) es suficientemente alta. Axelrod lo llamó 'la sombra del futuro'."),
        ("🛡️", "Robustez evolutiva", "TFT es una Estrategia Evolutivamente Estable (ESS): una población de TFT no puede ser invadida por mutantes defectores, siempre que δ sea lo suficientemente grande."),
        ("🌱", "Cooperación desde abajo", "Basta con que pequeños grupos de cooperadores existan para que la cooperación se extienda, incluso en poblaciones hostiles."),
        ("⚖️", "Reciprocidad directa", "El mecanismo fundamental es la reciprocidad: los individuos cooperan porque esperan ser reciprocados. No requiere empatía ni moralidad."),
    ]
    return html.Div([
        card([
            section_title("Hallazgos principales del paper"),
            html.Div([
                html.Div([
                    html.Div([
                        html.Span(emoji, style=dict(fontSize="22px", marginRight="10px")),
                        html.Div([
                            html.Div(title, style=dict(fontWeight=600, color=TEXT_PRI, fontSize="14px")),
                            html.Div(desc, style=dict(color=TEXT_SEC, fontSize="13px", marginTop="4px", lineHeight="1.6")),
                        ])
                    ], style=dict(display="flex", alignItems="flex-start"))
                ], className="strat-card")
                for emoji, title, desc in findings
            ])
        ])
    ])

def render_paper_math():
    return html.Div([
        card([
            section_title("Anexo matemático", "Por qué el equilibrio de Nash en el IPD es Tit-for-Tat"),
            html.P("Este anexo muestra formalmente las condiciones bajo las cuales TFT constituye un Equilibrio de Nash en el Dilema del Prisionero Iterado.", style=dict(color=TEXT_SEC, fontSize="13px", marginBottom="20px")),

            html.H5("1. El modelo", style=dict(color=ACCENT, marginBottom="8px")),
            html.P(["Sea δ ∈ (0,1) la probabilidad de que el juego continúe en la siguiente ronda (o el factor de descuento). El payoff total esperado de una secuencia de acciones es:",
                    html.Br(), html.Br()], style=dict(color=TEXT_PRI, fontSize="13px")),
            html.Div("V = Σ_{t=0}^{∞} δᵗ · π(t)   donde π(t) es el payoff en la ronda t", className="code-block", style=dict(marginBottom="16px")),

            html.H5("2. Payoff de TFT vs TFT", style=dict(color=ACCENT, marginBottom="8px")),
            html.P("Dos jugadores TFT siempre cooperan. El payoff presente es:", style=dict(color=TEXT_PRI, fontSize="13px")),
            html.Div("V(TFT | TFT) = R + δR + δ²R + ... = R / (1 - δ)", className="code-block", style=dict(marginBottom="16px")),

            html.H5("3. Payoff de Defección permanente (AllD) vs TFT", style=dict(color=ACCENT, marginBottom="8px")),
            html.P("Un mutante AllD obtiene T en la primera ronda (explota a TFT), luego P para siempre (TFT retalia):", style=dict(color=TEXT_PRI, fontSize="13px")),
            html.Div("V(AllD | TFT) = T + δP + δ²P + ... = T + δP/(1-δ)", className="code-block", style=dict(marginBottom="16px")),

            html.H5("4. Condición de estabilidad", style=dict(color=ACCENT, marginBottom="8px")),
            html.P("TFT es un Equilibrio de Nash si ningún jugador se beneficia desviándose:", style=dict(color=TEXT_PRI, fontSize="13px")),
            html.Div("V(TFT|TFT) ≥ V(AllD|TFT)\n\nR/(1-δ) ≥ T + δP/(1-δ)\n\nDespejando δ:\n\nδ ≥ (T - R) / (T - P)", className="code-block", style=dict(marginBottom="16px")),

            html.H5("5. Con los valores de Axelrod (T=5, R=3, P=1, S=0)", style=dict(color=ACCENT, marginBottom="8px")),
            html.Div("δ ≥ (5 - 3) / (5 - 1) = 2/4 = 0.50\n\nCon δ ≥ 0.5 (o ROUNDS ≥ 200 con w=0.995), TFT es estable.", className="code-block", style=dict(marginBottom="16px")),

            html.H5("6. Folk Theorem", style=dict(color=ACCENT, marginBottom="8px")),
            html.P(["El ", html.Strong("Folk Theorem del IPD"), " establece que cualquier resultado factible e individualmente racional puede sostenerse como equilibrio de Nash si δ es suficientemente alto. TFT es solo uno de muchos equilibrios posibles, pero es el más robusto porque también resiste invasión por estrategias cooperadoras menos estrictas (ESS)."],
                   style=dict(color=TEXT_PRI, fontSize="13px", lineHeight="1.8")),

            html.Div([
                html.Span("En el código: ", style=dict(color=TEXT_SEC, fontSize="12px")),
                html.Span(f"W=0.995 → δ≈0.995 >> 0.5 ✓  |  ROUNDS=200 asegura la sombra del futuro",
                          style=dict(color=ACCENT2, fontSize="12px", fontFamily="monospace")),
            ], style=dict(background="#12141c", padding="10px 14px", borderRadius="6px", marginTop="8px")),
        ])
    ])


# ═══════════════════════════════════════════════════════════
# TAB: METODOLOGÍA
# ═══════════════════════════════════════════════════════════
def render_methodology():
    sub_tabs = dcc.Tabs(id="method-sub", value="ms-rr", children=[
        dcc.Tab(label="Torneo Round-Robin", value="ms-rr", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="Las 15 estrategias", value="ms-strats", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
    ], style=dict(marginBottom="16px", background="transparent"))
    return html.Div([sub_tabs, html.Div(id="method-sub-content")])

def render_rr_explanation():
    return html.Div([
        card([
            section_title("¿Por qué Round-Robin?", "Justificación de la metodología elegida"),
            html.Div([
                html.Div([
                    html.Div("✅", style=dict(fontSize="20px")),
                    html.Div([html.Strong("Imparcialidad total: ", style=dict(color=TEXT_PRI)),
                              html.Span("Cada estrategia enfrenta a todas las demás exactamente el mismo número de veces. No hay ventajas por emparejamiento.", style=dict(color=TEXT_SEC))],
                             style=dict(fontSize="13px", lineHeight="1.7"))
                ], style=dict(display="flex", gap="12px", marginBottom="12px")),
                html.Div([
                    html.Div("✅", style=dict(fontSize="20px")),
                    html.Div([html.Strong("Fidelidad al paper: ", style=dict(color=TEXT_PRI)),
                              html.Span("Axelrod usó exactamente esta metodología en su torneo original de 1980.", style=dict(color=TEXT_SEC))],
                             style=dict(fontSize="13px", lineHeight="1.7"))
                ], style=dict(display="flex", gap="12px", marginBottom="12px")),
                html.Div([
                    html.Div("✅", style=dict(fontSize="20px")),
                    html.Div([html.Strong("Ranking robusto: ", style=dict(color=TEXT_PRI)),
                              html.Span("El score total refleja el desempeño contra todo el ecosistema, no solo contra un subconjunto.", style=dict(color=TEXT_SEC))],
                             style=dict(fontSize="13px", lineHeight="1.7"))
                ], style=dict(display="flex", gap="12px")),
            ], style=dict(marginBottom="20px")),
        ]),
        card([
            section_title("Cómo funciona el torneo", "Paso a paso de la implementación"),
            html.Div([
                html.Div([
                    html.Div(step, style=dict(background=ACCENT, color="white", borderRadius="50%",
                                              width="28px", height="28px", display="flex",
                                              alignItems="center", justifyContent="center",
                                              fontWeight=700, fontSize="12px", minWidth="28px")),
                    html.Div(desc, style=dict(color=TEXT_PRI, fontSize="13px", lineHeight="1.7", paddingLeft="12px")),
                ], style=dict(display="flex", alignItems="flex-start", marginBottom="14px"))
                for step, desc in [
                    ("1", "Para cada par (A, B) de estrategias, se juegan 5 repeticiones independientes de 200 rondas cada una."),
                    ("2", "Cada repetición usa una semilla CRC32 determinista derivada de los nombres de las estrategias y el número de repetición — garantizando reproducibilidad entre sesiones."),
                    ("3", "En cada ronda, ambas estrategias eligen simultáneamente C o D. Se aplica ruido probabilístico (w) para simular errores de implementación."),
                    ("4", "Los payoffs se acumulan según la matriz T > R > P > S. Al final de las 200 rondas, se registra el score total de A."),
                    ("5", "El score final de cada estrategia es el promedio de sus scores contra todos los oponentes (incluyendo sí misma) en todas las repeticiones."),
                ]
            ]),
            html.Div([
                html.Div("Parámetros del torneo clásico:", style=dict(color=TEXT_SEC, fontSize="12px", marginBottom="8px")),
                html.Div("ROUNDS = 200  |  REPEATS = 5  |  w = 0.0 (sin ruido)\nT = 5  |  R = 3  |  P = 1  |  S = 0", className="code-block"),
            ]),
        ]),
    ])

def render_strategies_cards():
    return html.Div([
        card([
            section_title("Las 15 estrategias del torneo", "Haz clic en cualquier estrategia para ver sus detalles"),
            html.Div([
                html.Div([
                    html.Div(id=f"strat-card-{s}", children=[
                        html.Div([
                            html.Span(STRAT_INFO[s]["emoji"], style=dict(fontSize="18px", marginRight="10px")),
                            html.Span(s, style=dict(fontWeight=600, color=TEXT_PRI, fontSize="14px")),
                        ], style=dict(display="flex", alignItems="center", marginBottom="6px")),
                        html.P(STRAT_INFO[s]["short"], style=dict(color=TEXT_SEC, fontSize="12px", margin=0, lineHeight="1.5")),
                        html.Div(badge(STRAT_INFO[s]["prop"].split(" · ")[0]), style=dict(marginTop="8px")),
                    ], className="strat-card", n_clicks=0, style=dict(cursor="pointer")),
                ], style=dict(width="calc(33% - 8px)", minWidth="200px"))
                for s in STRATS
            ], style=dict(display="flex", flexWrap="wrap", gap="8px")),
        ]),
        html.Div(id="strat-detail-panel"),
    ])


# ═══════════════════════════════════════════════════════════
# TAB: TORNEO CLÁSICO
# ═══════════════════════════════════════════════════════════
def render_classic():
    return html.Div([
        # Traducción estrategia → código
        card([
            section_title("De estrategia a código", "Cómo se traduce cada regla de decisión a Python"),
            dcc.Dropdown(id="code-strat-select", options=[{"label":s,"value":s} for s in STRATS],
                         value="TitForTat", clearable=False,
                         style=dict(background=CARD_BG, color=TEXT_PRI, borderColor=BORDER, marginBottom="12px")),
            html.Div(id="code-display"),
        ]),
        # Métricas
        html.Div(id="classic-metrics"),
        # Resultados
        dcc.Tabs(id="classic-result-tabs", value="crt-ranking", children=[
            dcc.Tab(label="Ranking", value="crt-ranking", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Heatmap", value="crt-heatmap", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Cooperación vs Score", value="crt-scatter", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Resultados individuales", value="crt-individual", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Pruebas estadísticas", value="crt-stats", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Código fuente", value="crt-code", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        ], style=dict(background="transparent", marginBottom="8px")),
        html.Div(id="classic-result-content"),
    ])


# ═══════════════════════════════════════════════════════════
# TAB: CASO REAL MEX-USA
# ═══════════════════════════════════════════════════════════
def render_real():
    return html.Div([
        card([
            section_title("Datos WITS — Aranceles Bilaterales", "World Integrated Trade Solution · Banco Mundial"),
            html.Div([
                html.Div([
                    html.P("""Los datos provienen del repositorio WITS (World Integrated Trade Solution) del Banco Mundial. 
                    Contienen los aranceles ponderados por valor de importación para 3 sectores estratégicos de la relación comercial MEX–USA, 
                    bajo el T-MEC/USMCA, en el período 2014–2023.""", style=dict(color=TEXT_PRI, fontSize="13px", lineHeight="1.8")),
                    html.Div([
                        html.Div([html.Strong("AHS", style=dict(color=ACCENT2)), " → ", html.Span("Proxy de ", style=dict(color=TEXT_SEC)), html.Strong("Cooperación", style=dict(color=ACCENT2)), html.Span(": arancel efectivamente aplicado (weighted average). Cuando ambos países respetan el T-MEC.", style=dict(color=TEXT_SEC))], style=dict(fontSize="13px", marginBottom="8px")),
                        html.Div([html.Strong("BND", style=dict(color="#ef5350")), " → ", html.Span("Proxy de ", style=dict(color=TEXT_SEC)), html.Strong("Traición", style=dict(color="#ef5350")), html.Span(": arancel consolidado máximo (bound rate). Escenario de ruptura del acuerdo / guerra comercial.", style=dict(color=TEXT_SEC))], style=dict(fontSize="13px")),
                    ], style=dict(background="#12141c", padding="14px", borderRadius="8px", marginTop="12px")),
                ], style=dict(flex=1, paddingRight="24px")),
                html.Div([
                    html.Div([
                        dcc.Dropdown(id="sector-select", options=[{"label":f"Sector {c} — {n}","value":str(c)} for c,n in {"72":"Acero","84":"Maquinaria","85":"Eléctricos"}.items()],
                                     value="72", clearable=False, style=dict(marginBottom="12px")),
                        html.Div(id="sector-payoff-display"),
                    ])
                ], style=dict(minWidth="260px")),
            ], style=dict(display="flex")),
        ]),
        card([dcc.Graph(id="tariff-history-chart")]),
        card([dcc.Graph(figure=build_impact_bar())]),
        dcc.Tabs(id="real-result-tabs", value="rrt-ranking", children=[
            dcc.Tab(label="Ranking", value="rrt-ranking", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Heatmap", value="rrt-heatmap", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Estadísticas", value="rrt-stats", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Conclusiones", value="rrt-conclusions", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
            dcc.Tab(label="Código fuente", value="rrt-code", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        ], style=dict(background="transparent", marginBottom="8px")),
        html.Div(id="real-result-content"),
    ])


# ═══════════════════════════════════════════════════════════
# TAB: SIMULACIÓN EN VIVO
# ═══════════════════════════════════════════════════════════
def render_simulation():
    return html.Div([
        card([
            section_title("Simulación en tiempo real", "Configura y ejecuta tu propio torneo"),
            html.Div([
                html.Div([
                    html.Label("Estrategias a incluir", style=dict(color=TEXT_SEC, fontSize="12px", marginBottom="6px")),
                    dcc.Checklist(id="sim-strats", options=[{"label": f" {s}","value":s} for s in STRATS],
                                  value=["TitForTat","GrimTrigger","Pavlov","AlwaysDefect","AlwaysCooperate","Gradual"],
                                  labelStyle=dict(display="block", fontSize="13px", color=TEXT_PRI, marginBottom="3px")),
                ], style=dict(minWidth="200px", marginRight="32px")),
                html.Div([
                    html.Div([
                        html.Label(["Rondas por partida", tooltip_icon("tt-rounds","Número de rondas que juega cada par de estrategias.")],
                                   style=dict(color=TEXT_SEC, fontSize="12px")),
                        dcc.Slider(id="sim-rounds", min=50, max=500, step=50, value=200,
                                   marks={50:"50",200:"200",500:"500"}, tooltip=dict(placement="bottom")),
                    ], style=dict(marginBottom="20px")),
                    html.Div([
                        html.Label(["Repeticiones", tooltip_icon("tt-reps","Veces que se repite cada matchup para promediar el azar.")],
                                   style=dict(color=TEXT_SEC, fontSize="12px")),
                        dcc.Slider(id="sim-repeats", min=1, max=10, step=1, value=5,
                                   marks={1:"1",5:"5",10:"10"}, tooltip=dict(placement="bottom")),
                    ], style=dict(marginBottom="20px")),
                    html.Div([
                        html.Label(["Ruido w (probabilidad de error)", tooltip_icon("tt-w","Probabilidad de que un movimiento se invierta por error. Simula fallos de implementación o incertidumbre política.")],
                                   style=dict(color=TEXT_SEC, fontSize="12px")),
                        dcc.Slider(id="sim-w", min=0.0, max=0.3, step=0.01, value=0.0,
                                   marks={0:"0",0.095:"0.095\n(Real)",0.1:"0.1",0.3:"0.3"}, tooltip=dict(placement="bottom")),
                    ], style=dict(marginBottom="20px")),
                    html.Div([
                        html.Label("Payoffs personalizados (T, R, P, S)", style=dict(color=TEXT_SEC, fontSize="12px", marginBottom="6px")),
                        html.Div([
                            dcc.Input(id="sim-T", type="number", value=5, placeholder="T", style=dict(width="60px", marginRight="8px")),
                            dcc.Input(id="sim-R", type="number", value=3, placeholder="R", style=dict(width="60px", marginRight="8px")),
                            dcc.Input(id="sim-P", type="number", value=1, placeholder="P", style=dict(width="60px", marginRight="8px")),
                            dcc.Input(id="sim-S", type="number", value=0, placeholder="S", style=dict(width="60px")),
                            html.Span("T  R  P  S", style=dict(color=TEXT_SEC, fontSize="11px", marginLeft="8px")),
                        ]),
                    ], style=dict(marginBottom="20px")),
                    html.Button("▶  Ejecutar simulación", id="sim-run-btn",
                                style=dict(background=ACCENT, color="white", border="none",
                                           borderRadius="6px", padding="10px 24px",
                                           cursor="pointer", fontSize="14px", fontWeight=600)),
                ], style=dict(flex=1)),
            ], style=dict(display="flex")),
        ]),
        dcc.Loading(id="sim-loading", type="circle", color=ACCENT, children=[
            html.Div(id="sim-results"),
        ]),
        # Store for sim results
        dcc.Store(id="sim-store"),
    ])


# ═══════════════════════════════════════════════════════════
# TAB: ANÁLISIS COMPARATIVO
# ═══════════════════════════════════════════════════════════
def render_analysis():
    return html.Div([
        card([
            section_title("Comparación clásico vs real", "¿Cómo cambia el ranking cuando los payoffs son aranceles reales?"),
            html.Div([
                html.Div([dcc.Graph(figure=build_ranking_bar(rank_classic, "Ranking — Torneo clásico (T=5,R=3,P=1,S=0)"))], style=dict(flex=1)),
                html.Div([dcc.Graph(figure=build_ranking_bar(rank_real, "Ranking — Caso real MEX–USA (w=9.5%, payoffs WITS)"))], style=dict(flex=1)),
            ], style=dict(display="flex", gap="16px")),
        ]),
        card([
            section_title("Cooperación vs eficiencia", "¿Cooperar más implica mejores resultados?"),
            html.Div([
                html.Div([dcc.Graph(figure=build_coop_scatter(df_classic, "Clásico"))], style=dict(flex=1)),
                html.Div([dcc.Graph(figure=build_coop_scatter(df_real, "Real"))], style=dict(flex=1)),
            ], style=dict(display="flex", gap="16px")),
        ]),
        card([
            section_title("Selector de estrategia individual"),
            dcc.Dropdown(id="analysis-strat", options=[{"label":s,"value":s} for s in STRATS],
                         value="TitForTat", clearable=False,
                         style=dict(marginBottom="12px")),
            html.Div(id="analysis-strat-detail"),
        ]),
    ])


# ═══════════════════════════════════════════════════════════
# TAB: ANEXOS
# ═══════════════════════════════════════════════════════════
def render_annexes():
    sub_tabs = dcc.Tabs(id="annex-sub", value="ax-rng", children=[
        dcc.Tab(label="Tests RNG", value="ax-rng", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="Exportar datos", value="ax-export", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
        dcc.Tab(label="Código fuente", value="ax-code", className="nav-tab sub-tab", selected_className="nav-tab--selected"),
    ], style=dict(marginBottom="16px", background="transparent"))
    return html.Div([sub_tabs, html.Div(id="annex-sub-content")])

def render_rng_panel():
    rng = DATA["rng_tests"]
    return html.Div([
        card([
            section_title("Panel de calidad del generador de números aleatorios", "PCG64 con semillas CRC32 deterministas"),
            html.Div([
                html.Div([html.Div(f"{rng['ks_stat']:.4f}", className="metric-val"),
                          html.Div("Estadístico KS", className="metric-lbl"),
                          html.Div(f"p = {rng['ks_p']:.4f}", style=dict(color=ACCENT2 if rng['ks_p']>0.05 else "#ef5350", fontSize="11px", marginTop="2px"))
                          ], className="metric-card"),
                html.Div([html.Div(f"{rng['autocorr']:.6f}", className="metric-val"),
                          html.Div("Autocorrelación lag-1", className="metric-lbl"),
                          html.Div("≈0 ✓" if abs(rng['autocorr'])<0.02 else "Alto ⚠", style=dict(color=ACCENT2 if abs(rng['autocorr'])<0.02 else "#ffa726", fontSize="11px", marginTop="2px"))
                          ], className="metric-card"),
                html.Div([html.Div("PCG64", className="metric-val"),
                          html.Div("Algoritmo RNG", className="metric-lbl"),
                          html.Div("NumPy 1.17+", style=dict(color=TEXT_SEC, fontSize="11px", marginTop="2px"))
                          ], className="metric-card"),
                html.Div([html.Div("CRC32", className="metric-val"),
                          html.Div("Semillas por matchup", className="metric-lbl"),
                          html.Div("zlib · estable entre sesiones", style=dict(color=TEXT_SEC, fontSize="11px", marginTop="2px"))
                          ], className="metric-card"),
            ], style=dict(display="grid", gridTemplateColumns="repeat(4,1fr)", gap="12px", marginBottom="16px")),
            html.P([
                html.Strong("KS Test (Kolmogorov-Smirnov): ", style=dict(color=TEXT_PRI)),
                html.Span(f"p = {rng['ks_p']:.4f} {'> 0.05 → No se rechaza H₀ de uniformidad ✓' if rng['ks_p']>0.05 else '< 0.05 ⚠'}",
                          style=dict(color=ACCENT2 if rng['ks_p']>0.05 else "#ef5350"))
            ], style=dict(fontSize="13px", marginBottom="4px")),
            html.P([
                html.Strong("Autocorrelación: ", style=dict(color=TEXT_PRI)),
                html.Span(f"{rng['autocorr']:.6f} — {'Prácticamente cero, sin dependencia serial ✓' if abs(rng['autocorr'])<0.02 else 'Revisar dependencia serial ⚠'}",
                          style=dict(color=ACCENT2 if abs(rng['autocorr'])<0.02 else "#ffa726"))
            ], style=dict(fontSize="13px")),
        ]),
        html.Div([
            html.Div([card([dcc.Graph(figure=build_rng_histogram())])], style=dict(flex=1)),
            html.Div([card([dcc.Graph(figure=build_rng_scatter())])], style=dict(flex=1)),
        ], style=dict(display="flex", gap="16px")),
    ])

def render_export_panel():
    return html.Div([
        card([
            section_title("Exportar resultados"),
            html.P("Descarga los datos del torneo en el formato que necesites.", style=dict(color=TEXT_SEC, fontSize="13px", marginBottom="16px")),
            html.Div([
                html.Div([
                    html.H6("Torneo Clásico", style=dict(color=TEXT_PRI)),
                    html.Button("⬇ CSV detallado", id="btn-csv-classic", className="btn",
                                style=dict(marginRight="8px", marginBottom="8px", background=ACCENT+"33", color=ACCENT, border=f"1px solid {ACCENT}44", borderRadius="6px", padding="7px 14px", cursor="pointer")),
                    html.Button("⬇ JSON resultados", id="btn-json-classic",
                                style=dict(marginRight="8px", marginBottom="8px", background=ACCENT2+"33", color=ACCENT2, border=f"1px solid {ACCENT2}44", borderRadius="6px", padding="7px 14px", cursor="pointer")),
                    dcc.Download(id="download-csv-classic"),
                    dcc.Download(id="download-json-classic"),
                ], style=dict(marginBottom="16px")),
                html.Div([
                    html.H6("Caso Real MEX–USA", style=dict(color=TEXT_PRI)),
                    html.Button("⬇ CSV detallado", id="btn-csv-real",
                                style=dict(marginRight="8px", marginBottom="8px", background=ACCENT+"33", color=ACCENT, border=f"1px solid {ACCENT}44", borderRadius="6px", padding="7px 14px", cursor="pointer")),
                    html.Button("⬇ JSON resultados", id="btn-json-real",
                                style=dict(marginRight="8px", background=ACCENT2+"33", color=ACCENT2, border=f"1px solid {ACCENT2}44", borderRadius="6px", padding="7px 14px", cursor="pointer")),
                    dcc.Download(id="download-csv-real"),
                    dcc.Download(id="download-json-real"),
                ]),
            ]),
        ])
    ])

CODE_SOURCE = {
"Motor IPD": '''def play_game(A, B, rounds, w, T, R, P, S, rng):
    """Runs one IPD game between strategies A and B."""
    A.reset(); B.reset()
    score_A = score_B = 0
    for _ in range(rounds):
        move_A = A.move()
        move_B = B.move()
        # Apply noise (w = prob. of implementation error)
        if rng.random() < w: move_A = 'D' if move_A=='C' else 'C'
        if rng.random() < w: move_B = 'D' if move_B=='C' else 'C'
        pA, pB = get_payoff(move_A, move_B, T, R, P, S)
        score_A += pA; score_B += pB
        A.update(move_A, move_B)
        B.update(move_B, move_A)
    return score_A, score_B''',

"Semillas CRC32": '''import zlib

def stable_seed(strat_A: str, strat_B: str, rep: int, base: int = 42) -> int:
    """Deterministic seed stable across Python sessions.
    Uses CRC32 instead of built-in hash() (which is session-randomized since Python 3.3)
    """
    material = f"{strat_A}|{strat_B}|{rep}|{base}"
    return zlib.crc32(material.encode("utf-8")) & 0xFFFFFFFF''',

"Torneo Round-Robin": '''def round_robin_tournament(strategies, rounds=200, repeats=5, w=0.0,
                           T=5, R=3, P=1, S=0, base_seed=42):
    names = [instantiate(s, ...).name for s in strategies]
    results = []
    for strat_A in strategies:
        for strat_B in strategies:
            for rep in range(repeats):
                seed = stable_seed(strat_A.__name__, strat_B.__name__, rep, base_seed)
                rng = ReproducibleRNG(seed)
                A = instantiate(strat_A, rng, T, R, P, S)
                B = instantiate(strat_B, rng, T, R, P, S)
                sA, sB = play_game(A, B, rounds, w, T, R, P, S, rng)
                results.append({"Strategy_A": A.name, "Strategy_B": B.name,
                                 "rep": rep, "Score_A": sA, "Score_B": sB})
    return pd.DataFrame(results)''',

"Payoffs WITS": '''def get_bilateral_payoffs(df_tmec, product_code: int) -> dict:
    """Extract cooperation (AHS) and defection (BND) tariffs from WITS data."""
    usa_to_mex = df_tmec[(df_tmec["Product"] == product_code) &
                          (df_tmec["Partner Name"] == "Mexico")]
    mex_to_usa = df_tmec[(df_tmec["Product"] == product_code) &
                          (df_tmec["Partner Name"] == "United States")]
    return {
        "coop_usa": usa_to_mex[usa_to_mex["DutyType"]=="AHS"]["Weighted Average"].mean(),
        "defect_usa": usa_to_mex[usa_to_mex["DutyType"]=="BND"]["Weighted Average"].mean(),
        "coop_mex": mex_to_usa[mex_to_usa["DutyType"]=="AHS"]["Weighted Average"].mean(),
        "defect_mex": mex_to_usa[mex_to_usa["DutyType"]=="BND"]["Weighted Average"].mean(),
    }
# Mapping to IPD payoffs:
# R = 100 - coop_usa  (both cooperate: low tariff)
# S = 100 - defect_usa  (we cooperate, they defect: high tariff on us)
# T = 100 - defect_usa + 5  (we defect: we collect tariff revenue)
# P = 100 - defect_mex  (both defect: mutual high tariffs)'''
}

def render_code_panel():
    return html.Div([
        card([
            section_title("Código fuente destacado"),
            dcc.Tabs(id="code-tabs", value="Motor IPD", children=[
                dcc.Tab(label=k, value=k, className="nav-tab sub-tab", selected_className="nav-tab--selected")
                for k in CODE_SOURCE
            ], style=dict(background="transparent", marginBottom="12px")),
            html.Div(id="code-source-display"),
        ])
    ])


# ═══════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════
@app.callback(Output("main-content","children"), Input("main-tabs","value"))
def render_main(tab):
    if tab == "tab-paper":    return render_paper()
    if tab == "tab-method":   return render_methodology()
    if tab == "tab-classic":  return render_classic()
    if tab == "tab-real":     return render_real()
    if tab == "tab-sim":      return render_simulation()
    if tab == "tab-analysis": return render_analysis()
    if tab == "tab-annexes":  return render_annexes()
    return html.Div()

@app.callback(Output("paper-sub-content","children"), Input("paper-sub","value"))
def render_paper_sub(val):
    if val == "ps-overview": return render_paper_overview()
    if val == "ps-findings": return render_paper_findings()
    if val == "ps-math":     return render_paper_math()

@app.callback(Output("method-sub-content","children"), Input("method-sub","value"))
def render_method_sub(val):
    if val == "ms-rr":     return render_rr_explanation()
    if val == "ms-strats": return render_strategies_cards()

@app.callback(Output("strat-detail-panel","children"),
              [Input(f"strat-card-{s}","n_clicks") for s in STRATS],
              prevent_initial_call=True)
def show_strat_detail(*clicks):
    triggered = ctx.triggered_id
    if not triggered: return html.Div()
    s = triggered.replace("strat-card-","")
    info = STRAT_INFO[s]
    rc = rank_classic.get(s, 0); rr = rank_real.get(s, 0)
    pos_c = list(rank_classic.index).index(s)+1
    pos_r = list(rank_real.index).index(s)+1
    return card([
        html.Div([
            html.Span(info["emoji"], style=dict(fontSize="28px", marginRight="14px")),
            html.Div([
                html.H4(s, style=dict(color=TEXT_PRI, marginBottom="2px")),
                html.P(info["short"], style=dict(color=TEXT_SEC, fontSize="13px", margin=0)),
            ])
        ], style=dict(display="flex", alignItems="center", marginBottom="16px")),
        html.Div([badge(p) for p in info["prop"].split(" · ")], style=dict(marginBottom="14px")),
        html.Div([
            html.Div([html.Div(f"#{pos_c}", className="metric-val"), html.Div(f"Ranking clásico — {rc:.1f}pts", className="metric-lbl")], className="metric-card"),
            html.Div([html.Div(f"#{pos_r}", className="metric-val"), html.Div(f"Ranking real — {rr:.1f}pts", className="metric-lbl")], className="metric-card"),
        ], style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="12px", marginBottom="14px")),
        html.Div(["Lógica de decisión: ", html.Br(), html.Span(info["code"], style=dict(fontFamily="monospace", fontSize="12px", color=ACCENT2))],
                 className="code-block"),
    ], style=dict(borderColor=ACCENT+"66"))

@app.callback(Output("code-display","children"), Input("code-strat-select","value"))
def show_code(s):
    if not s: return html.Div()
    info = STRAT_INFO[s]
    return html.Div([
        html.Div([
            html.Span(info["emoji"]+" ", style=dict(fontSize="16px")),
            html.Strong(s+": ", style=dict(color=TEXT_PRI)),
            html.Span(info["short"], style=dict(color=TEXT_SEC, fontSize="13px")),
        ], style=dict(marginBottom="10px")),
        html.Div(info["code"], className="code-block"),
    ])

@app.callback(Output("classic-metrics","children"), Input("main-tabs","value"))
def classic_metrics(_):
    top = rank_classic.index[0]
    return html.Div([
        html.Div([html.Div(f"{rank_classic.iloc[0]:.1f}", className="metric-val"), html.Div(f"Máx score — {top}", className="metric-lbl")], className="metric-card"),
        html.Div([html.Div(f"{rank_classic.iloc[-1]:.1f}", className="metric-val"), html.Div(f"Mín score — {rank_classic.index[-1]}", className="metric-lbl")], className="metric-card"),
        html.Div([html.Div(f"{DATA['classic']['anova']['F']:.2f}", className="metric-val"), html.Div("F-stat ANOVA", className="metric-lbl")], className="metric-card"),
        html.Div([html.Div(f"{DATA['classic']['anova']['p']:.2e}", className="metric-val"), html.Div("p-value", className="metric-lbl")], className="metric-card"),
    ], style=dict(display="grid", gridTemplateColumns="repeat(4,1fr)", gap="12px", marginBottom="16px"))

@app.callback(Output("classic-result-content","children"), Input("classic-result-tabs","value"))
def classic_results(tab):
    if tab == "crt-ranking":
        return card([dcc.Graph(figure=build_ranking_bar(rank_classic, "Ranking — Torneo clásico"))])
    if tab == "crt-heatmap":
        return card([dcc.Graph(figure=build_heatmap(mat_classic, "Matriz de payoffs — Torneo clásico"))])
    if tab == "crt-scatter":
        return card([dcc.Graph(figure=build_coop_scatter(df_classic, "Cooperación vs Score — Clásico"))])
    if tab == "crt-individual":
        return card([
            html.Label("Seleccionar estrategia:", style=dict(color=TEXT_SEC, fontSize="12px")),
            dcc.Dropdown(id="ind-strat-classic", options=[{"label":s,"value":s} for s in STRATS],
                         value="TitForTat", clearable=False, style=dict(marginBottom="12px")),
            html.Div(id="ind-classic-detail"),
        ])
    if tab == "crt-stats":
        anova = DATA["classic"]["anova"]
        tukey_df = pd.DataFrame(anova["tukey"])
        return html.Div([
            card([
                section_title("ANOVA — Análisis de varianza", "H₀: todas las estrategias tienen el mismo score promedio"),
                html.Div([
                    html.Div([html.Div(f"{anova['F']:.3f}", className="metric-val"), html.Div("F-statistic", className="metric-lbl")], className="metric-card"),
                    html.Div([html.Div(f"{anova['p']:.2e}", className="metric-val"),
                              html.Div("p-value", className="metric-lbl"),
                              html.Div("Rechazar H₀ ✓" if anova['p']<0.05 else "No rechazar H₀", style=dict(color=ACCENT2 if anova['p']<0.05 else "#ffa726", fontSize="11px"))
                              ], className="metric-card"),
                ], style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="12px", marginBottom="16px")),
                dcc.Graph(figure=build_boxplot(df_classic, "Distribución de scores por estrategia")),
            ]),
            card([
                section_title("Test de Welch (proxy Tukey HSD)", "Pares con diferencias estadísticamente significativas (p < 0.05)"),
                dash_table.DataTable(
                    data=tukey_df[tukey_df["reject"]==True].to_dict("records"),
                    columns=[{"name":c,"id":c} for c in ["group1","group2","meandiff","p-adj"]],
                    style_table=dict(overflowX="auto"),
                    style_cell=dict(background=CARD_BG, color=TEXT_PRI, border=f"1px solid {BORDER}", fontSize="12px", padding="8px"),
                    style_header=dict(background=DARK_BG, fontWeight=600, color=TEXT_SEC),
                )
            ]),
        ])
    if tab == "crt-code":
        return card([html.Div(CODE_SOURCE["Torneo Round-Robin"], className="code-block")])

@app.callback(Output("ind-classic-detail","children"), Input("ind-strat-classic","value"))
def ind_classic(s):
    if not s: return html.Div()
    sub = df_classic[df_classic["Strategy_A"]==s]
    fig = go.Figure()
    for opp in STRATS:
        vals = sub[sub["Strategy_B"]==opp]["Score_A"]
        fig.add_trace(go.Bar(name=opp, x=[opp], y=[vals.mean()],
                             marker_color=STRAT_COLOR.get(opp, ACCENT),
                             hovertemplate=f"vs {opp}: %{{y:.1f}}<extra></extra>"))
    fig.update_layout(**PLOTLY_LAYOUT, height=350, showlegend=False,
                      title=dict(text=f"{s} — score vs cada oponente", font=dict(size=13,color=TEXT_PRI)),
                      xaxis=dict(tickangle=-35), yaxis_title="Score promedio")
    return dcc.Graph(figure=fig)

@app.callback(Output("tariff-history-chart","figure"), Input("sector-select","value"))
def update_tariff_chart(code):
    return build_tariff_history(int(code))

@app.callback(Output("sector-payoff-display","children"), Input("sector-select","value"))
def sector_payoff(code):
    d = DATA["real"]["bilateral"].get(str(code), {})
    if not d: return html.Div()
    return html.Div([
        html.Div([
            html.Div([html.Strong("USA→MEX", style=dict(color=TEXT_SEC, fontSize="11px"))]),
            html.Div([html.Span("Coop (AHS): ", style=dict(color=TEXT_SEC, fontSize="12px")),
                      html.Span(f"{d['coop_usa']:.2f}%", style=dict(color=ACCENT2, fontWeight=600))]),
            html.Div([html.Span("Traición (BND): ", style=dict(color=TEXT_SEC, fontSize="12px")),
                      html.Span(f"{d['trai_usa']:.2f}%", style=dict(color="#ef5350", fontWeight=600))]),
        ], style=dict(marginBottom="8px")),
        html.Div([
            html.Div([html.Strong("MEX→USA", style=dict(color=TEXT_SEC, fontSize="11px"))]),
            html.Div([html.Span("Coop (AHS): ", style=dict(color=TEXT_SEC, fontSize="12px")),
                      html.Span(f"{d['coop_mex']:.2f}%", style=dict(color=ACCENT2, fontWeight=600))]),
            html.Div([html.Span("Traición (BND): ", style=dict(color=TEXT_SEC, fontSize="12px")),
                      html.Span(f"{d['trai_mex']:.2f}%", style=dict(color="#ef5350", fontWeight=600))]),
        ]),
    ], style=dict(background="#12141c", padding="12px", borderRadius="6px", fontSize="12px"))

@app.callback(Output("real-result-content","children"), Input("real-result-tabs","value"))
def real_results(tab):
    if tab == "rrt-ranking":
        return card([dcc.Graph(figure=build_ranking_bar(rank_real, "Ranking — Caso real MEX–USA (w=9.5%)"))])
    if tab == "rrt-heatmap":
        return card([dcc.Graph(figure=build_heatmap(mat_real, "Matriz de payoffs — Caso real MEX–USA"))])
    if tab == "rrt-stats":
        anova = DATA["real"]["anova"]
        return html.Div([
            card([
                section_title("ANOVA — Caso real"),
                html.Div([
                    html.Div([html.Div(f"{anova['F']:.3f}", className="metric-val"), html.Div("F-statistic", className="metric-lbl")], className="metric-card"),
                    html.Div([html.Div(f"{anova['p']:.2e}", className="metric-val"), html.Div("p-value", className="metric-lbl")], className="metric-card"),
                ], style=dict(display="grid", gridTemplateColumns="1fr 1fr", gap="12px", marginBottom="16px")),
                dcc.Graph(figure=build_boxplot(df_real, "Distribución de scores — Caso real")),
            ])
        ])
    if tab == "rrt-conclusions":
        top3 = list(rank_real.head(3).index)
        bot3 = list(rank_real.tail(3).index)
        return card([
            section_title("Conclusiones — Relación comercial MEX–USA"),
            html.Div([
                html.Div([
                    html.H5("¿Qué estrategias dominan bajo aranceles reales?", style=dict(color=ACCENT, marginBottom="8px")),
                    html.P(f"Las estrategias con mejor desempeño son {', '.join(top3)}. Esto es consistente con la interpretación geopolítica: en una relación bilateral de larga duración como la del T-MEC, los actores que se adaptan al comportamiento histórico del oponente (Adaptive, PSOPlayer) o que responden proporcionalmente (TwoTitsForTat) obtienen mejores resultados que los puristas del castigo permanente.", style=dict(color=TEXT_PRI, fontSize="13px", lineHeight="1.8")),
                ], style=dict(marginBottom="16px")),
                html.Div([
                    html.H5("¿Por qué cambia el ranking vs el torneo clásico?", style=dict(color=ACCENT, marginBottom="8px")),
                    html.P("El ruido político (w=0.095) castiga severamente a GrimTrigger y Friedman: una sola acción hostil por 'error diplomático' activa el castigo permanente. En contexto real, la flexibilidad y el olvido selectivo son más valiosos que la credibilidad absoluta del castigo.", style=dict(color=TEXT_PRI, fontSize="13px", lineHeight="1.8")),
                ], style=dict(marginBottom="16px")),
                html.Div([
                    html.H5("Implicación para política comercial", style=dict(color=ACCENT, marginBottom="8px")),
                    html.P(f"Las estrategias con peor desempeño ({', '.join(bot3)}) confirman que la cooperación unilateral irrestricta (AlwaysCooperate en bajo ruido) o la defección pura son subóptimas. El T-MEC funciona como mecanismo de compromiso que eleva δ (shadow of the future) y hace que la cooperación sea el equilibrio dominante — siempre que los payoffs de traición (BND) sean lo suficientemente costosos.", style=dict(color=TEXT_PRI, fontSize="13px", lineHeight="1.8")),
                ]),
            ]),
        ])
    if tab == "rrt-code":
        return card([html.Div(CODE_SOURCE["Payoffs WITS"], className="code-block")])

@app.callback(Output("sim-store","data"), Output("sim-results","children"),
              Input("sim-run-btn","n_clicks"),
              State("sim-strats","value"), State("sim-rounds","value"),
              State("sim-repeats","value"), State("sim-w","value"),
              State("sim-T","value"), State("sim-R","value"),
              State("sim-P","value"), State("sim-S","value"),
              prevent_initial_call=True)
def run_simulation(n, strats, rounds, repeats, w, T, R, P, S):
    if not strats or len(strats) < 2:
        return {}, card([html.P("Selecciona al menos 2 estrategias.", style=dict(color="#ef5350"))])
    T = float(T or 5); R = float(R or 3); P = float(P or 1); S = float(S or 0)
    if not (T > R > P >= S):
        return {}, card([html.P("Los payoffs deben cumplir T > R > P ≥ S.", style=dict(color="#ef5350"))])
    df_sim, mat_sim, rank_sim = run_live_tournament(strats, int(rounds), int(repeats), float(w), T, R, P, S)
    groups = [grp["Score_A"].values for _, grp in df_sim.groupby("Strategy_A")]
    F_val, p_val = stats.f_oneway(*groups) if len(groups)>1 else (0,1)
    result_data = df_sim.to_dict("records")
    content = html.Div([
        html.Div([
            html.Div([html.Div(rank_sim.index[0], className="metric-val"), html.Div("Estrategia ganadora", className="metric-lbl")], className="metric-card"),
            html.Div([html.Div(f"{rank_sim.iloc[0]:.1f}", className="metric-val"), html.Div("Score máximo", className="metric-lbl")], className="metric-card"),
            html.Div([html.Div(f"{F_val:.2f}", className="metric-val"), html.Div("F-stat ANOVA", className="metric-lbl")], className="metric-card"),
            html.Div([html.Div(f"{p_val:.2e}", className="metric-val"), html.Div("p-value", className="metric-lbl")], className="metric-card"),
        ], style=dict(display="grid", gridTemplateColumns="repeat(4,1fr)", gap="12px", marginBottom="16px")),
        html.Div([
            html.Div([card([dcc.Graph(figure=build_ranking_bar(rank_sim, "Ranking — Simulación"))])], style=dict(flex=1)),
            html.Div([card([dcc.Graph(figure=build_heatmap(mat_sim, "Heatmap — Simulación"))])], style=dict(flex=1)),
        ], style=dict(display="flex", gap="16px")),
        card([dcc.Graph(figure=build_boxplot(df_sim, "Distribución — Simulación"))]),
        html.Div([
            html.Button("⬇ Exportar CSV", id="sim-export-csv",
                        style=dict(background=ACCENT+"33", color=ACCENT, border=f"1px solid {ACCENT}44",
                                   borderRadius="6px", padding="8px 16px", cursor="pointer", marginRight="8px")),
            dcc.Download(id="sim-download"),
        ], style=dict(marginTop="8px")),
    ])
    return result_data, content

@app.callback(Output("sim-download","data"), Input("sim-export-csv","n_clicks"),
              State("sim-store","data"), prevent_initial_call=True)
def export_sim(n, data):
    if not data: return None
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "simulacion_resultados.csv", index=False)

@app.callback(Output("analysis-strat-detail","children"), Input("analysis-strat","value"))
def analysis_strat(s):
    if not s: return html.Div()
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Score vs oponente (Clásico)", "Score vs oponente (Real)"])
    for col, df_, rank_ in [(1, df_classic, rank_classic), (2, df_real, rank_real)]:
        sub = df_[df_["Strategy_A"]==s].groupby("Strategy_B")["Score_A"].mean().sort_values(ascending=False)
        fig.add_trace(go.Bar(x=sub.index, y=sub.values,
                             marker_color=[STRAT_COLOR.get(o,ACCENT) for o in sub.index],
                             showlegend=False,
                             hovertemplate="%{x}: %{y:.1f}<extra></extra>"), row=1, col=col)
    fig.update_layout(**PLOTLY_LAYOUT, height=380,
                      title=dict(text=f"{s} — desempeño detallado", font=dict(size=14,color=TEXT_PRI)))
    fig.update_xaxes(tickangle=-35, tickfont=dict(size=9))
    return card([dcc.Graph(figure=fig)])

@app.callback(Output("annex-sub-content","children"), Input("annex-sub","value"))
def annex_sub(val):
    if val == "ax-rng":    return render_rng_panel()
    if val == "ax-export": return render_export_panel()
    if val == "ax-code":   return render_code_panel()

@app.callback(Output("code-source-display","children"), Input("code-tabs","value"))
def show_code_source(k):
    return html.Div(CODE_SOURCE.get(k,""), className="code-block")

@app.callback(Output("download-csv-classic","data"), Input("btn-csv-classic","n_clicks"), prevent_initial_call=True)
def dl_csv_classic(_): return dcc.send_data_frame(df_classic.to_csv, "torneo_clasico_detallado.csv", index=False)

@app.callback(Output("download-json-classic","data"), Input("btn-json-classic","n_clicks"), prevent_initial_call=True)
def dl_json_classic(_):
    out = {"ranking": rank_classic.to_dict(), "anova": DATA["classic"]["anova"],
           "payoffs": DATA["classic"]["payoffs"]}
    return dict(content=json.dumps(out, indent=2), filename="torneo_clasico_resultados.json")

@app.callback(Output("download-csv-real","data"), Input("btn-csv-real","n_clicks"), prevent_initial_call=True)
def dl_csv_real(_): return dcc.send_data_frame(df_real.to_csv, "torneo_real_detallado.csv", index=False)

@app.callback(Output("download-json-real","data"), Input("btn-json-real","n_clicks"), prevent_initial_call=True)
def dl_json_real(_):
    out = {"ranking": rank_real.to_dict(), "anova": DATA["real"]["anova"],
           "payoffs": DATA["real"]["payoffs"], "bilateral": DATA["real"]["bilateral"],
           "impact": DATA["real"]["impact"]}
    return dict(content=json.dumps(out, indent=2), filename="torneo_real_resultados.json")

if __name__ == "__main__":
    app.run_server(debug=False, port=8050)
