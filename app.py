"""
Lotto-Lab Dashboard
Interactive, gamified lottery prediction system
"""

import dash
from dash import html, dcc, callback, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import json

import lotto_engine as engine
from fetch_sa_lotto import fetch_current_lotto_jackpot

LIVE_JACKPOT = fetch_current_lotto_jackpot(default=5_000_000.0)
print(f"[Dashboard] Live jackpot fetched: R{LIVE_JACKPOT:,.2f}")

MOBILE_CSS = """
@media (max-width: 576px) {
    .hero-title { font-size: 1.4rem !important; }
    .hero-subtitle { font-size: 1.1rem !important; }
    .hero-desc { font-size: 0.85rem !important; }
    .ball-el { width: 36px !important; height: 36px !important; font-size: 0.85rem !important; margin: 2px !important; }
    .ball-sm { width: 30px !important; height: 30px !important; font-size: 0.75rem !important; margin: 1px !important; }
    .ticket-meta { flex-direction: column !important; gap: 6px; }
    .ticket-meta > div { width: 100% !important; }
    .countdown-box { min-width: 52px !important; padding: 6px 8px !important; }
    .countdown-num { font-size: 1.4rem !important; }
    .countdown-sep { font-size: 1.2rem !important; margin: 0 3px !important; }
    .affiliate-btn { padding: 6px 12px !important; font-size: 0.8rem !important; }
    .affiliate-link { margin: 2px !important; }
    .recent-draw-row { flex-direction: column !important; gap: 6px; }
    .recent-draw-date { width: 100% !important; flex: none !important; max-width: 100% !important; }
    .recent-draw-balls { width: 100% !important; flex: none !important; max-width: 100% !important; }
    .recent-draw-id { display: none !important; }
    .info-card-title { font-size: 0.7rem !important; }
    .info-card-value { font-size: 1.5rem !important; }
    .info-card-desc { font-size: 0.6rem !important; }
    .section-heading { font-size: 1.3rem !important; }
    .unlock-form { max-width: 100% !important; }
    .unlock-input-group { flex-direction: column !important; }
    .unlock-input-group > * { border-radius: 6px !important; margin-bottom: 4px; }
    .navbar-brand { font-size: 1.1rem !important; }
    .container-fluid { padding-left: 12px !important; padding-right: 12px !important; }
}
@media (max-width: 400px) {
    .ball-el { width: 32px !important; height: 32px !important; font-size: 0.8rem !important; }
    .ball-sm { width: 28px !important; height: 28px !important; font-size: 0.7rem !important; }
    .countdown-box { min-width: 45px !important; padding: 4px 6px !important; }
    .countdown-num { font-size: 1.2rem !important; }
}
"""

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="Lotto-Lab",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

server = app.server
import os
server.secret_key = os.environ.get("SESSION_SECRET", "lotto-lab-dev-secret-change-me")

COLORS = {
    "primary": "#00d4ff",
    "secondary": "#ff6b6b",
    "success": "#51cf66",
    "warning": "#ffd43b",
    "dark": "#1a1a2e",
    "card": "#16213e",
    "text": "#e8e8e8",
    "gold": "#FFD700",
    "neon_green": "#39FF14",
    "subtle": "#8899aa",
}

draws, n_max, k = engine.load_data()

initial_result = engine.generate_predictions(draws, n_max, k, LIVE_JACKPOT, n_samples=8000, top_n=12)
print(f"[Dashboard] Initial predictions generated: {len(initial_result.get('tickets', []))} tickets")

BALL_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "48px",
    "height": "48px",
    "borderRadius": "50%",
    "fontWeight": "bold",
    "fontSize": "1.1rem",
    "margin": "3px",
    "boxShadow": "0 4px 15px rgba(0,0,0,0.3), inset 0 2px 4px rgba(255,255,255,0.2)",
}

def ball_element(number, color_bg="#00d4ff", color_text="#000", size="48px"):
    css_class = "ball-sm" if size in ("38px", "30px") else "ball-el"
    style = {
        **BALL_STYLE,
        "width": size,
        "height": size,
        "background": f"radial-gradient(circle at 35% 35%, {color_bg}, {color_bg}aa)",
        "color": color_text,
        "border": f"2px solid {color_bg}",
    }
    return html.Div(str(number), style=style, className=css_class)

def hot_ball(number):
    return ball_element(number, "#ff4444", "#fff")

def cold_ball(number):
    return ball_element(number, "#4488ff", "#fff")

def ticket_ball(number):
    return ball_element(number, "#FFD700", "#000")

def section_copy(text, className="text-muted small mb-3", style=None):
    default_style = {"maxWidth": "700px", "lineHeight": "1.6", "color": COLORS["subtle"]}
    if style:
        default_style.update(style)
    return html.P(text, className=className, style=default_style)

def create_navbar():
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col(html.I(className="fas fa-dice fa-lg me-2", style={"color": COLORS["gold"]})),
                dbc.Col(dbc.NavbarBrand("Lotto-Lab", className="fs-4 fw-bold")),
            ], align="center", className="g-0"),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Predictions", href="#predictions", className="nav-link-custom")),
                    dbc.NavItem(dbc.NavLink("Analytics", href="#analytics", className="nav-link-custom")),
                    dbc.NavItem(dbc.NavLink("Performance", href="#performance", className="nav-link-custom")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4",
        sticky="top",
    )

def get_next_draw_info():
    now = datetime.now()
    weekday = now.weekday()
    targets = [2, 5]
    candidates = []
    for t in targets:
        days_ahead = (t - weekday) % 7
        if days_ahead == 0:
            candidates.append(7)
        else:
            candidates.append(days_ahead)
    next_days = min(candidates)
    next_draw = now.replace(hour=20, minute=56, second=0, microsecond=0) + timedelta(days=next_days)
    draw_name = "Saturday" if next_draw.weekday() == 5 else "Wednesday"
    return next_draw.timestamp(), draw_name

def format_jackpot(amount):
    if amount >= 1_000_000:
        return f"R{amount/1_000_000:.1f}M"
    return f"R{amount:,.0f}"

def ev_flavour(intensity):
    if intensity > 0.7:
        return "Tonight looks... lively."
    elif intensity > 0.4:
        return "The patterns are getting interesting."
    else:
        return "The data looks quiet tonight."

def build_info_cards(result):
    jackpot = result["jackpot"]
    intensity = result["intensity"]
    mode = result["mode"]
    alpha = result["alpha"]

    return dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-coins fa-2x mb-2", style={"color": COLORS["gold"]}),
                    html.H6("Current Jackpot", className="text-muted mb-1 info-card-title"),
                    html.H2(format_jackpot(jackpot), className="mb-0 fw-bold info-card-value", style={"color": COLORS["gold"]}),
                    html.Small(
                        "Sometimes it's fun to look at the numbers a little closer before picking yours.",
                        className="text-muted d-block mt-2 info-card-desc d-none d-md-block",
                        style={"fontSize": "0.7rem", "lineHeight": "1.4"},
                    ),
                ], className="text-center"),
            ]),
        ], className="h-100 shadow", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-chess fa-2x mb-2", style={"color": COLORS["primary"]}),
                    html.H6("Strategy Mode", className="text-muted mb-1 info-card-title"),
                    html.H5(mode.replace(" -> ", " \u2192 ").replace("HYBRID \u2192 JACKPOT_HUNTER (mixed)", "Jackpot Hunter"),
                             className="mb-1 fw-bold info-card-value", style={"color": COLORS["primary"], "fontSize": "clamp(0.8rem, 2.5vw, 1.1rem)"}),
                    html.Small(f"Blend: {alpha:.0%}", className="text-muted d-block"),
                    html.Small(
                        "Different approaches reveal different possibilities.",
                        className="text-muted d-block mt-1 d-none d-md-block",
                        style={"fontSize": "0.7rem"},
                    ),
                ], className="text-center"),
            ]),
        ], className="h-100 shadow", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-bolt fa-2x mb-2", style={"color": COLORS["success"]}),
                    html.H6("EV Intensity", className="text-muted mb-1 info-card-title"),
                    html.H2(f"{intensity:.1%}", className="mb-0 fw-bold info-card-value", style={"color": COLORS["success"]}),
                    html.Small(
                        ev_flavour(intensity),
                        className="d-block mt-2 d-none d-md-block",
                        style={"fontSize": "0.7rem", "color": COLORS["success"], "fontStyle": "italic"},
                    ),
                ], className="text-center"),
            ]),
        ], className="h-100 shadow", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardBody([
                html.Div([
                    html.I(className="fas fa-database fa-2x mb-2", style={"color": COLORS["secondary"]}),
                    html.H6("Historical Draws", className="text-muted mb-1 info-card-title"),
                    html.H2(f"{len(draws):,}", className="mb-0 fw-bold info-card-value", style={"color": COLORS["secondary"]}),
                    html.Small(
                        "The more history you look at, the more patterns start to emerge.",
                        className="text-muted d-block mt-2 d-none d-md-block",
                        style={"fontSize": "0.7rem", "lineHeight": "1.4"},
                    ),
                ], className="text-center"),
            ]),
        ], className="h-100 shadow", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
    ], className="mb-4")

def build_hot_cold(result):
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-fire me-2", style={"color": "#ff4444"}),
                    html.Span("Hot Numbers", className="fw-bold"),
                ], style={"backgroundColor": COLORS["dark"]}),
                dbc.CardBody([
                    html.P(
                        "These numbers have been showing up more often than usual lately. Some players like to ride the streak. Others avoid them completely. But when a number gets hot... people tend to notice.",
                        className="small mb-3",
                        style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                    ),
                    html.Div(
                        [hot_ball(n) for n in result["hot_numbers"]],
                        className="d-flex flex-wrap justify-content-center gap-1",
                    ),
                ]),
            ], className="shadow", style={"backgroundColor": COLORS["card"]}),
        ], md=6, className="mb-3"),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="fas fa-snowflake me-2", style={"color": "#4488ff"}),
                    html.Span("Cold Numbers", className="fw-bold"),
                ], style={"backgroundColor": COLORS["dark"]}),
                dbc.CardBody([
                    html.P(
                        "Cold numbers haven't appeared for a while. In probability terms, that doesn't mean they're \"due.\" But it does make them interesting to watch. Sometimes the quiet ones come back with a bang.",
                        className="small mb-3",
                        style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                    ),
                    html.Div(
                        [cold_ball(n) for n in result["cold_numbers"]],
                        className="d-flex flex-wrap justify-content-center gap-1",
                    ),
                ]),
            ], className="shadow", style={"backgroundColor": COLORS["card"]}),
        ], md=6, className="mb-3"),
    ], className="mb-3")

def confidence_label(score, best_score):
    if best_score == 0:
        return "---", "secondary"
    ratio = abs(score / best_score) if best_score != 0 else 0
    if ratio >= 0.98:
        return "BEST PICK", "success"
    elif ratio >= 0.95:
        return "STRONG", "primary"
    elif ratio >= 0.90:
        return "GOOD", "info"
    else:
        return "VALUE", "warning"

def build_ticket_card(i, score, ticket, best_score, last_draw_nums, blur_numbers=False):
    label, color = confidence_label(score, best_score)
    matches = set(ticket) & set(last_draw_nums)
    match_count = len(matches)

    balls = []
    for n in sorted(ticket):
        if n in matches:
            balls.append(ball_element(n, "#39FF14", "#000"))
        else:
            balls.append(ticket_ball(n))

    medal = ""
    if i == 1:
        medal = " \U0001f947"
    elif i == 2:
        medal = " \U0001f948"
    elif i == 3:
        medal = " \U0001f949"

    match_badge = ""
    if match_count >= 2:
        badge_col = "warning" if match_count >= 4 else "success" if match_count >= 3 else "secondary"
        match_badge = dbc.Badge(
            f"{match_count} matched last draw!",
            color=badge_col,
            className="ms-2",
        )

    balls_style = {"filter": "blur(6px)", "userSelect": "none"} if blur_numbers else {}

    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div([
                    html.Span(f"Ticket #{i}{medal}", className="fw-bold fs-6"),
                    dbc.Badge(label, color=color, className="ms-2"),
                    match_badge if match_badge else None,
                ], className="d-flex align-items-center flex-wrap"),
                html.Div(
                    balls,
                    className="d-flex flex-wrap justify-content-center my-2",
                    style=balls_style,
                ),
                html.Div([
                    html.Small("Pattern Strength", className="text-muted d-block mb-1",
                               style={"fontSize": "0.7rem"}),
                    dbc.Progress(
                        value=max(10, min(100, abs(score / best_score) * 100)) if best_score != 0 else 50,
                        color=color,
                        className="mb-0",
                        style={"height": "8px"},
                    ),
                ]),
            ], className="ticket-meta"),
        ], className="py-2"),
    ], className="mb-2 shadow-sm", style={
        "backgroundColor": COLORS["card"],
        "border": f"1px solid {'#FFD700' if i <= 3 else '#333'}",
        "borderLeft": f"4px solid {'#FFD700' if i == 1 else '#C0C0C0' if i == 2 else '#CD7F32' if i == 3 else '#555'}",
    })

def build_tickets_section(result):
    tickets = result.get("tickets", [])
    if not tickets:
        return html.P("No tickets generated", className="text-muted text-center")

    last_draw = draws[0].numbers if draws else []
    best_score = tickets[0][0] if tickets else 1

    free_cards = []
    for i, (score, ticket) in enumerate(tickets[:2], 1):
        free_cards.append(build_ticket_card(i, score, ticket, best_score, last_draw, blur_numbers=True))

    locked_cards = []
    for i, (score, ticket) in enumerate(tickets[2:12], 3):
        locked_cards.append(build_ticket_card(i, score, ticket, best_score, last_draw, blur_numbers=True))

    hidden_count = min(len(tickets) - 2, 10)

    return html.Div([
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Span([
                        html.I(className="fas fa-ticket-alt me-2", style={"color": COLORS["gold"]}),
                        "Your Generated Tickets",
                    ], className="fw-bold fs-5"),
                    dbc.Badge(f"{len(tickets)} generated", color="primary", className="ms-2"),
                ], className="d-flex align-items-center"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.Div([
                    html.P(
                        "This is where the data turns into playable combinations. The engine mixes trending numbers, quieter numbers, balanced spreads across the number field, and historical distribution patterns.",
                        className="small mb-1",
                        style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                    ),
                    html.P([
                        html.I(className="fas fa-info-circle me-1"),
                        "Each ticket includes a pattern strength indicator. It's not a prediction. Just a signal from the model saying: ",
                        html.Em("\"this one looks interesting.\""),
                    ], className="small mb-3", style={"color": "#aabbcc", "lineHeight": "1.5"}),
                ], className="mb-3"),

                dcc.Store(id="tickets-store", data=[
                    {"score": score, "ticket": ticket} for score, ticket in tickets[:12]
                ]),

                html.Div(free_cards, id="preview-tickets-wrapper"),

                html.Div([
                    html.Div(locked_cards),

                    html.Div([
                        html.Hr(style={"borderColor": "#444"}),
                        html.Div([
                            html.I(className="fas fa-lock me-2", style={"color": COLORS["gold"]}),
                            html.Span(
                                f"Unlock all {hidden_count} ticket numbers",
                                className="fw-bold",
                                style={"color": COLORS["text"]},
                            ),
                        ], className="text-center mb-2"),
                        html.P(
                            "Join the Lotto-Lab inner circle to reveal your full ticket set before every draw.",
                            className="text-center mb-3",
                            style={"color": COLORS["subtle"], "fontSize": "0.85rem"},
                        ),
                        html.Div([
                            dbc.InputGroup([
                                dbc.InputGroupText(
                                    html.I(className="fas fa-phone", style={"color": COLORS["gold"]}),
                                    style={"backgroundColor": COLORS["card"], "borderColor": "#555"},
                                    className="d-none d-sm-flex",
                                ),
                                dbc.Input(
                                    id="email-input",
                                    type="tel",
                                    placeholder="Your mobile number",
                                    style={"backgroundColor": "#0d1b2a", "color": "#fff", "borderColor": "#555"},
                                ),
                                dbc.Button([
                                    html.I(className="fas fa-unlock me-2"),
                                    "Reveal",
                                ], id="unlock-btn", color="warning", className="fw-bold"),
                            ], className="mb-2 unlock-input-group"),
                            html.Div(id="email-error", style={"minHeight": "20px"}),
                            html.Div([
                                html.P("or", className="text-center mb-2",
                                       style={"color": "#556677", "fontSize": "0.8rem"}),
                                dbc.Button([
                                    html.I(className="fab fa-telegram me-2"),
                                    "Join Our Telegram Group",
                                ], id="telegram-btn", color="primary", outline=True,
                                   className="w-100 mb-2",
                                   style={"borderColor": "#0088cc", "color": "#0088cc"}),
                            ]),
                            html.P([
                                html.I(className="fas fa-shield-alt me-1"),
                                "We only send picks before each draw. No spam.",
                            ], className="text-center mb-0",
                               style={"color": "#667788", "fontSize": "0.75rem"}),
                        ], style={"maxWidth": "450px", "margin": "0 auto"}, className="unlock-form"),
                    ], className="py-3"),
                ], id="locked-tickets-wrapper"),

                html.Div(id="unlocked-tickets", style={"display": "none"}),
            ]),
        ], className="shadow", style={"backgroundColor": COLORS["card"]}),
    ])

def build_recent_draws(result):
    recent = engine.get_recent_draws(draws, 8)
    tickets = result.get("tickets", [])

    rows = []
    for i, draw in enumerate(recent):
        is_latest = (i == 0)
        draw_nums = draw["numbers"]
        draw_set = set(draw_nums)

        # Correct metric: best single-ticket match (same as backtest)
        best_match_count = 0
        best_matched_nums = set()
        for _, ticket in tickets[:12]:
            matched = draw_set & set(ticket)
            if len(matched) > best_match_count:
                best_match_count = len(matched)
                best_matched_nums = matched

        draw_balls = []
        for n in draw_nums:
            if n in best_matched_nums:
                draw_balls.append(ball_element(n, "#39FF14", "#000", "38px"))
            elif is_latest:
                draw_balls.append(ball_element(n, "#FFD700", "#000", "38px"))
            else:
                draw_balls.append(ball_element(n, "#00d4ff", "#000", "38px"))

        match_badge = None
        if best_match_count >= 2:
            badge_color = "warning" if best_match_count >= 4 else "success" if best_match_count >= 3 else "secondary"
            match_badge = dbc.Badge(
                f"best ticket: {best_match_count} matched",
                color=badge_color,
                className="ms-2",
                style={"fontSize": "0.65rem"},
            )

        rows.append(
            dbc.ListGroupItem([
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dbc.Badge("LATEST", color="success", className="me-2") if is_latest else None,
                            html.Small(draw["date"], className="text-muted"),
                            match_badge,
                        ]),
                    ], xs=12, md=3, className="mb-1 mb-md-0 recent-draw-date"),
                    dbc.Col([
                        html.Div(
                            draw_balls,
                            className="d-flex flex-wrap gap-1",
                        ),
                    ], xs=12, md=8, className="recent-draw-balls"),
                    dbc.Col([
                        html.Small(f"#{draw['draw_id']}", className="text-muted"),
                    ], width="auto", className="text-end d-none d-md-block recent-draw-id"),
                ], align="center", className="recent-draw-row"),
            ], style={"backgroundColor": "transparent", "borderColor": "#333"})
        )
    return dbc.ListGroup(rows, flush=True)

def create_frequency_chart(result):
    freq_scores = result.get("freq_scores", {})
    numbers = list(range(1, n_max + 1))
    freqs = [freq_scores.get(str(i), freq_scores.get(i, 0.02)) for i in numbers]
    hot_set = set(result.get("hot_numbers", []))
    cold_set = set(result.get("cold_numbers", []))

    colors = []
    for i, n in enumerate(numbers):
        if n in hot_set:
            colors.append("#ff4444")
        elif n in cold_set:
            colors.append("#4488ff")
        elif freqs[i] > 0.022:
            colors.append(COLORS["primary"])
        else:
            colors.append("#555")

    fig = go.Figure(data=[
        go.Bar(
            x=numbers,
            y=freqs,
            marker_color=colors,
            hovertemplate="Ball %{x}<br>Frequency: %{y:.4f}<extra></extra>",
        )
    ])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        xaxis=dict(title="Ball Number", gridcolor="#333", tickmode="linear", dtick=5),
        yaxis=dict(title="Frequency", gridcolor="#333"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=300,
    )
    return fig

def create_heatmap(result):
    probs = result.get("num_probs", {})
    if not probs:
        return go.Figure()

    labels = []
    vals = []
    for i in range(1, n_max + 1):
        labels.append(str(i))
        vals.append(probs.get(str(i), probs.get(i, 0)))

    grid, label_grid = [], []
    row_v, row_l = [], []
    for i in range(len(vals)):
        row_v.append(vals[i])
        row_l.append(labels[i])
        if (i + 1) % 13 == 0:
            grid.append(row_v)
            label_grid.append(row_l)
            row_v, row_l = [], []
    if row_v:
        while len(row_v) < 13:
            row_v.append(0)
            row_l.append("")
        grid.append(row_v)
        label_grid.append(row_l)

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        text=label_grid,
        texttemplate="%{text}",
        colorscale="Viridis",
        showscale=True,
        hovertemplate="Ball %{text}<br>Probability: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False, autorange="reversed"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=250,
    )
    return fig


def build_full_page(result):
    freq_fig = create_frequency_chart(result)
    heat_fig = create_heatmap(result)

    next_draw_ts, draw_name = get_next_draw_info()

    countdown_box_style = {
        "display": "inline-flex", "flexDirection": "column", "alignItems": "center",
        "backgroundColor": "rgba(255, 212, 59, 0.1)", "borderRadius": "8px",
        "padding": "8px 14px", "minWidth": "60px",
        "border": "1px solid rgba(255, 212, 59, 0.2)",
    }
    countdown_num_style = {
        "color": COLORS["warning"], "fontSize": "2rem", "fontWeight": "800",
        "lineHeight": "1", "fontFamily": "monospace",
    }
    countdown_label_style = {
        "color": "#8899aa", "fontSize": "0.65rem", "textTransform": "uppercase",
        "letterSpacing": "1px", "marginTop": "4px",
    }
    countdown_sep_style = {
        "color": COLORS["warning"], "fontSize": "1.8rem",
        "fontWeight": "800", "margin": "0 6px", "alignSelf": "center",
    }

    return html.Div([
        build_info_cards(result),

        dcc.Interval(id="countdown-interval", interval=1000, n_intervals=0),
        dcc.Store(id="next-draw-ts", data=next_draw_ts),

        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2",
                           style={"color": "#ff4444", "fontSize": "1.2rem"}),
                    html.Span("DRAW CLOSING SOON", className="fw-bold",
                              style={"color": "#ff4444", "fontSize": "0.85rem",
                                     "letterSpacing": "2px", "textTransform": "uppercase"}),
                ], className="d-flex align-items-center justify-content-center mb-3"),
                html.Div([
                    html.Div([
                        html.Span("--", id="cd-days", style=countdown_num_style, className="countdown-num"),
                        html.Span("DAYS", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                    html.Span(":", style=countdown_sep_style, className="countdown-sep"),
                    html.Div([
                        html.Span("--", id="cd-hours", style=countdown_num_style, className="countdown-num"),
                        html.Span("HRS", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                    html.Span(":", style=countdown_sep_style, className="countdown-sep"),
                    html.Div([
                        html.Span("--", id="cd-mins", style=countdown_num_style, className="countdown-num"),
                        html.Span("MIN", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                    html.Span(":", style=countdown_sep_style, className="countdown-sep"),
                    html.Div([
                        html.Span("--", id="cd-secs", style=countdown_num_style, className="countdown-num"),
                        html.Span("SEC", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                ], className="d-flex align-items-center justify-content-center flex-wrap mb-2"),
                html.P(
                    f"until {draw_name} draw",
                    className="text-center mb-2",
                    style={"color": "#8899aa", "fontSize": "0.85rem"},
                ),
                html.P(
                    "Your tickets are ready. Don\u2019t miss this draw.",
                    className="text-center mb-3",
                    style={"color": COLORS["text"], "fontSize": "0.95rem", "fontWeight": "500"},
                ),
                html.Div([
                    html.A(
                        dbc.Button([
                            html.Span("Betway", className="fw-bold"),
                        ], color="success", className="px-3 py-2 fw-bold affiliate-btn",
                           style={"fontSize": "0.9rem", "borderRadius": "6px",
                                  "backgroundColor": "#00a651", "borderColor": "#00a651"}),
                        href="#", target="_blank", id="affiliate-betway",
                        className="text-decoration-none mx-1 affiliate-link",
                    ),
                    html.A(
                        dbc.Button([
                            html.Span("Hollywoodbets", className="fw-bold"),
                        ], color="danger", className="px-3 py-2 fw-bold affiliate-btn",
                           style={"fontSize": "0.9rem", "borderRadius": "6px",
                                  "backgroundColor": "#800020", "borderColor": "#800020"}),
                        href="#", target="_blank", id="affiliate-hollywood",
                        className="text-decoration-none mx-1 affiliate-link",
                    ),
                    html.A(
                        dbc.Button([
                            html.Span("SA Lottery", className="fw-bold"),
                        ], color="warning", className="px-3 py-2 fw-bold affiliate-btn",
                           style={"fontSize": "0.9rem", "borderRadius": "6px",
                                  "color": "#000"}),
                        href="https://www.nationallottery.co.za", target="_blank",
                        id="affiliate-salottery",
                        className="text-decoration-none mx-1 affiliate-link",
                    ),
                ], className="d-flex align-items-center justify-content-center flex-wrap gap-2 mb-2"),
                html.P(
                    "Play responsibly. 18+ only.",
                    className="text-center mb-0 mt-2",
                    style={"color": "#556677", "fontSize": "0.7rem"},
                ),
            ]),
        ], className="mb-4 py-3 px-3 text-center",
           style={"backgroundColor": "rgba(255, 68, 68, 0.06)", "borderRadius": "10px",
                   "border": "1px solid rgba(255, 68, 68, 0.25)",
                   "boxShadow": "0 0 20px rgba(255, 68, 68, 0.08)"}),

        dcc.Store(id="jackpot-store", data=LIVE_JACKPOT),

        build_hot_cold(result),

        html.Div([
            html.P(
                "Most people pick numbers randomly.",
                className="text-center mb-1",
                style={"color": "#778899", "fontSize": "0.95rem"},
            ),
            html.P(
                "These combinations come from historical draw patterns.",
                className="text-center mb-3",
                style={"color": COLORS["text"], "fontSize": "0.95rem", "fontWeight": "500"},
            ),
        ], className="mb-2"),

        build_tickets_section(result),

        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-triangle me-2",
                           style={"color": "#ff4444", "fontSize": "1.2rem"}),
                    html.Span("DRAW CLOSING SOON", className="fw-bold",
                              style={"color": "#ff4444", "fontSize": "0.85rem",
                                     "letterSpacing": "2px", "textTransform": "uppercase"}),
                ], className="d-flex align-items-center justify-content-center mb-3"),
                html.Div([
                    html.Div([
                        html.Span("--", id="cd-days-2", style=countdown_num_style, className="countdown-num"),
                        html.Span("DAYS", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                    html.Span(":", style=countdown_sep_style, className="countdown-sep"),
                    html.Div([
                        html.Span("--", id="cd-hours-2", style=countdown_num_style, className="countdown-num"),
                        html.Span("HRS", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                    html.Span(":", style=countdown_sep_style, className="countdown-sep"),
                    html.Div([
                        html.Span("--", id="cd-mins-2", style=countdown_num_style, className="countdown-num"),
                        html.Span("MIN", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                    html.Span(":", style=countdown_sep_style, className="countdown-sep"),
                    html.Div([
                        html.Span("--", id="cd-secs-2", style=countdown_num_style, className="countdown-num"),
                        html.Span("SEC", style=countdown_label_style),
                    ], style=countdown_box_style, className="countdown-box"),
                ], className="d-flex align-items-center justify-content-center flex-wrap mb-2"),
                html.P(
                    f"until {draw_name} draw",
                    className="text-center mb-2",
                    style={"color": "#8899aa", "fontSize": "0.85rem"},
                ),
                html.P(
                    "Your tickets are ready. Don\u2019t miss this draw.",
                    className="text-center mb-3",
                    style={"color": COLORS["text"], "fontSize": "0.95rem", "fontWeight": "500"},
                ),
                html.Div([
                    html.A(
                        dbc.Button([
                            html.Span("Betway", className="fw-bold"),
                        ], color="success", className="px-3 py-2 fw-bold affiliate-btn",
                           style={"fontSize": "0.9rem", "borderRadius": "6px",
                                  "backgroundColor": "#00a651", "borderColor": "#00a651"}),
                        href="#", target="_blank",
                        className="text-decoration-none mx-1 affiliate-link",
                    ),
                    html.A(
                        dbc.Button([
                            html.Span("Hollywoodbets", className="fw-bold"),
                        ], color="danger", className="px-3 py-2 fw-bold affiliate-btn",
                           style={"fontSize": "0.9rem", "borderRadius": "6px",
                                  "backgroundColor": "#800020", "borderColor": "#800020"}),
                        href="#", target="_blank",
                        className="text-decoration-none mx-1 affiliate-link",
                    ),
                    html.A(
                        dbc.Button([
                            html.Span("SA Lottery", className="fw-bold"),
                        ], color="warning", className="px-3 py-2 fw-bold affiliate-btn",
                           style={"fontSize": "0.9rem", "borderRadius": "6px",
                                  "color": "#000"}),
                        href="https://www.nationallottery.co.za", target="_blank",
                        className="text-decoration-none mx-1 affiliate-link",
                    ),
                ], className="d-flex align-items-center justify-content-center flex-wrap gap-2 mb-2"),
                html.P(
                    "Play responsibly. 18+ only.",
                    className="text-center mb-0 mt-2",
                    style={"color": "#556677", "fontSize": "0.7rem"},
                ),
            ]),
        ], className="my-5 py-3 px-3 text-center",
           style={"backgroundColor": "rgba(255, 68, 68, 0.06)", "borderRadius": "10px",
                   "border": "1px solid rgba(255, 68, 68, 0.25)",
                   "boxShadow": "0 0 20px rgba(255, 68, 68, 0.08)"}),

        html.Div([
            html.H2([
                html.I(className="fas fa-microscope me-2"),
                "Analytics Lab",
            ], className="mb-1 section-heading", id="analytics"),
            html.P(
                "The patterns behind the numbers most players never look at.",
                className="mb-1",
                style={"color": COLORS["text"], "fontSize": "1.05rem"},
            ),
            html.P(
                "Number frequency. Distribution trends. Probability heat zones. Some players prefer to trust their gut. Others like to peek behind the curtain.",
                className="mb-4",
                style={"color": COLORS["subtle"], "fontStyle": "italic"},
            ),
        ]),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-chart-bar me-2", style={"color": COLORS["primary"]}),
                html.Span("Number Frequency", className="fw-bold"),
                html.Small("  (red = hot, blue = cold)", className="text-muted ms-2"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.P(
                    "This chart shows how often each number has appeared across the historical dataset. Some numbers seem to show up everywhere. Others hide in the background. The chart simply shows the footprints left behind by past draws.",
                    className="small mb-3",
                    style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                ),
                dcc.Graph(figure=freq_fig, config={"displayModeBar": False}),
            ]),
        ], className="shadow mb-4", style={"backgroundColor": COLORS["card"]}),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-th me-2", style={"color": COLORS["success"]}),
                html.Span("Probability Heatmap", className="fw-bold"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.P(
                    "The heatmap visualises where numbers tend to appear across the full range. Brighter zones highlight numbers that show up more frequently in the dataset. Whether you chase them or avoid them... is entirely up to you.",
                    className="small mb-3",
                    style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                ),
                dcc.Graph(figure=heat_fig, config={"displayModeBar": False}),
            ]),
        ], className="shadow mb-4", style={"backgroundColor": COLORS["card"]}),

        html.Hr(className="my-5"),

        html.Div([
            html.H2([
                html.I(className="fas fa-trophy me-2"),
                "Performance",
            ], className="mb-1 section-heading", id="performance"),
            html.P(
                "Every model deserves a reality check. The performance engine compares generated combinations against past draw results to see how the patterns behave historically.",
                className="text-muted mb-1",
            ),
            html.P(
                "No promises. Just data and curiosity.",
                className="mb-4",
                style={"color": COLORS["subtle"], "fontStyle": "italic"},
            ),
        ]),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-flask me-2", style={"color": COLORS["warning"]}),
                html.Span("Backtest Engine", className="fw-bold"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.P(
                    "Run the model against past draws and see how different strategies interact with historical results. It's part experiment. Part number obsession.",
                    className="small mb-3",
                    style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                ),
                dbc.Button([
                    html.I(className="fas fa-play me-2"),
                    "Run Backtest",
                ], id="run-backtest-btn", color="info", className="mb-3"),
                dcc.Loading(
                    id="backtest-loading",
                    type="circle",
                    children=html.Div(id="backtest-results"),
                    color=COLORS["primary"],
                ),
            ]),
        ], className="shadow mb-4", style={"backgroundColor": COLORS["card"]}),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-history me-2", style={"color": COLORS["secondary"]}),
                html.Span("Recent Draws", className="fw-bold"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.P(
                    "The latest official results feed straight back into the dataset. Every new draw changes the landscape. Which means the patterns are always shifting.",
                    className="small mb-3",
                    style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                ),
                build_recent_draws(result),
            ]),
        ], className="shadow mb-4", style={"backgroundColor": COLORS["card"]}),

        html.Footer([
            html.Hr(className="mt-5"),
            html.Div([
                html.P(
                    "Some players enjoy exploring the patterns first.",
                    className="text-center mb-1",
                    style={"color": "#aabbcc", "fontSize": "0.95rem"},
                ),
                html.P(
                    "Share it with a close friend if you think they'd appreciate it.",
                    className="text-center mb-3",
                    style={"color": "#aabbcc", "fontSize": "0.95rem", "fontWeight": "500"},
                ),
                html.Hr(style={"borderColor": "#333", "maxWidth": "200px", "margin": "0 auto 1rem auto"}),
                html.P([
                    html.I(className="fas fa-exclamation-triangle me-2 text-warning"),
                    "For entertainment purposes only. Lottery outcomes are random and cannot be predicted.",
                ], className="text-center text-muted small"),
            ]),
        ], className="mt-5 mb-3"),
    ])


app.index_string = '''<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>''' + MOBILE_CSS + '''</style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>'''

app.layout = html.Div([
    create_navbar(),

    dbc.Container([
        html.Div([
            html.H2([
                "The Lottery Is Random.",
            ], className="mb-0 fw-bold hero-title", id="predictions", style={"fontSize": "2rem"}),
            html.H3(
                "But The Numbers Still Leave Clues.",
                className="mb-3 hero-subtitle",
                style={"color": COLORS["gold"], "fontWeight": "300", "fontSize": "1.5rem"},
            ),
            html.P(
                "Every draw adds another layer of data. Lotto-Lab scans historical results, number momentum, and distribution patterns to surface combinations and trends many players never notice.",
                className="mb-1",
                style={"color": "#bbc8d4", "maxWidth": "700px", "lineHeight": "1.6"},
            ),
            html.P(
                "It won't change the rules of the game. But it might change how you play it.",
                className="mb-4",
                style={"color": COLORS["subtle"], "fontStyle": "italic"},
            ),
        ]),

        html.Div(id="main-content", children=build_full_page(initial_result)),

    ], fluid=True, className="px-4"),

    dcc.Loading(
        id="loading-overlay",
        type="circle",
        fullscreen=True,
        style={"backgroundColor": "rgba(0,0,0,0.7)"},
    ),
])


@callback(
    [Output("cd-days", "children"),
     Output("cd-hours", "children"),
     Output("cd-mins", "children"),
     Output("cd-secs", "children"),
     Output("cd-days-2", "children"),
     Output("cd-hours-2", "children"),
     Output("cd-mins-2", "children"),
     Output("cd-secs-2", "children")],
    Input("countdown-interval", "n_intervals"),
    State("next-draw-ts", "data"),
)
def update_countdown(n, target_ts):
    if not target_ts:
        return "--", "--", "--", "--", "--", "--", "--", "--"
    now = datetime.now().timestamp()
    remaining = max(0, target_ts - now)
    if remaining <= 0:
        return "00", "00", "00", "00", "00", "00", "00", "00"
    days = int(remaining // 86400)
    hours = int((remaining % 86400) // 3600)
    mins = int((remaining % 3600) // 60)
    secs = int(remaining % 60)
    d, h, m, s = f"{days:02d}", f"{hours:02d}", f"{mins:02d}", f"{secs:02d}"
    return d, h, m, s, d, h, m, s


@callback(
    [Output("locked-tickets-wrapper", "style"),
     Output("unlocked-tickets", "children"),
     Output("unlocked-tickets", "style"),
     Output("preview-tickets-wrapper", "style"),
     Output("email-error", "children")],
    [Input("unlock-btn", "n_clicks"),
     Input("telegram-btn", "n_clicks")],
    [State("email-input", "value"),
     State("tickets-store", "data")],
    prevent_initial_call=True,
)
def unlock_tickets(phone_clicks, tg_clicks, phone_value, stored_tickets):
    triggered = ctx.triggered_id

    if triggered == "telegram-btn":
        print(f"[Dashboard] Telegram join request")
    elif triggered == "unlock-btn":
        if not phone_value:
            error_msg = html.Small(
                "Please enter your mobile number.",
                style={"color": "#ff6b6b", "fontSize": "0.8rem"},
            )
            return no_update, no_update, no_update, no_update, error_msg

        digits = "".join(c for c in str(phone_value) if c.isdigit())
        if len(digits) < 7:
            error_msg = html.Small(
                "Please enter a valid mobile number.",
                style={"color": "#ff6b6b", "fontSize": "0.8rem"},
            )
            return no_update, no_update, no_update, no_update, error_msg

        masked = digits[:3] + "***" + digits[-2:]
        print(f"[Dashboard] New member signup (mobile): {masked}")
    else:
        return no_update, no_update, no_update, no_update, no_update

    last_draw = draws[0].numbers if draws else []
    tickets = stored_tickets or []
    best_score = tickets[0]["score"] if tickets else 1

    all_cards = []
    for i, t in enumerate(tickets, 1):
        all_cards.append(build_ticket_card(i, t["score"], t["ticket"], best_score, last_draw))

    unlocked_content = html.Div([
        dbc.Alert([
            html.I(className="fas fa-check-circle me-2"),
            f"All {len(tickets)} tickets unlocked. Good luck!",
        ], color="success", className="text-center"),
        html.Div(all_cards),
    ])

    return (
        {"display": "none"},
        unlocked_content,
        {"display": "block"},
        {"display": "none"},
        "",
    )


@callback(
    Output("backtest-results", "children"),
    Input("run-backtest-btn", "n_clicks"),
    prevent_initial_call=True,
)
def run_backtest(n_clicks):
    try:
        result = engine.run_backtest_analysis(draws, n_max, k, window=100, n_samples_per_step=1000, top_n_per_step=10)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error running backtest: {str(e)}", color="danger")

    if "error" in result:
        return dbc.Alert(result["error"], color="warning")

    bt = result["backtest"]
    model_ev = result["model_ev"]
    random_ev = result["random_ev"]
    model_sim = result["model_sim"]
    random_sim = result["random_sim"]
    top_hits = result.get("top_hit_examples", [])

    model_hist = bt["model_hist"]
    random_hist = bt["random_hist"]

    hits = list(range(7))
    fig = go.Figure(data=[
        go.Bar(name="Lotto-Lab Model", x=hits, y=[model_hist[i] if i < len(model_hist) else 0 for i in hits],
               marker_color=COLORS["success"]),
        go.Bar(name="Random Picks", x=hits, y=[random_hist[i] if i < len(random_hist) else 0 for i in hits],
               marker_color=COLORS["secondary"]),
    ])
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color=COLORS["text"],
        xaxis=dict(title="Numbers Matched", gridcolor="#333"),
        yaxis=dict(title="Frequency", gridcolor="#333"),
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    model_3plus = sum(model_hist[i] for i in range(3, 7) if i < len(model_hist))
    random_3plus = sum(random_hist[i] for i in range(3, 7) if i < len(random_hist))
    total_eval = bt.get("n_eval", sum(model_hist))
    model_3pct = (model_3plus / total_eval * 100) if total_eval > 0 else 0
    random_3pct = (random_3plus / total_eval * 100) if total_eval > 0 else 0

    proof_cards = []
    for ex in top_hits[:8]:
        matched_set = set(ex["matched"])
        ticket_balls = []
        for n in ex["ticket"]:
            if n in matched_set:
                ticket_balls.append(ball_element(n, "#39FF14", "#000", "36px"))
            else:
                ticket_balls.append(ball_element(n, "#555", "#aaa", "36px"))

        actual_balls = []
        for n in ex["actual"]:
            if n in matched_set:
                actual_balls.append(ball_element(n, "#39FF14", "#000", "36px"))
            else:
                actual_balls.append(ball_element(n, "#00d4ff", "#000", "36px"))

        hit_color = "#FFD700" if ex["hits"] >= 4 else "#39FF14" if ex["hits"] >= 3 else COLORS["primary"]

        proof_cards.append(
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                dbc.Badge(f"{ex['hits']} MATCHED", color="success" if ex["hits"] < 4 else "warning",
                                          className="me-2 fs-6"),
                                html.Small(ex["date"], className="text-muted"),
                                html.Small(f"  Draw #{ex['draw_id']}", className="text-muted ms-2"),
                            ]),
                        ]),
                    ], className="mb-2"),
                    dbc.Row([
                        dbc.Col([
                            html.Small("Our Ticket:", className="text-muted d-block mb-1"),
                            html.Div(ticket_balls, className="d-flex flex-wrap gap-1"),
                        ], md=6),
                        dbc.Col([
                            html.Small("Actual Draw:", className="text-muted d-block mb-1"),
                            html.Div(actual_balls, className="d-flex flex-wrap gap-1"),
                        ], md=6),
                    ]),
                ], className="py-2"),
            ], className="mb-2 shadow-sm", style={
                "backgroundColor": COLORS["card"],
                "borderLeft": f"4px solid {hit_color}",
            })
        )

    return html.Div([
        html.Div([
            html.P(
                "The model was tested against every historical draw. Here's what happened.",
                className="mb-3",
                style={"color": COLORS["subtle"]},
            ),
        ]),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Draws Tested", className="text-center text-muted"),
                html.H2(f"{total_eval}", className="text-center fw-bold", style={"color": COLORS["primary"]}),
                html.P("Historical simulations", className="text-center text-muted small"),
            ])], className="shadow-sm", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Model Avg Hits", className="text-center text-muted"),
                html.H2(f"{bt['avg_model']:.2f}", className="text-center fw-bold", style={"color": COLORS["success"]}),
                html.P("Per draw (best ticket)", className="text-center text-muted small"),
            ])], className="shadow-sm", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("3+ Match Rate", className="text-center text-muted"),
                html.H2(f"{model_3pct:.1f}%", className="text-center fw-bold", style={"color": COLORS["gold"]}),
                html.P(f"vs {random_3pct:.1f}% random", className="text-center text-muted small"),
            ])], className="shadow-sm", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Model EV", className="text-center text-muted"),
                html.H2(f"R{model_ev['gross']:.2f}", className="text-center fw-bold", style={"color": COLORS["warning"]}),
                html.P(f"vs R{random_ev['gross']:.2f} random", className="text-center text-muted small"),
            ])], className="shadow-sm", style={"backgroundColor": COLORS["card"]}), xs=6, md=3, className="mb-3"),
        ]),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-chart-bar me-2", style={"color": COLORS["success"]}),
                html.Span("Match Distribution: Model vs Random", className="fw-bold"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.P(
                    "How often each match count occurred across all tested draws. The model aims to shift the distribution rightward.",
                    className="small mb-2",
                    style={"color": COLORS["subtle"]},
                ),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ]),
        ], className="shadow mb-4", style={"backgroundColor": COLORS["card"]}),

        dbc.Card([
            dbc.CardHeader([
                html.I(className="fas fa-star me-2", style={"color": COLORS["gold"]}),
                html.Span("Best Hits: Proof from the Data", className="fw-bold"),
                dbc.Badge(f"{len(top_hits)} matches of 3+", color="success", className="ms-2"),
            ], style={"backgroundColor": COLORS["dark"]}),
            dbc.CardBody([
                html.P(
                    "These are real examples where the model's generated ticket matched 3 or more numbers from the actual draw. Green balls show the matched numbers.",
                    className="small mb-3",
                    style={"color": COLORS["subtle"], "lineHeight": "1.5"},
                ),
                html.Div(proof_cards) if proof_cards else html.P("No 3+ matches found in this run.", className="text-muted"),
            ]),
        ], className="shadow mb-4", style={"backgroundColor": COLORS["card"]}),

        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H6("1-Year Bankroll Simulation", className="mb-2"),
                        html.P("Based on historical hit patterns, simulating 104 draws with R5 tickets.",
                               className="small text-muted mb-2"),
                    ], md=6),
                    dbc.Col([
                        dbc.Row([
                            dbc.Col([
                                html.Small("Model Median", className="text-muted d-block"),
                                html.H5(f"R{model_sim['p50']:,.0f}", className="fw-bold mb-0",
                                         style={"color": COLORS["success"]}),
                            ], xs=6),
                            dbc.Col([
                                html.Small("Random Median", className="text-muted d-block"),
                                html.H5(f"R{random_sim['p50']:,.0f}", className="fw-bold mb-0",
                                         style={"color": COLORS["secondary"]}),
                            ], xs=6),
                        ]),
                    ], md=6),
                ]),
            ]),
        ], className="shadow", style={"backgroundColor": COLORS["card"]}),
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
