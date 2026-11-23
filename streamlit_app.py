"""
DISASTERSHIELD - AI Disaster Alert & Smart Response System (Streamlit edition)

HONESTY NOTES (also shown in the UI):
 * LIVE data  : Open-Meteo (weather/soil), Open-Meteo Flood API (river discharge),
                USGS (earthquakes), NASA FIRMS (fires, needs a free MAP_KEY).
 * DEMO data  : clearly labelled synthetic inputs, only when Demo mode is switched on.
 * RISK SCORE : transparent rule-based weighted index (see Risk Assessment > Method).
 * ML         : scikit-learn IsolationForest anomaly detection trained on the last
                30 days of each location's own data. It is NOT a supervised
                disaster-prediction model, and earthquakes cannot be predicted.
 * ALERTS     : simulated / test only. Nothing is sent to the public.
"""
import io
import os
import sqlite3
import zlib
from datetime import datetime, timedelta, timezone

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from folium import plugins
from sklearn.ensemble import IsolationForest
from streamlit_folium import st_folium

st.set_page_config(page_title="DisasterShield", page_icon="🛡️", layout="wide")

DB = "disastershield.db"
SAMPLE_AREAS = [
    ("Nashik", 19.9975, 73.7898), ("Pune", 18.5204, 73.8567), ("Mumbai", 19.0760, 72.8777),
    ("Kolhapur", 16.7050, 74.2433), ("Satara", 17.6805, 74.0183), ("Nagpur", 21.1458, 79.0882),
]
HAZARDS = ["Flood", "Cyclone", "Earthquake", "Wildfire", "Landslide", "Extreme weather"]
FEATS = ["temp", "hum", "rain", "wind", "pres", "soil"]
LEVELS = [(25, "LOW", "#22c55e"), (50, "MODERATE", "#eab308"), (75, "HIGH", "#f97316"), (100, "CRITICAL", "#ef4444")]

# ----------------------------------------------------------------- styling
st.markdown("""
<style>
.stApp{background:linear-gradient(135deg,#0a1128 0%,#0d1b3d 55%,#0b2447 100%);color:#e2e8f0}
[data-testid=stSidebar]{background:rgba(10,17,40,.92);border-right:1px solid rgba(96,165,250,.25)}
[data-testid=stMetric]{background:rgba(255,255,255,.06);backdrop-filter:blur(10px);border:1px solid rgba(96,165,250,.25);
  border-radius:14px;padding:12px 16px}
.badge{display:inline-block;padding:3px 12px;border-radius:999px;font-size:.8rem;font-weight:700}
.live{background:#14532d;color:#86efac}.demo{background:#713f12;color:#fde047}
.pulse{display:inline-block;width:9px;height:9px;border-radius:50%;background:#22c55e;margin-right:6px;
  animation:p 1.6s infinite}@keyframes p{0%{opacity:1}50%{opacity:.25}100%{opacity:1}}
.glass{background:rgba(255,255,255,.06);border:1px solid rgba(96,165,250,.25);border-radius:14px;padding:14px 18px;margin-bottom:10px}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers
def secret(key):
    try:
        return st.secrets.get(key) or os.getenv(key)
    except Exception:
        return os.getenv(key)


def level(score):
    for hi, name, col in LEVELS:
        if score <= hi:
            return name, col
    return LEVELS[-1][1], LEVELS[-1][2]


def ramp(x, lo, hi):
    """Linear 0..1 ramp; None when the input is missing (so it is excluded, not treated as 0)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def wsum(parts):
    ok = [(w, v) for w, v in parts if v is not None]
    tw = sum(w for w, _ in ok)
    return round(100 * sum(w * v for w, v in ok) / tw, 1) if tw else 0.0


def fmt(x, unit="", nd=0):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}{unit}"


# ----------------------------------------------------------------- database
def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE IF NOT EXISTS areas(name TEXT PRIMARY KEY, lat REAL, lon REAL, subscribed INTEGER DEFAULT 0)")
    c.execute("CREATE TABLE IF NOT EXISTS alerts(id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, area TEXT, hazard TEXT,"
              " score REAL, level TEXT, reason TEXT, is_test INTEGER)")
    if c.execute("SELECT COUNT(*) FROM areas").fetchone()[0] == 0:
        c.executemany("INSERT INTO areas(name,lat,lon) VALUES(?,?,?)", SAMPLE_AREAS)
        c.commit()
    return c


def get_areas():
    with db() as c:
        return [dict(r) for r in c.execute("SELECT * FROM areas ORDER BY name")]


# ----------------------------------------------------------------- data connectors
@st.cache_data(ttl=600, show_spinner=False)
def fetch_weather(lat, lon):
    r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=25, params={
        "latitude": lat, "longitude": lon, "timezone": "UTC", "past_days": 30, "forecast_days": 3,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,pressure_msl,soil_moisture_0_to_7cm"})
    r.raise_for_status()
    h = pd.DataFrame(r.json()["hourly"])
    h["time"] = pd.to_datetime(h["time"], utc=True)
    h = h.set_index("time")
    h.columns = FEATS
    return h


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_flood(lat, lon):
    """River discharge today vs its seasonal mean (proxy for water level; modelled, not a gauge)."""
    r = requests.get("https://flood-api.open-meteo.com/v1/flood", timeout=25, params={
        "latitude": lat, "longitude": lon, "forecast_days": 3, "daily": "river_discharge,river_discharge_mean"})
    r.raise_for_status()
    d = r.json()["daily"]
    q, m = d["river_discharge"][0], d["river_discharge_mean"][0]
    return (q / m) if q is not None and m else None


@st.cache_data(ttl=900, show_spinner=False)
def fetch_quakes(lat, lon):
    start = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    r = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", timeout=25, params={
        "format": "geojson", "latitude": lat, "longitude": lon, "maxradiuskm": 300,
        "starttime": start, "minmagnitude": 2.5})
    r.raise_for_status()
    return [{"mag": f["properties"]["mag"], "place": f["properties"]["place"],
             "lon": f["geometry"]["coordinates"][0], "lat": f["geometry"]["coordinates"][1]}
            for f in r.json()["features"]]


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_fires(lat, lon):
    key = secret("FIRMS_MAP_KEY")
    if not key:
        return None
    d = 1.0
    url = (f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_SNPP_NRT/"
           f"{lon - d},{lat - d},{lon + d},{lat + d}/2")
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return len(pd.read_csv(io.StringIO(r.text)))


def sensor_overrides(area_name):
    """EXTENSION POINT: return {'rain_now':..,'soil_pct':..} from IoT gauges/sensors (MQTT, Supabase, REST...)."""
    return {}


# ----------------------------------------------------------------- demo data
def demo_hourly(name):
    rs = np.random.RandomState(zlib.crc32(name.encode()) % 10000)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(hours=72), periods=33 * 24, freq="h")
    t = np.arange(len(idx))
    df = pd.DataFrame({
        "temp": 27 + 5 * np.sin(2 * np.pi * t / 24) + rs.normal(0, .6, len(t)),
        "hum": 65 + 15 * np.cos(2 * np.pi * t / 24) + rs.normal(0, 3, len(t)),
        "rain": np.clip(rs.gamma(.15, 1.0, len(t)), 0, None),
        "wind": 10 + rs.gamma(2, 2, len(t)),
        "pres": 1008 + rs.normal(0, 1.5, len(t)),
        "soil": 0.28 + rs.normal(0, .01, len(t)),
    }, index=idx)
    spike = slice(len(t) - 72 - 30, len(t) - 72 - 6)  # heavy-rain episode ending ~6h ago
    df.loc[df.index[spike], "rain"] += rs.uniform(2, 9, 24) * (1 + rs.rand())
    df.loc[df.index[spike], "soil"] += np.linspace(0, .14, 24)
    df.loc[df.index[spike], "pres"] -= np.linspace(0, 9, 24)
    return df


def anomaly_score(past):
    X = past[FEATS].ffill().bfill().dropna(axis=1, how="all")
    if len(X) < 200:
        return 0.0
    model = IsolationForest(n_estimators=100, contamination=0.02, random_state=0).fit(X.iloc[:-24])
    return round(100 * float((model.predict(X.iloc[-24:]) == -1).mean()), 1)


@st.cache_data(ttl=600, show_spinner=False)
def get_snapshot(name, lat, lon, demo):
    errors, src = [], {}
    if demo:
        h, dis, quakes, fires = demo_hourly(name), 1 + (zlib.crc32(name.encode()) % 40) / 10, [], 0
        src = {k: "DEMO (synthetic sample)" for k in ["Weather", "Water level", "Seismic", "Fire"]}
    else:
        try:
            h = fetch_weather(lat, lon)
            src["Weather"] = "Open-Meteo Forecast API"
        except Exception as e:
            return {"error": f"Open-Meteo weather unavailable ({type(e).__name__})"}
        dis = quakes = fires = None
        try:
            dis = fetch_flood(lat, lon)
            src["Water level"] = "Open-Meteo Flood API (modelled river discharge)"
        except Exception as e:
            errors.append(f"Flood API unavailable ({type(e).__name__})")
        try:
            quakes = fetch_quakes(lat, lon)
            src["Seismic"] = "USGS Earthquake API (300 km, 30 days)"
        except Exception as e:
            errors.append(f"USGS unavailable ({type(e).__name__})")
        try:
            fires = fetch_fires(lat, lon)
            src["Fire"] = "NASA FIRMS VIIRS (48h)" if fires is not None else "NASA FIRMS not configured (set FIRMS_MAP_KEY)"
        except Exception as e:
            errors.append(f"FIRMS unavailable ({type(e).__name__})")
    now = pd.Timestamp.now(tz="UTC").floor("h")
    past, fut = h[h.index <= now], h[h.index > now]
    last = past.iloc[-1]
    soil = None if pd.isna(last["soil"]) else float(last["soil"])
    s = dict(
        mode="DEMO" if demo else "LIVE", time=datetime.now(timezone.utc), errors=errors, sources=src,
        temp=float(last["temp"]), hum=float(last["hum"]), wind=float(last["wind"]), pres=float(last["pres"]),
        rain_now=float(last["rain"]), r24=float(past["rain"].tail(24).sum()), r72=float(past["rain"].tail(72).sum()),
        fc24=float(fut["rain"].head(24).sum()), soil_pct=None if soil is None else min(100, soil / 0.5 * 100),
        dis=dis, quakes=quakes, fires=fires, anomaly=anomaly_score(past), hourly=h, now=now)
    s.update(sensor_overrides(name))
    return s


# ----------------------------------------------------------------- risk engine (rule-based, documented)
def score_hazards(s):
    q = s["quakes"]
    qmax = max((x["mag"] or 0 for x in q), default=0) if q is not None else None
    qn = len(q) if q is not None else None
    return {
        "Flood": wsum([(.30, ramp(s["r24"], 20, 150)), (.15, ramp(s["r72"], 40, 250)), (.15, ramp(s["fc24"], 20, 150)),
                       (.10, ramp(s["soil_pct"], 50, 90)), (.30, ramp(s["dis"], 1, 5))]),
        "Cyclone": wsum([(.50, ramp(s["wind"], 40, 120)), (.30, ramp(1010 - s["pres"], 0, 25)), (.20, ramp(s["r24"], 30, 200))]),
        "Earthquake": wsum([(.70, ramp(qmax, 3, 6)), (.30, ramp(qn, 0, 15))]) if q is not None else 0.0,
        "Wildfire": wsum([(.30, ramp(s["temp"], 30, 45)), (.30, ramp(60 - s["hum"], 0, 45)), (.15, ramp(s["wind"], 10, 50)),
                          (.10, 1 - ramp(s["r72"], 0, 10)), (.15, ramp(s["fires"], 0, 10))]),
        "Landslide": wsum([(.45, ramp(s["r72"], 50, 300)), (.35, ramp(s["soil_pct"], 50, 90)), (.20, ramp(s["r24"], 30, 150))]),
        "Extreme weather": round(100 * max(ramp(s["temp"], 35, 47), ramp(5 - s["temp"], 0, 15),
                                           ramp(s["wind"], 40, 100), ramp(s["rain_now"], 5, 40)), 1),
    }


FACT = {
    "Flood": lambda s: f"24h rain {fmt(s['r24'], ' mm')}, 72h rain {fmt(s['r72'], ' mm')}, forecast next-24h rain "
                       f"{fmt(s['fc24'], ' mm')}, soil saturation {fmt(s['soil_pct'], '%')}, river discharge "
                       f"{fmt(s['dis'], 'x its seasonal mean', 1)}",
    "Cyclone": lambda s: f"wind {fmt(s['wind'], ' km/h')}, sea-level pressure {fmt(s['pres'], ' hPa')}, 24h rain {fmt(s['r24'], ' mm')}",
    "Earthquake": lambda s: f"{fmt(len(s['quakes']) if s['quakes'] is not None else None)} quakes (M2.5+) within 300 km in 30 days, "
                            f"max M{max((x['mag'] or 0 for x in s['quakes']), default=0):.1f}" if s["quakes"] is not None else "USGS data unavailable",
    "Wildfire": lambda s: f"temperature {fmt(s['temp'], ' °C')}, humidity {fmt(s['hum'], '%')}, wind {fmt(s['wind'], ' km/h')}, "
                          f"72h rain {fmt(s['r72'], ' mm')}, satellite hotspots {fmt(s['fires'])}",
    "Landslide": lambda s: f"72h rain {fmt(s['r72'], ' mm')}, soil saturation {fmt(s['soil_pct'], '%')}, 24h rain {fmt(s['r24'], ' mm')}",
    "Extreme weather": lambda s: f"temperature {fmt(s['temp'], ' °C')}, wind {fmt(s['wind'], ' km/h')}, rain rate {fmt(s['rain_now'], ' mm/h', 1)}",
}


def assess(s):
    hz = score_hazards(s)
    top = max(hz, key=hz.get)
    overall = round(min(100, .75 * hz[top] + .25 * np.mean(list(hz.values())) + .10 * s["anomaly"]), 1)
    lv = level(overall)[0]
    why = (f"{top} risk is {level(hz[top])[0].lower()} ({hz[top]:.0f}/100), driven by: {FACT[top](s)}. "
           f"Anomaly detector flags {s['anomaly']:.0f}% of the last 24h as unusual for this location. "
           "Residents in vulnerable areas should monitor official advisories (IMD, NDMA, local authorities).")
    return dict(hazards=hz, top=top, overall=overall, level=lv, why=why)


@st.cache_data(ttl=600, show_spinner="Collecting environmental data...")
def collect(demo, areas_key):
    data, errs = {}, []
    for a in get_areas():
        s = get_snapshot(a["name"], a["lat"], a["lon"], demo)
        if "error" in s:
            errs.append(f"{a['name']}: {s['error']}")
            continue
        data[a["name"]] = (a, s, assess(s))
    return data, errs


def active_alerts(data):
    out = []
    for n, (a, s, r) in data.items():
        for h, v in r["hazards"].items():
            if v >= 51:
                out.append(dict(area=n, hazard=h, score=v, level=level(v)[0], reason=FACT[h](s)))
    return sorted(out, key=lambda x: -x["score"])


# ----------------------------------------------------------------- knowledge base
INFO = {
    "Flood": dict(
        what="Overflow of water onto normally dry land from rivers, heavy rain, dam release or storm surge.",
        why="Rainfall exceeds the ground's absorption and the drainage/river capacity.",
        signs=["Sustained heavy rain", "Rapidly rising river/nala levels", "Official flood or dam-release warnings"],
        causes=["Intense monsoon rain", "Saturated soil", "Blocked drains, encroachment, dam release"],
        impacts=["Loss of life and homes", "Crop and road damage", "Waterborne disease"],
        before=["Know your evacuation route and shelter", "Keep documents in waterproof bags", "Move valuables upstairs"],
        during=["Move to higher ground early", "Never walk or drive through flood water", "Switch off electricity at the mains"],
        after=["Return only when authorities say it is safe", "Boil/purify drinking water", "Photograph damage for claims"],
        checklist=["Torch + batteries", "Drinking water (3 days)", "First-aid kit", "Power bank", "Documents copy", "Essential medicines"]),
    "Earthquake": dict(
        what="Sudden shaking of the ground caused by release of stress along faults.",
        why="Tectonic plates move and fractures slip; timing cannot be predicted.",
        signs=["No reliable short-term sign", "Small foreshocks sometimes occur", "Unusual animal behaviour is NOT a proven predictor"],
        causes=["Fault movement", "Volcanic activity", "Reservoir-induced seismicity (rare)"],
        impacts=["Building collapse", "Fires and gas leaks", "Landslides and aftershocks"],
        before=["Secure heavy furniture", "Identify safe spots", "Check building safety with an engineer"],
        during=["Drop, Cover, Hold On", "Stay away from windows", "If outside, move to open ground"],
        after=["Expect aftershocks", "Check for gas leaks and injuries", "Use stairs, not lifts"],
        checklist=["Whistle", "Torch", "First-aid kit", "Sturdy shoes", "Water + food", "Family meeting point"]),
    "Cyclone": dict(
        what="A rotating storm system with very strong winds, heavy rain and storm surge over warm seas.",
        why="Warm ocean water feeds low-pressure systems that organise into cyclones.",
        signs=["Falling pressure", "Rising winds and swell", "IMD cyclone watch/warning"],
        causes=["Sea surface temperature above ~26.5 °C", "Low wind shear", "Moist atmosphere"],
        impacts=["Wind damage", "Coastal flooding and surge", "Power and communication failure"],
        before=["Follow IMD bulletins", "Secure loose objects and roofs", "Know the nearest cyclone shelter"],
        during=["Stay indoors away from windows", "Do not go out in the calm 'eye'", "Keep a battery radio"],
        after=["Avoid fallen power lines", "Wait for official all-clear", "Do not drink untreated water"],
        checklist=["Battery radio", "Power bank", "Dry food", "Water", "Medicines", "Waterproof document pouch"]),
    "Wildfire": dict(
        what="Uncontrolled fire in forest, grass or scrubland.",
        why="Heat, dry fuel and wind allow ignition (human or natural) to spread.",
        signs=["Very low humidity", "High temperature and wind", "Smoke or satellite hotspots nearby"],
        causes=["Prolonged dry spell", "Careless fires/burning", "Lightning"],
        impacts=["Loss of forest and wildlife", "Smoke and respiratory illness", "Threat to villages"],
        before=["Clear dry vegetation near homes", "Never burn waste in dry season", "Know evacuation routes"],
        during=["Evacuate early if advised", "Cover nose and mouth", "Call Forest Dept / 112"],
        after=["Avoid ash and hot spots", "Check for smouldering embers", "Report remaining fires"],
        checklist=["N95 masks", "Water", "Torch", "Documents", "First-aid kit", "Phone numbers list"]),
    "Landslide": dict(
        what="Downslope movement of rock, soil and debris.",
        why="Rain-saturated or disturbed slopes lose strength and fail.",
        signs=["New cracks in slopes/roads", "Tilting trees/poles", "Muddy spring water, rumbling sound"],
        causes=["Prolonged heavy rain", "Deforestation, quarrying, road cutting", "Earthquakes"],
        impacts=["Burial of homes and roads", "Blocked rivers", "Isolation of communities"],
        before=["Avoid building on steep slopes", "Maintain drainage", "Know the warning signs"],
        during=["Move away from the path, not downhill along it", "Run to stable high ground", "Alert neighbours"],
        after=["Stay away from the slide area", "Watch for further slides", "Report broken utilities"],
        checklist=["Torch", "Whistle", "Water", "First-aid kit", "Emergency contacts", "Sturdy shoes"]),
    "Extreme weather": dict(
        what="Rare or severe weather: heatwaves, cold waves, thunderstorms, lightning, hail, very heavy rain.",
        why="Unusual atmospheric conditions push temperature, wind or rainfall past normal limits.",
        signs=["IMD heat/cold/thunderstorm alerts", "Rapid temperature change", "Dark clouds and gusty winds"],
        causes=["Heat domes", "Strong convection", "Climate variability and change"],
        impacts=["Heatstroke", "Lightning injuries", "Crop and property damage"],
        before=["Check forecasts", "Prepare cooling/heating options", "Trim weak branches"],
        during=["Heat: stay hydrated and indoors 12-4pm", "Storm: shelter in a solid building", "Avoid open fields and trees"],
        after=["Check on elderly neighbours", "Avoid damaged wires", "Seek care for heat illness"],
        checklist=["ORS", "Water", "Torch", "Radio", "First-aid kit", "Umbrella/raincoat"]),
}
ACTIONS = {h: INFO[h]["before"] for h in INFO}


# ----------------------------------------------------------------- AI assistant
SYSTEM = ("You are the DisasterShield assistant for Indian communities. Explain environmental data simply, answer disaster "
          "preparedness questions and prioritise official guidance (NDMA, IMD, local disaster management). Use ONLY the "
          "risk context given for current conditions; say so if data is missing. Risk scores are a rule-based index plus "
          "an anomaly detector, not a validated forecast; earthquakes cannot be predicted. Never tell people to ignore official "
          "orders. For emergencies advise calling 112.\n\nCURRENT CONTEXT (JSON):\n")


def context_json(data):
    import json
    return json.dumps({n: dict(mode=s["mode"], overall=r["overall"], level=r["level"], top=r["top"], hazards=r["hazards"],
                               why=r["why"], temp=s["temp"], hum=s["hum"], wind=s["wind"], pressure=s["pres"], r24=s["r24"],
                               r72=s["r72"], soil_pct=s["soil_pct"]) for n, (a, s, r) in data.items()}, default=str)


def fallback_answer(q, data):
    ql = q.lower()
    for n, (a, s, r) in data.items():
        if n.lower() in ql:
            return f"**{n}** - overall {r['overall']:.0f}/100 ({r['level']}). {r['why']}"
    for h, d in INFO.items():
        if h.lower() in ql or (h == "Extreme weather" and "weather" in ql):
            return f"**{h}** - {d['what']}\n\n**Before:** " + "; ".join(d["before"]) + "\n\n**During:** " + "; ".join(d["during"])
    rank = sorted(data.items(), key=lambda kv: -kv[1][2]["overall"])
    return "Current ranking:\n" + "\n".join(f"- {n}: {r['overall']:.0f} ({r['level']}), main hazard {r['top']}" for n, (a, s, r) in rank)


def ask(q, data, history):
    key = secret("ANTHROPIC_API_KEY")
    if not key or not data:
        return fallback_answer(q, data) if data else "No data loaded yet.", "rule-based fallback (no ANTHROPIC_API_KEY set)"
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": secret("LLM_MODEL") or "claude-sonnet-4-5", "max_tokens": 700,
                                "system": SYSTEM + context_json(data),
                                "messages": history[-6:] + [{"role": "user", "content": q}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"], "LLM (server-side)"
    except Exception as e:
        return fallback_answer(q, data), f"rule-based fallback (LLM error: {type(e).__name__})"


def chat_ui(key, data, examples=()):
    hist = st.session_state.setdefault(key, [])
    cols = st.columns(len(examples) or 1)
    picked = None
    for c, ex in zip(cols, examples):
        if c.button(ex, key=f"{key}-{ex}"):
            picked = ex
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input("Ask about risks, preparedness or safety...", key=f"{key}-in") or picked
    if q:
        with st.chat_message("user"):
            st.markdown(q)
        ans, how = ask(q, data, hist)
        with st.chat_message("assistant"):
            st.markdown(ans)
            st.caption(f"Source: {how}")
        hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]


# ----------------------------------------------------------------- UI components
def gauge(v, title, h=230):
    fig = go.Figure(go.Indicator(mode="gauge+number", value=v, title={"text": title}, gauge={
        "axis": {"range": [0, 100]}, "bar": {"color": level(v)[1]},
        "steps": [{"range": [a, b], "color": c} for a, b, c in
                  [(0, 25, "rgba(34,197,94,.2)"), (25, 50, "rgba(234,179,8,.2)"), (50, 75, "rgba(249,115,22,.2)"), (75, 100, "rgba(239,68,68,.2)")]]}))
    fig.update_layout(height=h, margin=dict(l=15, r=15, t=50, b=0), paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
    return fig


def risk_map(data, alerts_only=False, incidents=None, quakes=True):
    lat0 = np.mean([a["lat"] for a, _, _ in data.values()]) if data else 19.5
    lon0 = np.mean([a["lon"] for a, _, _ in data.values()]) if data else 75.5
    m = folium.Map(location=[lat0, lon0], zoom_start=6, tiles="OpenStreetMap")
    plugins.Fullscreen().add_to(m)
    plugins.LocateControl().add_to(m)
    plugins.Geocoder().add_to(m)
    for n, (a, s, r) in data.items():
        if alerts_only and r["overall"] < 51:
            continue
        col = level(r["overall"])[1]
        pop = (f"<b>{n}</b> [{s['mode']}]<br>Risk {r['overall']:.0f} ({r['level']})<br>Main hazard: {r['top']}<br>"
               f"{s['temp']:.1f}°C, RH {s['hum']:.0f}%, wind {s['wind']:.0f} km/h<br>24h rain {s['r24']:.0f} mm")
        folium.CircleMarker([a["lat"], a["lon"]], radius=10 + r["overall"] / 8, color=col, fill=True, fill_opacity=.7,
                            popup=folium.Popup(pop, max_width=260), tooltip=f"{n}: {r['level']}").add_to(m)
        if r["overall"] >= 51:
            folium.Circle([a["lat"], a["lon"]], radius=30000, color=col, fill=True, fill_opacity=.08).add_to(m)
        if quakes and s["quakes"]:
            for qk in s["quakes"][:20]:
                folium.CircleMarker([qk["lat"], qk["lon"]], radius=3 + (qk["mag"] or 0), color="#a78bfa",
                                    tooltip=f"M{qk['mag']} {qk['place']}").add_to(m)
    for inc in incidents or []:
        folium.Marker([inc["lat"], inc["lon"]], tooltip=f"SAMPLE: {inc['type']} ({inc['status']})",
                      icon=folium.Icon(color="red" if inc["kind"] == "incident" else "blue",
                                       icon="warning-sign" if inc["kind"] == "incident" else "plus-sign")).add_to(m)
    return m


# ----------------------------------------------------------------- sidebar / state
st.sidebar.markdown("## 🛡️ DisasterShield")
PAGES = ["Overview Dashboard", "Area Monitoring", "Live Disaster Map", "AI Risk Assessment", "Emergency Alerts",
         "Disaster Information Centre", "AI Assistant", "Emergency Response", "Settings"]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
demo = st.sidebar.toggle("Demo mode (sample data)", value=False, help="Synthetic inputs, clearly labelled DEMO.")
if st.sidebar.button("🔄 Refresh data now"):
    st.cache_data.clear()
areas_key = str([(a["name"], a["lat"], a["lon"]) for a in get_areas()])
data, errs = collect(demo, areas_key)
last_ts = max((s["time"] for _, s, _ in data.values()), default=None)
st.sidebar.markdown(
    f"<span class='badge {'demo' if demo else 'live'}'>{'DEMO DATA' if demo else 'LIVE DATA'}</span><br>"
    f"<span class='pulse'></span>Last successful update: {last_ts.strftime('%d %b %H:%M UTC') if last_ts else 'none'}",
    unsafe_allow_html=True)
st.sidebar.caption("Alerts are simulated/test only. Follow IMD, NDMA and local authorities in real emergencies. Emergency: 112")

st.title(page)
if demo:
    st.warning("DEMO MODE - values below are synthetic samples, not real measurements.")
for e in errs:
    st.error(f"Live data unavailable - {e}. Switch on Demo mode in the sidebar to see clearly-labelled sample data.")
if not data and page not in ("Area Monitoring", "Disaster Information Centre", "Settings"):
    st.stop()

# ================================================================= PAGES
if page == "Overview Dashboard":
    alerts = active_alerts(data)
    c = st.columns(4)
    c[0].metric("Monitored areas", len(data))
    c[1].metric("Active alerts (score ≥ 51)", len(alerts))
    c[2].metric("Affected areas", len({a["area"] for a in alerts}))
    c[3].metric("Average risk", f"{np.mean([r['overall'] for _, _, r in data.values()]):.0f}/100")
    sel = st.selectbox("Area", list(data))
    a, s, r = data[sel]
    left, right = st.columns([1, 2])
    left.plotly_chart(gauge(r["overall"], f"Overall risk - {r['level']}"), use_container_width=True, key="ov-g")
    left.caption(f"Main hazard: **{r['top']}**")
    with right:
        m = st.columns(4)
        m[0].metric("Temperature", fmt(s["temp"], " °C", 1))
        m[1].metric("Humidity", fmt(s["hum"], "%"))
        m[2].metric("Rainfall (now)", fmt(s["rain_now"], " mm/h", 1))
        m[3].metric("Wind", fmt(s["wind"], " km/h"))
        m = st.columns(4)
        m[0].metric("Pressure (MSL)", fmt(s["pres"], " hPa"))
        m[1].metric("Water level (discharge ÷ mean)", fmt(s["dis"], "x", 1))
        m[2].metric("Soil saturation", fmt(s["soil_pct"], "%"))
        m[3].metric("Seismic (M2.5+, 30d)", fmt(len(s["quakes"]) if s["quakes"] is not None else None))
    h = s["hourly"][s["hourly"].index >= s["now"] - pd.Timedelta(hours=72)]
    fig = px.line(h.reset_index(), x="time", y=["temp", "rain", "pres"], title="Last 72h + 3-day forecast (UTC)")
    fig.add_vline(x=s["now"].to_pydatetime(), line_dash="dot")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=320)
    st.plotly_chart(fig, use_container_width=True, key="ov-ts")
    st.subheader("All monitored areas")
    st.dataframe(pd.DataFrame([{"Area": n, "Risk": r["overall"], "Level": r["level"], "Main hazard": r["top"],
                                "Data": s["mode"]} for n, (a, s, r) in data.items()]).sort_values("Risk", ascending=False),
                 use_container_width=True, hide_index=True)
    st.caption("Sources: " + " | ".join(f"{k}: {v}" for k, v in s["sources"].items()))
    quakes = [q for _, ss, _ in data.values() for q in (ss["quakes"] or [])]
    st.subheader("Recent seismic events near monitored areas")
    st.dataframe(pd.DataFrame(quakes).drop_duplicates().sort_values("mag", ascending=False).head(10)
                 if quakes else pd.DataFrame({"info": ["No M2.5+ events in range, or seismic data unavailable."]}),
                 use_container_width=True, hide_index=True)

elif page == "Area Monitoring":
    st.subheader("Add a location")
    q = st.text_input("Search city / area (any country)")
    if q:
        try:
            res = requests.get("https://geocoding-api.open-meteo.com/v1/search", timeout=15,
                               params={"name": q, "count": 8}).json().get("results", [])
        except Exception:
            res = []
            st.error("Location search unavailable right now.")
        if res:
            opts = {f"{x['name']}, {x.get('admin1', '')}, {x.get('country', '')}": x for x in res}
            ch = st.selectbox("Matches (city, state, country)", list(opts))
            if st.button("➕ Add to monitoring"):
                x = opts[ch]
                with db() as c:
                    c.execute("INSERT OR REPLACE INTO areas(name,lat,lon) VALUES(?,?,?)", (x["name"], x["latitude"], x["longitude"]))
                st.cache_data.clear()
                st.rerun()
    with st.expander("Use coordinates / my current location"):
        st.caption("Browsers only share location through a permission prompt; paste your coordinates here "
                   "(or add the `streamlit-js-eval` package to read them automatically).")
        n, la, lo = st.text_input("Name"), st.number_input("Latitude", value=19.9975, format="%.4f"), st.number_input("Longitude", value=73.7898, format="%.4f")
        if st.button("Add coordinates") and n:
            with db() as c:
                c.execute("INSERT OR REPLACE INTO areas(name,lat,lon) VALUES(?,?,?)", (n, la, lo))
            st.cache_data.clear()
            st.rerun()
    st.subheader("Monitored areas")
    for a in get_areas():
        c = st.columns([2, 2, 2, 1, 1])
        c[0].write(f"**{a['name']}**  \n{a['lat']:.3f}, {a['lon']:.3f}")
        if a["name"] in data:
            _, s, r = data[a["name"]]
            c[1].write(f"{fmt(s['temp'], '°C', 1)} · RH {fmt(s['hum'], '%')} · wind {fmt(s['wind'], ' km/h')}")
            c[2].markdown(f"<span style='color:{level(r['overall'])[1]}'>● {r['level']} ({r['overall']:.0f})</span> · {r['top']}", unsafe_allow_html=True)
        else:
            c[1].write("no data")
        sub = c[3].checkbox("Subscribe", bool(a["subscribed"]), key=f"sub-{a['name']}")
        if sub != bool(a["subscribed"]):
            with db() as cn:
                cn.execute("UPDATE areas SET subscribed=? WHERE name=?", (int(sub), a["name"]))
            st.rerun()
        if c[4].button("🗑", key=f"rm-{a['name']}"):
            with db() as cn:
                cn.execute("DELETE FROM areas WHERE name=?", (a["name"],))
            st.cache_data.clear()
            st.rerun()

elif page == "Live Disaster Map":
    c = st.columns(3)
    only = c[0].checkbox("Only areas with active alerts")
    qk = c[1].checkbox("Show recent earthquakes", True)
    c[2].markdown("🟢 Low · 🟡 Moderate · 🟠 High · 🔴 Critical · 🟣 earthquake")
    st_folium(risk_map(data, only, quakes=qk), height=620, use_container_width=True, returned_objects=[])

elif page == "AI Risk Assessment":
    sel = st.selectbox("Area", list(data))
    a, s, r = data[sel]
    c = st.columns([1, 2])
    c[0].plotly_chart(gauge(r["overall"], f"Overall - {r['level']}", 280), use_container_width=True, key="ra-o")
    df = pd.DataFrame({"Hazard": list(r["hazards"]), "Score": list(r["hazards"].values())})
    df["Level"] = df["Score"].map(lambda v: level(v)[0])
    fig = px.bar(df, x="Score", y="Hazard", orientation="h", color="Level", range_x=[0, 100],
                 color_discrete_map={n: c_ for _, n, c_ in LEVELS})
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", height=280)
    c[1].plotly_chart(fig, use_container_width=True, key="ra-b")
    cols = st.columns(6)
    for i, (h, v) in enumerate(r["hazards"].items()):
        cols[i].plotly_chart(gauge(v, h, 170), use_container_width=True, key=f"ra-{h}")
    st.markdown(f"<div class='glass'><b>Why:</b> {r['why']}</div>", unsafe_allow_html=True)
    st.markdown(f"**Contributing factors ({r['top']}):** {FACT[r['top']](s)}")
    st.markdown("**Potential impact:** " + "; ".join(INFO[r["top"]]["impacts"]))
    st.markdown("**Recommended preventive actions:** " + "; ".join(ACTIONS[r["top"]]))
    with st.expander("Method & honesty note"):
        st.markdown("""
- **Labels:** hazard scores = *rule-based assessment* (documented weights below). Anomaly % = *machine learning*
  (IsolationForest trained on this location's previous 30 days, scoring the last 24 h). Nothing here is a trained
  supervised disaster-forecast model, and earthquakes are not predictable (seismic score = recent-activity index).
- Each factor is mapped to 0-1 with a linear ramp between a *low* and *high* threshold, then weighted:
  - **Flood** 24h rain 20-150 mm (.30), 72h rain 40-250 (.15), forecast 24h rain 20-150 (.15), soil saturation 50-90% (.10), river discharge 1-5x mean (.30)
  - **Cyclone** wind 40-120 km/h (.5), MSL pressure 1010→985 hPa (.3), 24h rain 30-200 mm (.2)
  - **Earthquake** max magnitude 3-6 (.7), event count 0-15 (.3) within 300 km / 30 days
  - **Wildfire** temp 30-45 °C (.3), dryness 60%→15% RH (.3), wind 10-50 (.15), low 72h rain (.1), FIRMS hotspots 0-10 (.15)
  - **Landslide** 72h rain 50-300 mm (.45), soil saturation 50-90% (.35), 24h rain 30-150 (.2)
  - **Extreme weather** max of heat (35-47 °C), cold (5→-10 °C), wind (40-100 km/h), rain rate (5-40 mm/h)
- Missing inputs are excluded and weights renormalised. **Overall = min(100, 0.75·max hazard + 0.25·mean hazard + 0.10·anomaly%)**.
- Thresholds are illustrative and should be calibrated with local historical data / IMD guidance before operational use.""")

elif page == "Emergency Alerts":
    st.info("🧪 Simulated notification system - alerts are generated from computed risk and are NOT sent to the public.")
    alerts = active_alerts(data)
    if not alerts:
        st.success("No active alerts (all hazard scores below 51).")
    for al in alerts:
        a = data[al["area"]][0]
        st.markdown(f"<div class='glass' style='border-left:6px solid {level(al['score'])[1]}'>"
                    f"<b>{al['level']} - {al['hazard']} - {al['area']}</b> (score {al['score']:.0f}) "
                    f"<span class='badge demo'>{'TEST / DEMO' if demo else 'TEST'}</span><br>"
                    f"{datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}<br><b>Reason:</b> {al['reason']}<br>"
                    f"<b>Immediate precautions:</b> {'; '.join(INFO[al['hazard']]['during'])}</div>", unsafe_allow_html=True)
    c = st.columns(2)
    if c[0].button("Log current alerts to history") and alerts:
        with db() as cn:
            cn.executemany("INSERT INTO alerts(ts,area,hazard,score,level,reason,is_test) VALUES(?,?,?,?,?,?,1)",
                           [(datetime.now(timezone.utc).isoformat(timespec="seconds"), x["area"], x["hazard"], x["score"], x["level"], x["reason"]) for x in alerts])
        st.rerun()
    if c[1].button("Send TEST notification to subscribers"):
        subs = [a["name"] for a in get_areas() if a["subscribed"]]
        st.toast(f"[TEST] Simulated alert delivered for: {', '.join(subs) or 'no subscribed areas'}", icon="🧪")
        st.caption("Real browser-push / e-mail / SMS needs an authorised integration (e.g. SMTP, Twilio, FCM) - intentionally not enabled.")
    st.subheader("Alert history (test records)")
    with db() as cn:
        hist = pd.read_sql_query("SELECT ts,area,hazard,score,level,reason FROM alerts ORDER BY id DESC LIMIT 100", cn)
    st.dataframe(hist, use_container_width=True, hide_index=True)
    if alerts:
        st_folium(risk_map(data, True), height=380, use_container_width=True, returned_objects=[], key="alert-map")

elif page == "Disaster Information Centre":
    h = st.selectbox("Disaster", list(INFO))
    d = INFO[h]
    st.markdown(f"**1. What is it?** {d['what']}\n\n**2. Why does it happen?** {d['why']}")
    for i, (k, t) in enumerate([("signs", "Early warning signs"), ("causes", "Possible environmental causes"),
                                ("impacts", "Potential impacts"), ("before", "Before-disaster safety"),
                                ("during", "During-disaster safety"), ("after", "After-disaster safety")], start=3):
        with st.expander(f"{i}. {t}", expanded=i < 5):
            st.markdown("\n".join(f"- {x}" for x in d[k]))
    with st.expander("9. Emergency preparedness checklist"):
        for x in d["checklist"]:
            st.checkbox(x, key=f"ck-{h}-{x}")
    st.caption("General guidance - always follow NDMA (ndma.gov.in), IMD (mausam.imd.gov.in) and local authority instructions.")
    st.subheader("💬 Preparedness chatbot")
    chat_ui("chat-info", data, ["What should be in an emergency kit?", "What to do during an earthquake?"])

elif page == "AI Assistant":
    st.caption("Uses an LLM through a server-side key if ANTHROPIC_API_KEY is configured; otherwise a rule-based fallback.")
    names = list(data)
    chat_ui("chat-ai", data, [f"Why is {names[0]} showing its current risk?", "Summarise active alerts", "Compare all locations"])

elif page == "Emergency Response":
    st.warning("All incidents, hospitals and resources below are SAMPLE records - not verified real-world data.")
    base = data.get("Nashik", next(iter(data.values())))[0]
    inc = pd.DataFrame([
        dict(id="INC-001", type="Flood (sample)", area=base["name"], status="Reported", lat=base["lat"] + .05, lon=base["lon"] - .04, kind="incident"),
        dict(id="INC-002", type="Landslide (sample)", area="Satara", status="Responding", lat=17.70, lon=73.99, kind="incident"),
        dict(id="INC-003", type="Road blockage (sample)", area="Pune", status="Resolved", lat=18.55, lon=73.80, kind="incident")])
    res = pd.DataFrame([
        dict(id="RES-01", type="Hospital (sample)", area=base["name"], status="Available", lat=base["lat"] - .02, lon=base["lon"] + .03, kind="resource"),
        dict(id="RES-02", type="NDRF team (sample)", area="Pune", status="Available", lat=18.48, lon=73.90, kind="resource")])
    st.subheader("Incident tracking (editable status)")
    inc = st.data_editor(inc[["id", "type", "area", "status", "lat", "lon", "kind"]], hide_index=True, use_container_width=True,
                         column_config={"status": st.column_config.SelectboxColumn(options=["Reported", "Responding", "Resolved"])})
    st.subheader("Resources")
    st.dataframe(res, hide_index=True, use_container_width=True)
    recs = [f"{x['area']}: pre-position resources for {x['hazard']} ({x['level']})" for x in active_alerts(data)[:5]]
    st.markdown("**Response recommendations:**\n" + ("\n".join(f"- {r}" for r in recs) if recs else "- No active alerts."))
    st_folium(risk_map(data, incidents=pd.concat([inc, res]).to_dict("records")), height=450,
              use_container_width=True, returned_objects=[], key="resp-map")

elif page == "Settings":
    st.subheader("Integrations status")
    st.table(pd.DataFrame([
        {"Item": "Open-Meteo weather / flood", "Status": "No key needed"},
        {"Item": "USGS earthquakes", "Status": "No key needed"},
        {"Item": "NASA FIRMS (FIRMS_MAP_KEY)", "Status": "configured" if secret("FIRMS_MAP_KEY") else "not set"},
        {"Item": "LLM (ANTHROPIC_API_KEY)", "Status": "configured" if secret("ANTHROPIC_API_KEY") else "not set - fallback"},
        {"Item": "Database", "Status": f"SQLite file '{DB}' (swap db() for PostgreSQL/Supabase in production)"}]))
    st.info("Keys go in `.streamlit/secrets.toml` or Streamlit Cloud > Settings > Secrets - never in code.")
    st.markdown("**IoT-ready:** implement `sensor_overrides(area)` in `app.py` to merge live gauge / sensor readings "
                "(rain, soil, river level) into each snapshot; replace `score_hazards` with a trained model when available.")
    if st.button("Reset database to sample areas"):
        os.remove(DB) if os.path.exists(DB) else None
        st.cache_data.clear()
        st.rerun()
