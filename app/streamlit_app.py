"""
Stage 4 -- Smart Schedule Optimizer Web App
Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import sys
import os
import calendar as cal_module
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT        = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

# ── Alberta holiday calendar ──────────────────────────────────────────────────
def _easter_sunday(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = (h + l - 7 * m + 114) % 31 + 1
    return date(year, month, day)


def get_alberta_holidays(year: int) -> dict:
    h = {}
    h[date(year, 1, 1)]   = "New Year's Day"
    h[date(year, 7, 1)]   = "Canada Day"
    h[date(year, 11, 11)] = "Remembrance Day"
    h[date(year, 12, 25)] = "Christmas Day"
    h[date(year, 12, 26)] = "Boxing Day"
    mondays = [date(year, 2, d) for d in range(1, 29) if date(year, 2, d).weekday() == 0]
    if len(mondays) >= 3:
        h[mondays[2]] = "Family Day (Alberta)"
    easter = _easter_sunday(year)
    h[easter - timedelta(days=2)] = "Good Friday"
    h[easter + timedelta(days=1)] = "Easter Monday"
    for day in range(24, 17, -1):
        vd = date(year, 5, day)
        if vd.weekday() == 0:
            h[vd] = "Victoria Day"; break
    for day in range(1, 8):
        hd = date(year, 8, day)
        if hd.weekday() == 0:
            h[hd] = "Heritage Day (Alberta)"; break
    for day in range(1, 8):
        ld = date(year, 9, day)
        if ld.weekday() == 0:
            h[ld] = "Labour Day"; break
    mondays = [date(year, 10, d) for d in range(1, 32) if date(year, 10, d).weekday() == 0]
    if len(mondays) >= 2:
        h[mondays[1]] = "Thanksgiving"
    return h


def get_week_calendar(week_start: date) -> dict:
    year     = week_start.year
    holidays = get_alberta_holidays(year)
    holidays.update(get_alberta_holidays(year + 1))
    calendar = {}
    for offset in range(7):
        d = week_start + timedelta(days=offset)
        calendar[d] = {"holiday": holidays.get(d), "long_weekend": False}
    for offset in range(7):
        d = week_start + timedelta(days=offset)
        if d.weekday() == 5:
            if d + timedelta(days=2) in holidays: calendar[d]["long_weekend"] = True
            if d - timedelta(days=1) in holidays: calendar[d]["long_weekend"] = True
        elif d.weekday() == 6:
            if d + timedelta(days=1) in holidays: calendar[d]["long_weekend"] = True
            if d - timedelta(days=2) in holidays: calendar[d]["long_weekend"] = True
    return calendar


# ── Event definitions ─────────────────────────────────────────────────────────
EVENT_PRESETS = {
    "None":                        1.00,
    "Long Weekend":                1.12,
    "School Day Off / PA Day":     1.15,
    "School Break Week":           1.20,
    "Community Event Nearby":      1.25,
    "Local Festival / Parade":     1.30,
    "Major Sporting Event":        1.20,
    "Construction / Road Closure": 0.80,
    "Nearby Competitor Closed":    1.15,
    "Custom":                      1.00,
}
EVENT_ICONS = {
    "None": "", "Long Weekend": "📅",
    "School Day Off / PA Day": "🎒", "School Break Week": "🎒",
    "Community Event Nearby": "🎪", "Local Festival / Parade": "🎉",
    "Major Sporting Event": "🏆", "Construction / Road Closure": "🚧",
    "Nearby Competitor Closed": "🔒", "Custom": "✏️",
}
WEATHER_OPTIONS = ["sunny", "cloudy", "rainy", "snowy"]
WEATHER_ICONS   = {"sunny": "☀️", "cloudy": "☁️", "rainy": "🌧️", "snowy": "❄️"}
PEAK_HOURS      = {12, 18}
REL_ICON        = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Schedule Optimizer",
    page_icon="📅",
    layout="wide",
)

# Light CSS: make calendar buttons taller and centred
st.markdown("""
<style>
div[data-testid="stColumns"] button {
    height: 56px;
    font-size: 0.80rem;
    line-height: 1.3;
    white-space: pre-line;
}
</style>
""", unsafe_allow_html=True)

# ── Load optimizer (cached) ───────────────────────────────────────────────────
@st.cache_resource
def load_optimizer():
    import stage3_optimizer
    return stage3_optimizer

opt = load_optimizer()

# ── Session state defaults ────────────────────────────────────────────────────
today = date.today()
_defaults = {
    "view_year":     2024,
    "view_month":    1,
    "selected_date": None,
    "schedule_df":   None,
    "day_proj":      None,
    "week_start":    None,
    "weeks_ahead":   3,
    "apply_growth":  False,
    "day_overrides": {},
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📅 Smart Schedule Optimizer")
st.caption(
    "ML-powered QSR staff scheduling  ·  LY context (±3 weeks)  ·  "
    "trend adoption  ·  reliability scoring  ·  Alberta holidays auto-detected"
)
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
# MONTHLY CALENDAR
# ═════════════════════════════════════════════════════════════════════════════
view_year  = st.session_state.view_year
view_month = st.session_state.view_month

# Precompute holidays for displayed year + neighbours
all_holidays = {}
for yr in [view_year - 1, view_year, view_year + 1]:
    all_holidays.update(get_alberta_holidays(yr))

# Precompute long-weekend days for the entire displayed month
_, num_days = cal_module.monthrange(view_year, view_month)
long_weekend_days = set()
for dn in range(1, num_days + 1):
    d = date(view_year, view_month, dn)
    if d.weekday() == 5:
        if d + timedelta(days=2) in all_holidays: long_weekend_days.add(d)
        if d - timedelta(days=1) in all_holidays: long_weekend_days.add(d)
    elif d.weekday() == 6:
        if d + timedelta(days=1) in all_holidays: long_weekend_days.add(d)
        if d - timedelta(days=2) in all_holidays: long_weekend_days.add(d)

# Selected week boundaries
sel_date  = st.session_state.selected_date
sel_week_start = (sel_date - timedelta(days=sel_date.weekday())) if sel_date else None

# ── Month navigation bar ──────────────────────────────────────────────────────
nav_l, nav_c, nav_r = st.columns([1, 5, 1])

with nav_l:
    if st.button("◀  Prev", use_container_width=True):
        if view_month == 1:
            st.session_state.view_month = 12
            st.session_state.view_year  = view_year - 1
        else:
            st.session_state.view_month -= 1
        st.session_state.schedule_df = None
        st.rerun()

with nav_c:
    month_label = date(view_year, view_month, 1).strftime("%B  %Y")
    st.markdown(
        f"<h3 style='text-align:center; margin:0'>{month_label}</h3>",
        unsafe_allow_html=True,
    )

with nav_r:
    if st.button("Next  ▶", use_container_width=True):
        if view_month == 12:
            st.session_state.view_month = 1
            st.session_state.view_year  = view_year + 1
        else:
            st.session_state.view_month += 1
        st.session_state.schedule_df = None
        st.rerun()

# ── Day-of-week header ────────────────────────────────────────────────────────
hdr_cols = st.columns(7)
for i, dname in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
    colour = "#555" if i < 5 else "#c0392b"
    hdr_cols[i].markdown(
        f"<p style='text-align:center;font-weight:700;color:{colour};margin:0'>{dname}</p>",
        unsafe_allow_html=True,
    )

# ── Calendar grid ─────────────────────────────────────────────────────────────
month_grid = cal_module.monthcalendar(view_year, view_month)

for week_row in month_grid:
    row_cols = st.columns(7)
    for col_i, day_num in enumerate(week_row):
        if day_num == 0:
            row_cols[col_i].write("")
            continue

        d       = date(view_year, view_month, day_num)
        is_hol  = d in all_holidays
        is_lw   = d in long_weekend_days
        in_week = bool(sel_week_start and sel_week_start <= d < sel_week_start + timedelta(days=7))

        # Cell label — date number + indicator
        if is_hol:
            label = f"{day_num}\n🇨🇦"
        elif is_lw:
            label = f"{day_num}\n📅"
        else:
            label = str(day_num)

        btn_type = "primary" if in_week else "secondary"

        if row_cols[col_i].button(
            label, key=f"cal_{d}",
            type=btn_type,
            use_container_width=True,
        ):
            st.session_state.selected_date = d
            # Clear previous schedule if switching to a different week
            new_week = d - timedelta(days=d.weekday())
            if new_week != st.session_state.week_start:
                st.session_state.schedule_df = None
            st.rerun()

# Legend
st.caption("🇨🇦 Alberta Statutory Holiday  ·  📅 Long Weekend  ·  Blue = selected week  ·  Click any day to load its week")

# ── Gate: no day selected yet ─────────────────────────────────────────────────
if sel_date is None:
    st.info("👆 Click any day on the calendar above to view or generate its week schedule.")
    st.stop()

week_start = sel_date - timedelta(days=sel_date.weekday())
week_end   = week_start + timedelta(days=6)
week_cal   = get_week_calendar(week_start)

st.divider()
st.subheader(f"Week of  {week_start.strftime('%B %d')} – {week_end.strftime('%B %d, %Y')}")

# Holiday summary for the week
week_hols = [(d, v["holiday"]) for d, v in week_cal.items() if v["holiday"]]
week_lws  = [d for d, v in week_cal.items() if v["long_weekend"] and not v["holiday"]]
if week_hols or week_lws:
    cols_hol = st.columns(max(len(week_hols) + len(week_lws), 1))
    ci = 0
    for d, hname in week_hols:
        cols_hol[ci].success(f"🇨🇦 **{d.strftime('%A %b %d')}**\n{hname}")
        ci += 1
    for d in week_lws:
        cols_hol[ci].warning(f"📅 **{d.strftime('%A %b %d')}**\nLong Weekend (+12% suggested)")
        ci += 1

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR — controls for selected week
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("⚙️ Schedule Controls")
    st.caption(f"**{week_start.strftime('%b %d')} – {week_end.strftime('%b %d, %Y')}**")

    weeks_ahead = st.slider("Weeks scheduled in advance", 1, 3, 3,
                            help="More weeks ahead = lower reliability score.")

    apply_growth = st.checkbox(
        "Apply +5% YoY growth factor", value=False,
        help="Best used when scheduling 2025 using 2024 as LY.",
    )

    st.divider()

    # ── Weather forecast per day ───────────────────────────────────────────
    with st.expander("🌤️ Weather Forecast (per day)"):
        st.caption("Defaults to seasonal average.")
        day_weather = {}
        for offset in range(7):
            d       = week_start + timedelta(days=offset)
            default = opt.SEASONAL_WEATHER[d.month]
            chosen  = st.selectbox(
                d.strftime("%A %b %d"),
                WEATHER_OPTIONS,
                index=WEATHER_OPTIONS.index(default),
                format_func=lambda w: f"{WEATHER_ICONS[w]} {w.capitalize()}",
                key=f"wx_{week_start}_{d}",
            )
            day_weather[d] = chosen

    # ── Promotion days ─────────────────────────────────────────────────────
    with st.expander("🏷️ Promotion Days"):
        st.caption("Check days running a promo deal.")
        day_promos = {}
        for offset in range(7):
            d = week_start + timedelta(days=offset)
            day_promos[d] = 1 if st.checkbox(
                d.strftime("%A %b %d"), key=f"promo_{week_start}_{d}"
            ) else 0

    # ── Events & disruptions ───────────────────────────────────────────────
    with st.expander("📅 Events & Disruptions (per day)"):
        st.caption("Select events affecting customer traffic.")
        day_events = {}
        event_keys = list(EVENT_PRESETS.keys())
        for offset in range(7):
            d        = week_start + timedelta(days=offset)
            cal_info = week_cal.get(d, {})

            if cal_info.get("holiday"):
                st.markdown(f"**{d.strftime('%A %b %d')}** 🇨🇦 *{cal_info['holiday']}*")
                st.caption("Holiday handled by ML model — no multiplier needed.")
                day_events[d] = {"label": "None", "multiplier": 1.0}
                continue

            auto_idx = event_keys.index("Long Weekend") if cal_info.get("long_weekend") else 0
            st.markdown(
                f"**{d.strftime('%A %b %d')}**"
                + (" 📅 *Long Weekend*" if cal_info.get("long_weekend") else "")
            )
            col_a, col_b = st.columns([2, 1])
            with col_a:
                etype = st.selectbox(
                    "Event", event_keys, index=auto_idx,
                    format_func=lambda e: f"{EVENT_ICONS[e]} {e}" if EVENT_ICONS.get(e) else e,
                    key=f"evt_{week_start}_{d}",
                    label_visibility="collapsed",
                )
            with col_b:
                default_pct = int((EVENT_PRESETS[etype] - 1) * 100)
                custom_pct  = st.number_input(
                    "Impact %", min_value=-50, max_value=100,
                    value=default_pct, step=5,
                    key=f"evt_pct_{week_start}_{d}",
                    help="+% = more traffic, -% = less traffic",
                )
            day_events[d] = {"label": etype, "multiplier": round(1 + custom_pct / 100, 3)}
            if etype != "None" or custom_pct != 0:
                st.caption(
                    f"{EVENT_ICONS.get(etype,'')} {etype}  →  "
                    f"{'+' if custom_pct >= 0 else ''}{custom_pct}% traffic"
                )

    st.divider()
    run_btn = st.button("Generate Schedule", type="primary", use_container_width=True)

    st.divider()
    st.caption("**Model:** RandomForest (R²=0.91, MAE=0.22)")
    st.caption("**LY reference:** 2023 same ISO week ± 3 weeks")
    st.caption("**Reliability decay:** 3% per week ahead")

# ── Run schedule ──────────────────────────────────────────────────────────────
if run_btn:
    day_overrides = {}
    for i in range(7):
        d = week_start + timedelta(days=i)
        day_overrides[d] = {
            "weather":          day_weather[d],
            "is_promotion":     day_promos[d],
            "event_multiplier": day_events[d]["multiplier"],
            "event_label":      day_events[d]["label"],
        }

    opt.APPLY_GROWTH = apply_growth
    with st.spinner(f"Generating schedule for week of {week_start}…"):
        schedule_df, day_proj = opt.schedule_week(week_start, weeks_ahead, day_overrides)

    st.session_state.schedule_df   = schedule_df
    st.session_state.day_proj      = day_proj
    st.session_state.week_start    = week_start
    st.session_state.weeks_ahead   = weeks_ahead
    st.session_state.apply_growth  = apply_growth
    st.session_state.day_overrides = day_overrides
    st.success(f"Schedule generated — {len(schedule_df)} shifts across 7 days.")

# ── Gate: schedule not yet generated for this week ────────────────────────────
if (st.session_state.schedule_df is None or
        st.session_state.week_start != week_start):
    st.info("Configure settings in the sidebar and click **Generate Schedule**.")
    st.stop()

df            = st.session_state.schedule_df
day_proj      = st.session_state.day_proj
weeks_ahead   = st.session_state.weeks_ahead
grow_on       = st.session_state.apply_growth
day_overrides = st.session_state.day_overrides or {}

# ═════════════════════════════════════════════════════════════════════════════
# TABS — Schedule · Day Detail · Employee Hours · Download
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab_day, tab2, tab3, tab4 = st.tabs([
    "📋 Schedule",
    "📈 Day Detail",
    "📊 Projections",
    "👥 Employee Hours",
    "📥 Download",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    growth_badge = "🟢 +5% growth ON" if grow_on else "⚪ Growth OFF (LY + trend only)"
    st.subheader(f"Week of {week_start}  ·  {weeks_ahead} week(s) ahead  ·  {growth_badge}")

    for day_date, proj in day_proj.items():
        day_name = day_date.strftime("%A")
        day_df   = df[df["date"] == day_date.strftime("%Y-%m-%d")].sort_values("start_hour")
        icon     = REL_ICON.get(proj["rel_label"], "⚪")
        trend_s  = (f"+{(proj['trend']-1)*100:.1f}%" if proj["trend"] >= 1.0
                    else f"{(proj['trend']-1)*100:.1f}%")

        # Event badge
        evt_lbl  = proj.get("event_label", "None")
        evt_mult = proj.get("event_multiplier", 1.0)
        evt_pct  = int(round((evt_mult - 1) * 100))
        evt_str  = (f"  ·  {EVENT_ICONS.get(evt_lbl,'')} {evt_lbl} ({'+' if evt_pct>=0 else ''}{evt_pct}%)"
                    if evt_lbl != "None" else "")

        header = (
            f"{icon} **{day_name} {day_date}**  ·  "
            f"{proj['peak_staff']} staff peak  ·  "
            f"~{proj['peak_txn']} txn/hr  ·  "
            f"trend {trend_s}  ·  "
            f"reliability {proj['rel_pct']} [{proj['rel_label']}]"
            f"{evt_str}"
        )
        expand = day_name in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        with st.expander(header, expanded=expand):
            display = day_df[["name","role","shift_type","start_hour","end_hour","hours"]].copy()
            display.columns = ["Name","Role","Shift Type","Start","End","Hrs"]
            display["Start"] = display["Start"].apply(lambda h: f"{int(h):02d}:00")
            display["End"]   = display["End"].apply(lambda h: f"{int(h):02d}:00")
            display = display.reset_index(drop=True)
            st.table(display)
            st.caption(f"Total shifts today: {len(day_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# TAB — DAY DETAIL
# ─────────────────────────────────────────────────────────────────────────────
with tab_day:
    st.subheader("Day Detail View")

    day_options    = {d.strftime("%A  %Y-%m-%d"): d for d in day_proj.keys()}
    selected_label = st.selectbox("Select day", list(day_options.keys()))
    selected_day   = day_options[selected_label]

    opt.APPLY_GROWTH = grow_on
    day_ov = day_overrides.get(selected_day, {})
    hourly = opt.predict_day(
        selected_day, weeks_ahead,
        weather=day_ov.get("weather"),
        is_promotion=day_ov.get("is_promotion", 0),
        event_multiplier=day_ov.get("event_multiplier", 1.0),
    )

    hours       = list(range(6, 24))
    hour_labels = [f"{h:02d}:00" for h in hours]
    proj_txn    = [hourly[h]["txn_proj"]   for h in hours]
    ml_staff    = [hourly[h]["staff_raw"]  for h in hours]
    base_staff  = [hourly[h]["staff_base"] for h in hours]
    rel_scores  = [hourly[h]["rel_pct"]    for h in hours]

    day_df_det = df[df["date"] == selected_day.strftime("%Y-%m-%d")].copy()
    actual_staff = [
        int(((day_df_det["start_hour"] <= h) & (day_df_det["end_hour"] > h)).sum())
        for h in hours
    ]

    proj      = day_proj[selected_day]
    rel_col   = REL_ICON.get(proj["rel_label"], "⚪")
    trend_s   = (f"+{(proj['trend']-1)*100:.1f}%" if proj["trend"] >= 1.0
                 else f"{(proj['trend']-1)*100:.1f}%")
    evt_label = proj.get("event_label", "None")
    evt_mult  = proj.get("event_multiplier", 1.0)
    evt_pct   = int(round((evt_mult - 1) * 100))
    evt_icon  = EVENT_ICONS.get(evt_label, "")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Transactions/hr", proj["peak_txn"])
    c2.metric("Peak Staff Needed",    proj["peak_staff"])
    c3.metric("Trend (3 weeks)",      trend_s)
    c4.metric(f"Reliability {rel_col}", f"{proj['rel_pct']} [{proj['rel_label']}]")

    wx_icon  = {"sunny": "☀️", "cloudy": "☁️", "rainy": "🌧️", "snowy": "❄️"}
    wx_label = proj.get("weather", "sunny")
    promo_lbl = "🏷️ Promo ON" if proj.get("is_promotion") else ""
    evt_str   = (f"{evt_icon} **{evt_label}** ({'+' if evt_pct>=0 else ''}{evt_pct}% traffic)"
                 if evt_label != "None" else "No events")
    badges = "  ·  ".join(filter(None, [promo_lbl, evt_str]))
    st.caption(
        f"Trend: **{trend_s}**  ·  "
        f"Weather: {wx_icon.get(wx_label,'')} **{wx_label.capitalize()}**  ·  "
        f"{badges}  ·  "
        f"Peak hours: 🔴 **12:00** (lunch) and **18:00** (dinner)"
    )

    if evt_label != "None" and evt_mult != 1.0:
        base_peak = max(hourly[h]["staff_base"] for h in hourly)
        adj_peak  = max(hourly[h]["staff_raw"]  for h in hourly)
        direction = "up" if adj_peak > base_peak else "down"
        st.info(
            f"{evt_icon} **{evt_label}** adjusts peak staffing from "
            f"**{base_peak}** → **{adj_peak}** staff "
            f"({'+' if evt_pct>=0 else ''}{evt_pct}% traffic, rounded {direction})."
        )

    st.divider()

    # Combined line chart
    st.markdown("**Hourly Projected Transactions/hr · Projected Staff/hr · Scheduled Staff/hr**")
    st.caption("🔴 Peak hours: 12:00 (lunch) and 18:00 (dinner)")

    chart_data = {
        "Projected Transactions/hr": proj_txn,
        "ML Staff (base)":           base_staff,
        "Scheduled Staff/hr":        actual_staff,
    }
    if evt_label != "None" and evt_mult != 1.0:
        chart_data["ML Staff (event-adjusted)"] = ml_staff

    st.line_chart(pd.DataFrame(chart_data, index=hour_labels), height=300)

    # Gap alerts
    gap_rows = []
    for i, h in enumerate(hours):
        diff = actual_staff[i] - ml_staff[i]
        if diff < 0:
            gap_rows.append(f"**{hour_labels[i]}** — understaffed by {abs(diff)} ({ml_staff[i]} needed, {actual_staff[i]} scheduled)")
        elif diff > 2:
            gap_rows.append(f"{hour_labels[i]} — over-staffed by {diff}")
    if gap_rows:
        with st.expander("⚠️ Staffing gaps detected"):
            for r in gap_rows:
                st.markdown(f"- {r}")

    st.divider()
    st.markdown("**Full Hourly Breakdown**")

    hourly_table = pd.DataFrame({
        "Hour":               hour_labels,
        "Peak":               ["🔴 PEAK" if h in PEAK_HOURS else "" for h in hours],
        "Proj. Txn/hr":       proj_txn,
        "Proj. Staff/hr":     ml_staff,
        "Scheduled Staff/hr": actual_staff,
        "Gap":                [a - m for a, m in zip(actual_staff, ml_staff)],
        "Reliability":        rel_scores,
    })

    def style_hourly(row):
        styles = [""] * len(row)
        h_idx  = hour_labels.index(row["Hour"])
        if hours[h_idx] in PEAK_HOURS:
            styles = ["background-color: #fff3e0"] * len(row)
        gi = hourly_table.columns.get_loc("Gap")
        if row["Gap"] < 0:  styles[gi] = "background-color: #f8d7da; font-weight: bold"
        elif row["Gap"] > 2: styles[gi] = "background-color: #fff3cd"
        return styles

    st.table(hourly_table.reset_index(drop=True))
    st.caption("Gap = Scheduled − Projected.  Negative Gap = understaffed.  Peak hours: 12:00 and 18:00.")

    st.divider()
    st.markdown("**Staff & Manager Schedule Timeline**")
    st.caption("🟩 Working  ·  🟧 Peak hour  ·  ★ = Manager / Supervisor")

    if day_df_det.empty:
        st.info("No staff scheduled on this day.")
    else:
        grid_rows = {}
        for _, row in day_df_det.sort_values(
            "role", key=lambda s: s.map({"Manager":0,"Shift Supervisor":1}).fillna(2)
        ).iterrows():
            is_mgr = row["role"] in {"Manager","Shift Supervisor"}
            label  = f"{'★ ' if is_mgr else '  '}{row['name']}  [{row['shift_type']}]"
            slots  = {}
            for h, lbl in zip(hours, hour_labels):
                in_shift = int(row["start_hour"]) <= h < int(row["end_hour"])
                slots[lbl] = ("PEAK" if h in PEAK_HOURS else "ON") if in_shift else ""
            grid_rows[label] = slots

        grid_df = pd.DataFrame(grid_rows).T
        grid_df.columns = hour_labels
        display_grid = grid_df.replace({"ON":"■","PEAK":"▰","":""})

        def color_gantt(val):
            if val == "▰": return "background-color:#f4a261;color:#000;text-align:center"
            if val == "■": return "background-color:#52b788;color:#000;text-align:center"
            return "background-color:#f8f9fa;color:#f8f9fa"

        st.dataframe(display_grid.style.map(color_gantt), width='stretch')
        st.caption("▰ = peak hour (amber)  ·  ■ = regular (green)  ·  ★ = Manager/Supervisor")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — PROJECTIONS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Daily Demand Projections")
    st.caption(
        "Based on: LY same week · LY ±3 week context · last 3 week trend · "
        + ("**+5% YoY growth applied**" if grow_on else "no growth adjustment")
    )

    all_staff = [p["peak_staff"] for p in day_proj.values()]
    all_cust  = [p["peak_cust"]  for p in day_proj.values()]
    all_rel   = [p["rel_pct"]    for p in day_proj.values()]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Avg Peak Staff",        f"{np.mean(all_staff):.1f}")
    c2.metric("Avg Peak Customers/hr", f"{np.mean(all_cust):.0f}")
    c3.metric("Max Peak Day",          f"{max(all_staff)} staff")
    c4.metric("Avg Reliability",       all_rel[len(all_rel)//2])
    st.divider()

    proj_rows = []
    for day_date, proj in day_proj.items():
        trend_s = (f"+{(proj['trend']-1)*100:.1f}%" if proj["trend"] >= 1.0
                   else f"{(proj['trend']-1)*100:.1f}%")
        evt_lbl = proj.get("event_label","None")
        proj_rows.append({
            "Day":            day_date.strftime("%A"),
            "Date":           str(day_date),
            "Peak Staff":     proj["peak_staff"],
            "Cust/hr (peak)": proj["peak_cust"],
            "Txn/hr (peak)":  proj["peak_txn"],
            "Trend (3wk)":    trend_s,
            "Event":          f"{EVENT_ICONS.get(evt_lbl,'')} {evt_lbl}" if evt_lbl != "None" else "",
            "Reliability":    proj["rel_pct"],
            "Confidence":     proj["rel_label"],
        })
    proj_df = pd.DataFrame(proj_rows)

    def color_confidence(val):
        return {"HIGH":"background-color:#d4edda;color:#155724",
                "MEDIUM":"background-color:#fff3cd;color:#856404",
                "LOW":"background-color:#f8d7da;color:#721c24"}.get(val,"")

    st.dataframe(proj_df.style.map(color_confidence, subset=["Confidence"]),
                 width='stretch', hide_index=True)
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Peak Customer & Transaction Volume by Day**")
        st.bar_chart(proj_df.set_index("Day")[["Cust/hr (peak)","Txn/hr (peak)"]])
    with col_b:
        st.markdown("**Peak Staff Required by Day**")
        st.bar_chart(proj_df.set_index("Day")[["Peak Staff"]])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — EMPLOYEE HOURS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Weekly Hours Summary")
    hours_df = (
        df.groupby(["emp_id","name","role"])["hours"]
        .agg(Total_Hours="sum", Shifts="count")
        .reset_index()
        .sort_values("Total_Hours", ascending=False)
        .rename(columns={"emp_id":"ID","name":"Name","role":"Role"})
    )
    emp_data = pd.read_csv(os.path.join(ROOT, "data", "qsr_employees.csv"))
    merged   = hours_df.merge(emp_data[["emp_id","max_hours"]], left_on="ID", right_on="emp_id", how="left")
    merged["Max Hours"]  = merged["max_hours"]
    merged["Utilization %"] = (merged["Total_Hours"] / merged["Max Hours"] * 100).round(1)

    def color_util(val):
        if val >= 90: return "background-color:#f8d7da"
        if val >= 75: return "background-color:#fff3cd"
        return ""

    st.dataframe(
        merged[["ID","Name","Role","Total_Hours","Shifts","Max Hours","Utilization %"]]
        .style.map(color_util, subset=["Utilization %"]),
        width='stretch', hide_index=True,
    )
    at_limit = merged[merged["Utilization %"] >= 90]
    if not at_limit.empty:
        st.warning(f"⚠️ Near/at weekly hour limit: {', '.join(at_limit['Name'].tolist())}")

    st.divider()
    st.markdown("**Hours by Employee**")
    st.bar_chart(hours_df.set_index("Name")[["Total_Hours"]])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Export Schedule")
    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            "📄 Download Schedule CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"schedule_{week_start}.csv",
            mime="text/csv", use_container_width=True,
        )
        st.caption("Full shift data — one row per shift assignment.")

    with col2:
        grow_note = "ON (+5%)" if grow_on else "OFF (LY + trend only)"
        lines = [
            "=" * 70,
            f"  WEEKLY SCHEDULE -- Week of {week_start}",
            f"  Generated {weeks_ahead} week(s) in advance | Growth factor: {grow_note}",
            "=" * 70,
        ]
        for day_date, proj in day_proj.items():
            day_name  = day_date.strftime("%A")
            day_df2   = df[df["date"] == day_date.strftime("%Y-%m-%d")].sort_values("start_hour")
            trend_s   = (f"+{(proj['trend']-1)*100:.1f}%" if proj["trend"] >= 1.0
                         else f"{(proj['trend']-1)*100:.1f}%")
            evt_lbl   = proj.get("event_label","None")
            evt_mult  = proj.get("event_multiplier",1.0)
            evt_pct   = int(round((evt_mult-1)*100))
            evt_note  = (f"  | {EVENT_ICONS.get(evt_lbl,'')} {evt_lbl} ({'+' if evt_pct>=0 else ''}{evt_pct}%)"
                         if evt_lbl != "None" else "")
            lines += [
                f"\n{day_name.upper()} {day_date}  |  ISO Week {proj['iso_week']}",
                f"  Projection: ~{proj['peak_cust']} cust/hr | ~{proj['peak_txn']} txn/hr | {proj['peak_staff']} staff needed",
                f"  Reliability: {proj['rel_pct']} [{proj['rel_label']}]  |  Trend: {trend_s}{evt_note}",
                "-" * 70,
                f"  {'Name':<22} {'Role':<20} {'Shift':<8} {'Hours'}",
                f"  {'-'*22} {'-'*20} {'-'*8} {'-'*13}",
            ]
            for _, row in day_df2.iterrows():
                t = f"{int(row['start_hour']):02d}:00–{int(row['end_hour']):02d}:00"
                lines.append(f"  {row['name']:<22} {row['role']:<20} {row['shift_type']:<8} {t}  ({int(row['hours'])}h)")
            lines.append(f"  Total shifts: {len(day_df2)}")

        st.download_button(
            "📝 Download Report TXT",
            data="\n".join(lines).encode("utf-8"),
            file_name=f"schedule_report_{week_start}.txt",
            mime="text/plain", use_container_width=True,
        )
        st.caption("Formatted report with projections and confidence summary.")

    st.divider()
    st.markdown("**Full Schedule Data Preview**")
    st.table(df.reset_index(drop=True))
