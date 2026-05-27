import os

import requests
import streamlit as st

st.set_page_config(page_title="IPL Match Predictor", layout="centered")
st.title("IPL Live Streaming Prediction")
st.markdown("---")
st.write("Enterprise MLOps multi-model inference system")

st.sidebar.header("Real-time match simulation feed")
pp_runs = st.sidebar.slider("Powerplay runs", 15, 130, 46)
pp_wickets = st.sidebar.slider("Powerplay wickets lost", 0, 10, 1)
mid_crr = st.sidebar.slider("Middle overs current run rate", 3.5, 18.0, 7.8)
venue_avg = st.sidebar.number_input("Venue average 1st innings score", 120, 260, 159)
rolling_avg = st.sidebar.number_input("Batting team rolling 3-match avg runs", 100, 260, 159)
h2h_score = st.sidebar.number_input("Head-to-head historic avg scoring", 100, 260, 156)

backend_api_url = os.getenv(
    "BACKEND_API_URL", "http://localhost:8000/predict/all_tracker"
)

if st.button("Run full prediction suite", use_container_width=True):
    payload_packet = {
        "pp_runs": float(pp_runs),
        "pp_wickets": float(pp_wickets),
        "mid_crr": float(mid_crr),
        "venue_avg_score": float(venue_avg),
        "batting_team_rolling": float(rolling_avg),
        "h2h_avg": float(h2h_score),
    }

    with st.spinner("Sending features to inference services..."):
        try:
            response = requests.post(backend_api_url, json=payload_packet, timeout=10)
            if response.status_code == 200:
                result = response.json()

                st.balloons()
                st.success("All 3 inference channels completed successfully")
                st.markdown("---")

                grid_c1, grid_c2, grid_c3 = st.columns(3)
                with grid_c1:
                    st.metric(
                        label="Model 1 winner",
                        value=result["model_1_outcome"]["predicted_winner"],
                        delta=f'{result["model_1_outcome"]["team1_win_probability_pct"]}% team1 win',
                    )
                with grid_c2:
                    st.metric(
                        label="Model 2 innings-1",
                        value=f'{result["model_2_innings1"]["projected_first_innings_runs"]} runs',
                    )
                with grid_c3:
                    st.metric(
                        label="Model 3 chase",
                        value=f'{result["model_3_chase"]["projected_final_chase_score"]} runs',
                        delta=f'Node: {result["model_3_chase"]["champion_architecture"]}',
                    )
            else:
                st.error(f"API returned error status code: {response.status_code}")
        except Exception as api_error:
            st.error(f"Connection timeout / request failed: {api_error}")
