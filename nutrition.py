import streamlit as st
from typing import Dict
from user_management import UserManager
from datetime import datetime

ACTIVITY_FACTORS = {
    "Sedentary (little or no exercise)": 1.2,
    "Lightly active (1-3 days/wk)": 1.375,
    "Moderately active (3-5 days/wk)": 1.55,
    "Very active (6-7 days/wk)": 1.725,
    "Super active (twice per day)": 1.9,
}

GOAL_DELTAS = {
    "Maintain weight": 0,
    "Mild fat loss (~-10%)": -0.10,
    "Fat loss (~-20%)": -0.20,
    "Mild muscle gain (~+5%)": 0.05,
    "Muscle gain (~+10%)": 0.10,
}

def estimate_bmr(weight_kg: float, height_cm: float, age: int, sex: str = "male") -> float:
    """Mifflin-St Jeor BMR formula"""
    if sex.lower() == "female":
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
    return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5


def render_nutrition_page() -> None:
    st.markdown("## 🍽️ Diet & Calories")
    user_manager = UserManager()
    user = user_manager.get_current_user()

    # -------------------------
    # Calorie Calculator
    # -------------------------
    with st.expander("Calorie Calculator", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=13, max_value=100, value=(user.age if user else 25))
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=(user.weight if user else 70.0))
            height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=(user.height if user else 170.0))
            sex = st.selectbox("Sex", ["male", "female"], index=0)
        with col2:
            activity = st.selectbox("Activity level", list(ACTIVITY_FACTORS.keys()), index=2)
            goal = st.selectbox("Goal", list(GOAL_DELTAS.keys()), index=1)
            protein_pref = st.selectbox("Diet Preference", ["None", "Vegetarian", "Vegan", "High-protein"])    

        if st.button("Calculate", type="primary"):
            bmr = estimate_bmr(weight, height, int(age), sex)
            tdee = bmr * ACTIVITY_FACTORS[activity]
            target = tdee * (1 + GOAL_DELTAS[goal])
            protein_g = max(1.6 if "gain" in goal.lower() else 1.2, 1.2) * weight
            fats_g = 0.8 * weight
            carbs_kcal = target - (protein_g * 4 + fats_g * 9)
            carbs_g = max(carbs_kcal / 4, 0)

            # Save results to session_state
            st.session_state["calc_result"] = {
                "bmr": bmr, "tdee": tdee, "target": target,
                "protein": protein_g, "fats": fats_g, "carbs": carbs_g,
                "plan": build_sample_plan(int(target), int(protein_g), int(fats_g), int(carbs_g), protein_pref)
            }

    # -------------------------
    # Display Results (OUTSIDE)
    # -------------------------
    if "calc_result" in st.session_state:
        result = st.session_state["calc_result"]

        st.success(f"BMR: {result['bmr']:.0f} kcal • "
                   f"TDEE: {result['tdee']:.0f} kcal • "
                   f"Target: {result['target']:.0f} kcal/day")

        colm = st.columns(3)
        with colm[0]:
            st.metric("Protein", f"{result['protein']:.0f} g")
        with colm[1]:
            st.metric("Fats", f"{result['fats']:.0f} g")
        with colm[2]:
            st.metric("Carbs", f"{result['carbs']:.0f} g")

        st.markdown("### 🥗 Sample Day Plan")
        for meal, items in result["plan"].items():
            with st.expander(meal):
                for line in items:
                    st.markdown(f"- {line}")

    st.markdown("---")

    # -------------------------
    # Calorie Tracker
    # -------------------------
    st.markdown("### 📘 Calorie Tracker")
    if not user:
        st.info("Log in to track your daily calorie intake.")
        return

    with st.form("calorie_log_form"):
        log_cols = st.columns(5)
        with log_cols[0]:
            log_date = st.date_input("Date", value=datetime.now().date())
        with log_cols[1]:
            food = st.text_input("Food / Meal")
        with log_cols[2]:
            calories = st.number_input("Calories", min_value=0, value=0)
        with log_cols[3]:
            protein_g = st.number_input("Protein (g)", min_value=0, value=0)
        with log_cols[4]:
            carbs_g = st.number_input("Carbs (g)", min_value=0, value=0)
        fats_g = st.number_input("Fats (g)", min_value=0, value=0)
        notes = st.text_input("Notes", value="")
        submitted = st.form_submit_button("Add Entry")
        if submitted:
            if not food or calories <= 0:
                st.error("Please enter a food name and positive calories.")
            else:
                user_manager.db.add_calorie_log(
                    user.id, datetime.combine(log_date, datetime.min.time()), food,
                    float(calories), float(protein_g), float(carbs_g), float(fats_g), notes,
                )
                st.success("Entry added.")
                st.experimental_rerun()

    # -------------------------
    # Summary
    # -------------------------
    period = st.selectbox("Summary Period", [7, 14, 30], format_func=lambda d: f"Last {d} days")
    summary = user_manager.db.get_calorie_summary(user.id, days=int(period))
    st.metric("Total Calories", f"{summary['total_kcal']:.0f} kcal")

    if summary["history"]:
        import pandas as pd
        import plotly.express as px
        df = pd.DataFrame(summary["history"]).sort_values("date")
        fig = px.bar(df, x="date", y="calories", title="Daily Calorie Intake")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df, use_container_width=True)


def build_sample_plan(target_kcal: int, protein_g: int, fats_g: int, carbs_g: int, pref: str) -> Dict[str, list]:
    veg = pref in ["Vegetarian", "Vegan"]
    high_pro = pref == "High-protein"
    meals = {
        "Breakfast": [
            ("Oats", 60, 10, 6, 36),
            ("Eggs", 3, 18, 15, 0) if not veg else ("Tofu scramble", 150, 16, 9, 4),
            ("Berries", 100, 1, 0, 14),
        ],
        "Lunch": [
            ("Chicken breast", 150, 46, 3, 0) if not veg else ("Lentils", 200, 18, 1, 36),
            ("Rice", 200, 4, 1, 56),
            ("Veggies", 200, 4, 0, 14),
        ],
        "Snack": [
            ("Greek yogurt", 170, 17, 0, 6) if not veg else ("Soy yogurt", 200, 12, 5, 12),
            ("Almonds", 20, 4, 10, 2),
        ],
        "Dinner": [
            ("Salmon", 150, 34, 13, 0) if not veg else ("Paneer/Tempeh", 150, 30, 15, 6),
            ("Quinoa", 150, 6, 2, 27),
            ("Salad", 200, 4, 10, 8),
        ],
    }
    if high_pro:
        meals.setdefault("Post-workout", []).append(("Whey/Plant protein", 1, 25, 2, 3))

    def fmt(name, qty, p, f, c):
        return f"{name} — {qty} g/serv • P {p}g • F {f}g • C {c}g"

    return {meal: [fmt(*item) for item in items] for meal, items in meals.items()}
