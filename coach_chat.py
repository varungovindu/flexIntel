import os
from typing import Dict, List, Optional

import streamlit as st

from database import FitnessDatabase
from user_management import UserManager


def _get_user_manager() -> UserManager:
	user_manager: Optional[UserManager] = st.session_state.get("user_manager")
	if not user_manager:
		user_manager = UserManager()
		st.session_state.user_manager = user_manager
	return user_manager


def _get_user_context() -> Dict:
	user_manager = _get_user_manager()
	user = user_manager.get_current_user()
	if not user:
		return {}
	progress_30 = user_manager.db.get_user_progress(user.id, days=30)
	return {
		"profile": {
			"username": user.username,
			"age": user.age,
			"weight": user.weight,
			"height": user.height,
			"fitness_level": user.fitness_level,
			"fitness_goal": user.fitness_goal,
			"goals": user.goals,
		},
		"progress": progress_30,
	}


def _call_llm(messages: List[Dict[str, str]], system_prompt: str) -> Optional[str]:
	"""Call an LLM if available. Returns None if not configured.

	Avoids raising if no secrets file exists by guarding access to st.secrets.
	"""
	api_key = os.environ.get("OPENAI_API_KEY")
	if not api_key:
		try:
			api_key = st.secrets["OPENAI_API_KEY"]  # may raise if no secrets file
		except Exception:
			api_key = None
	if not api_key:
		return None
	try:
		from openai import OpenAI
		openai_client = OpenAI(api_key=api_key)
		full_messages = [{"role": "system", "content": system_prompt}] + messages
		resp = openai_client.chat.completions.create(
			model="gpt-4o-mini",
			messages=full_messages,
			temperature=0.4,
		)
		return resp.choices[0].message.content
	except Exception as e:
		st.info(f"LLM unavailable, using on-device coach. ({e})")
		return None


def _local_coach_reply(query: str, context: Dict) -> str:
	"""Simple on-device fallback – rule-based diet/workout tips using profile + progress."""
	profile = context.get("profile", {})
	progress = context.get("progress", {})
	level = profile.get("fitness_level", "beginner")
	goal = profile.get("fitness_goal", "general_fitness")
	avg_form = progress.get("avg_form_score", 0)
	total_reps = progress.get("total_reps", 0)

	lines: List[str] = []
	q = query.lower()

	if any(k in q for k in ["diet", "meal", "protein", "calorie", "nutrition", "food"]):
		prot_g = max(60, int(profile.get("weight", 70) * 1.6))
		lines.append("Here’s a simple 1‑day diet outline you can rotate:")
		if goal in ["muscle_gain", "strength"]:
			lines += [
				f"- Breakfast: Oats + greek yogurt + berries + nuts (30–40g protein)",
				"- Lunch: Chicken or tofu bowl: rice/quinoa, mixed veggies, olive oil (40–50g protein)",
				"- Snack: Protein shake + fruit (25–35g protein)",
				"- Dinner: Salmon/eggs/legumes + sweet potato + salad (40–50g protein)",
			]
		else:
			lines += [
				"- Breakfast: Veg omelette + 1 slice whole‑grain toast",
				"- Lunch: Lean protein salad (beans/chicken) + olive oil vinaigrette",
				"- Snack: Skyr/curd + fruit",
				"- Dinner: Stir‑fried veggies + tofu/eggs + small rice portion",
			]
		lines.append(f"Target protein: ~{prot_g} g/day. Hydration: 2–3 L/day.")

	if any(k in q for k in ["workout", "plan", "exercise", "split", "routine"]):
		difficulty_hint = {
			"beginner": "3 days/wk full‑body",
			"intermediate": "4 days/wk UL‑UL split",
			"advanced": "5 days/wk PPL + accessories",
		}.get(level, "3 days/wk full‑body")
		lines.append(f"Suggested schedule: {difficulty_hint}.")
		if goal in ["strength", "muscle_gain"]:
			lines += [
				"- Day A: Squat, Press (shoulder), Row, Curls",
				"- Day B: Deadlift/RDL, Bench/Pushups, Pulldown, Lateral raise",
				"- Progress: +2 reps or +2.5 kg each week if form ≥80%",
			]
		elif goal in ["weight_loss", "endurance"]:
			lines += [
				"- Add 20–30 min cardio after workouts or on rest days",
				"- Superset to keep heart rate up",
			]
		lines.append(f"Your recent form avg: {avg_form}%. Total reps last 30d: {total_reps}.")

	if not lines:
		lines.append("Tell me if you want diet tips or a workout plan, and any constraints (veg, time, equipment).")
	return "\n".join(lines)


def render_coach_chat_page() -> None:
	st.markdown("## 🤖 Coach Chat")
	st.caption("Get diet plans and workout suggestions. Uses an LLM if configured; otherwise a smart on‑device coach.")

	if "coach_messages" not in st.session_state:
		st.session_state.coach_messages = []  # [{role, content}]

	context = _get_user_context()
	system_prompt = (
		"You are flexIntel Coach. Be concise and actionable. Use the user's profile (level, goal, age, height, weight) "
		"and recent progress to tailor advice. Prefer bullet points. Provide safe, general guidance and defer to professionals for medical concerns."
	)

	# History
	for m in st.session_state.coach_messages:
		with st.chat_message(m["role"]):
			st.write(m["content"])

	user_input = st.chat_input("Ask about meals, macros, or a workout schedule…")
	if user_input:
		st.session_state.coach_messages.append({"role": "user", "content": user_input})
		with st.chat_message("user"):
			st.write(user_input)

		# Try LLM, fallback to local
		assistant_text = _call_llm(st.session_state.coach_messages, system_prompt)
		if not assistant_text:
			assistant_text = _local_coach_reply(user_input, context)
		st.session_state.coach_messages.append({"role": "assistant", "content": assistant_text})
		with st.chat_message("assistant"):
			st.write(assistant_text)

	# Quick prompts
	with st.expander("Quick prompts"):
		cols = st.columns(3)
		prompts = [
			"Create a 3‑day beginner workout plan",
			"Give a high‑protein vegetarian day plan",
			"How do I improve squat depth safely?",
			"Suggest a 30‑min full‑body session at home",
		]
		for i, p in enumerate(prompts):
			if cols[i % 3].button(p):
				st.session_state.coach_messages.append({"role": "user", "content": p})
				assistant_text = _call_llm(st.session_state.coach_messages, system_prompt) or _local_coach_reply(p, context)
				st.session_state.coach_messages.append({"role": "assistant", "content": assistant_text})
				st.rerun()


