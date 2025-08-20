import cv2
import numpy as np
from typing import List, Tuple


def draw_header(frame, frame_w: int, title: str) -> None:
	"""Draw a consistent top header with app brand and exercise title."""
	header_height = 60
	header_bg_color = (40, 40, 40)
	header_text_color = (255, 255, 255)
	# Background
	cv2.rectangle(frame, (0, 0), (frame_w, header_height), header_bg_color, -1)
	cv2.rectangle(frame, (0, 0), (frame_w, header_height), (100, 100, 100), 2)
	# Brand
	cv2.putText(frame, "flexIntel", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 212, 170), 2, cv2.LINE_AA)
	# Exercise title
	title_x = frame_w - 20 - cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
	cv2.putText(frame, title, (title_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, header_text_color, 2, cv2.LINE_AA)


def draw_fps(frame, fps: float) -> None:
	"""Disabled per request: keep API but draw nothing."""
	return


def draw_feedback_panel(
	frame,
	items: List[Tuple[str, bool]],
	frame_w: int,
	frame_h: int,
	title: str = "FEEDBACK",
) -> None:
	"""Draw a unified feedback panel similar to the provided design.

	Each item is a tuple of (label, is_positive).
	Up to two items are shown side-by-side separated by a thin divider line.
	"""
	if not items:
		return

	panel_margin = 20
	panel_y = frame_h - 160
	panel_h = 120
	panel_w = int(min(frame_w * 0.6, 520))
	panel_x = panel_margin

	# Semi-transparent background
	overlay = frame.copy()
	cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
	cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
	# Border
	cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (160, 160, 160), 1)

	# Title with underline
	cv2.putText(frame, title, (panel_x + 18, panel_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (235, 235, 235), 2, cv2.LINE_AA)
	cv2.line(frame, (panel_x + 18, panel_y + 36), (panel_x + panel_w - 18, panel_y + 36), (200, 200, 200), 1)

	# Layout for two columns
	col_w = (panel_w - 40) // 2
	for i in range(min(2, len(items))):
		label, good = items[i]
		x0 = panel_x + 20 + i * col_w
		y0 = panel_y + 60
		# simple icon: circle with check/cross
		icon_center = (x0 + 20, y0 + 18)
		icon_color = (40, 200, 60) if good else (40, 40, 200)
		cv2.circle(frame, icon_center, 12, (230, 230, 230), 2)
		if good:
			# check mark
			cv2.line(frame, (icon_center[0] - 6, icon_center[1] + 2), (icon_center[0] - 1, icon_center[1] + 7), icon_color, 2)
			cv2.line(frame, (icon_center[0] - 1, icon_center[1] + 7), (icon_center[0] + 7, icon_center[1] - 6), icon_color, 2)
		else:
			# cross
			cv2.line(frame, (icon_center[0] - 7, icon_center[1] - 7), (icon_center[0] + 7, icon_center[1] + 7), (220, 60, 60), 2)
			cv2.line(frame, (icon_center[0] + 7, icon_center[1] - 7), (icon_center[0] - 7, icon_center[1] + 7), (220, 60, 60), 2)
		# label
		cv2.putText(frame, label, (x0 + 45, y0 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (235, 235, 235), 2, cv2.LINE_AA)

	# Divider line between columns (if 2 items)
	if len(items) >= 2:
		mid_x = panel_x + 20 + col_w
		cv2.line(frame, (mid_x, panel_y + 46), (mid_x, panel_y + panel_h - 12), (200, 200, 200), 1)


