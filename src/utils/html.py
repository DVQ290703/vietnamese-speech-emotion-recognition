from .misc import history, MAX_HISTORY

EMOTION_COLORS = {
    "Enjoyment 😊": "#4CAF50",
    "Sadness 😢":   "#2196F3",
    "Anger 😠":     "#F44336",
    "Surprise 😲":  "#FF9800",
    "Other 😐":     "#9E9E9E",
}
EMOTION_BG = {
    "Enjoyment 😊": "#f0faf0",
    "Sadness 😢":   "#f0f6ff",
    "Anger 😠":     "#fff5f5",
    "Surprise 😲":  "#fff8f0",
    "Other 😐":     "#f5f5f5",
}


def build_result_html(label_scores: dict, top_label: str) -> str:
    bars = ""
    for label, score in label_scores.items():
        color  = EMOTION_COLORS.get(label, "#9E9E9E")
        pct    = score * 100
        weight = "font-weight:700;" if label == top_label else ""
        bars += f"""
        <div style="margin:6px 0;">
          <div style="display:flex;justify-content:space-between;{weight}">
            <span>{label}</span><span>{pct:.1f}%</span>
          </div>
          <div style="background:#e0e0e0;border-radius:4px;height:12px;margin-top:3px;">
            <div style="background:{color};width:{pct:.1f}%;height:12px;border-radius:4px;"></div>
          </div>
        </div>"""
    top_color = EMOTION_COLORS.get(top_label, "#9E9E9E")
    emoji = top_label.split()[-1] if top_label else "🤔"
    name  = top_label.split(" ")[0] if top_label else "?"
    return f"""
    <div style="font-family:sans-serif;padding:16px;border-radius:10px;border:1px solid #e0e0e0;background:#fafafa;">
      <div style="text-align:center;margin-bottom:16px;">
        <div style="font-size:2.5rem;">{emoji}</div>
        <div style="font-size:1.2rem;font-weight:700;color:{top_color};margin-top:4px;">{name}</div>
      </div>
      {bars}
    </div>"""


def build_error_html(msg: str) -> str:
    return (f"<p style='color:#e53935;font-family:sans-serif;padding:12px;"
            f"border:1px solid #ffcdd2;border-radius:8px;background:#fff5f5;'>❌ {msg}</p>")


def build_history_html() -> str:
    if not history:
        return "<p style='color:#888;font-family:sans-serif;text-align:center;padding:32px;'>Chưa có lịch sử phân tích.</p>"
    rows = ""
    for i, entry in enumerate(history):
        color   = EMOTION_COLORS.get(entry["emotion"], "#9E9E9E")
        bg      = EMOTION_BG.get(entry["emotion"], "#f5f5f5")
        icon    = "📝" if entry["source"] == "text" else "🎤"
        preview = entry["text"][:77] + "…" if len(entry["text"]) > 80 else entry["text"]
        rows += f"""
        <tr style="background:{'white' if i % 2 == 0 else '#f9f9f9'};">
          <td style="padding:10px 12px;color:#666;white-space:nowrap;">{entry["time"]}</td>
          <td style="padding:10px 12px;font-size:1.1rem;text-align:center;">{icon}</td>
          <td style="padding:10px 12px;max-width:400px;word-break:break-word;">{preview}</td>
          <td style="padding:10px 12px;">
            <span style="background:{bg};color:{color};border:1px solid {color};border-radius:16px;
                         padding:3px 10px;font-weight:600;white-space:nowrap;font-size:0.9rem;">
              {entry["emotion"]}
            </span>
          </td>
          <td style="padding:10px 12px;color:#555;">{entry["confidence"]}</td>
        </tr>"""
    return f"""
    <div style="font-family:sans-serif;overflow-x:auto;">
      <table style="width:100%;border-collapse:collapse;font-size:0.95rem;">
        <thead>
          <tr style="background:#f0f0f0;text-align:left;">
            <th style="padding:10px 12px;">Thời gian</th>
            <th style="padding:10px 12px;text-align:center;">Nguồn</th>
            <th style="padding:10px 12px;">Văn bản</th>
            <th style="padding:10px 12px;">Cảm xúc</th>
            <th style="padding:10px 12px;">Độ tin cậy</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
      <p style="color:#999;font-size:0.82rem;margin-top:8px;text-align:right;">
        {len(history)} bản ghi (tối đa {MAX_HISTORY})
      </p>
    </div>"""
