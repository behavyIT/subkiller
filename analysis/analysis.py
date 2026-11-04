"""
Анализ данных для обоснования проекта «Убийца подписок».
Используются numpy, pandas, matplotlib. Графики сохраняются в папку charts/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("charts", exist_ok=True)
plt.rcParams.update({
    "figure.figsize": (9, 5.5), "figure.dpi": 130, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 13,
})
BLUE, RED, GREY, GREEN = "#2a6fdb", "#e0533d", "#9aa3ad", "#2e9e6a"

resp = pd.read_csv("respondents.csv")
subs = pd.read_csv("subscriptions.csv")

# ---------- Ключевые метрики ----------
n = len(resp)
avg_subs = resp["n_subscriptions"].mean()
avg_est = resp["estimated_month_eur"].mean()
avg_act = resp["actual_month_eur"].mean()
ratio = np.median(resp["actual_month_eur"] / resp["estimated_month_eur"])
share_unused = (resp["unused_subscriptions"] > 0).mean()
avg_unused_year = resp["unused_cost_month_eur"].mean() * 12
share_trial = resp["forgot_to_cancel_trial"].mean()
share_date = resp["forgot_payment_date"].mean()
share_interest = (resp["bot_interest_1to5"] >= 4).mean()

print("=== Ключевые показатели ===")
print(f"Респондентов: {n}")
print(f"Среднее число подписок: {avg_subs:.1f}")
print(f"Оценка трат навскидку: {avg_est:.2f} €/мес, реально: {avg_act:.2f} €/мес")
print(f"Медианная недооценка: в {ratio:.1f} раза")
print(f"Есть неиспользуемая подписка: {share_unused:.0%}")
print(f"Потери на неиспользуемых подписках: {avg_unused_year:.0f} € в год на человека")
print(f"Забывали отменить пробный период: {share_trial:.0%}")
print(f"Путались в датах списаний: {share_date:.0%}")
print(f"Готовы пользоваться ботом (оценка 4–5): {share_interest:.0%}")

# ---------- 1. Сколько подписок у людей ----------
fig, ax = plt.subplots()
bins = np.arange(0.5, resp["n_subscriptions"].max() + 1.5, 1)
ax.hist(resp["n_subscriptions"], bins=bins, color=BLUE, edgecolor="white")
ax.axvline(avg_subs, color=RED, ls="--", lw=2, label=f"Среднее: {avg_subs:.1f}")
ax.set(title="Сколько платных подписок у одного человека",
       xlabel="Количество подписок", ylabel="Число респондентов")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("charts/1_subscriptions_count.png"); plt.close(fig)

# ---------- 2. Оценка навскидку vs реальные траты ----------
fig, ax = plt.subplots()
ax.scatter(resp["estimated_month_eur"], resp["actual_month_eur"], s=14, alpha=0.5, color=BLUE)
lim = resp["actual_month_eur"].max() * 1.05
ax.plot([0, lim], [0, lim], color=GREY, ls="--", label="Если бы люди знали точно")
k, b = np.polyfit(resp["estimated_month_eur"], resp["actual_month_eur"], 1)
xs = np.linspace(0, resp["estimated_month_eur"].max(), 50)
ax.plot(xs, k * xs + b, color=RED, lw=2, label=f"Реальность: тратят в ~{ratio:.1f} раза больше")
ax.set(title="Люди не знают, сколько тратят на подписки",
       xlabel="Оценка «навскидку», €/мес", ylabel="Реальные траты, €/мес",
       xlim=(0, lim), ylim=(0, lim))
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("charts/2_estimate_vs_actual.png"); plt.close(fig)

# ---------- 3. На что уходят деньги (по категориям) ----------
by_cat = (subs.groupby("category")["price_month_eur"].sum() / n * 12).sort_values()
fig, ax = plt.subplots()
ax.barh(by_cat.index, by_cat.values, color=BLUE)
for i, v in enumerate(by_cat.values):
    ax.text(v + 1, i, f"{v:.0f} €", va="center", fontsize=10)
ax.set(title="Средние годовые траты на человека по категориям",
       xlabel="€ в год")
fig.tight_layout(); fig.savefig("charts/3_spend_by_category.png"); plt.close(fig)

# ---------- 4. Масштаб проблемы ----------
labels = ["Путались в датах\nсписаний", "Забывали отменить\nпробный период",
          "Платят за\nнеиспользуемое", "Готовы пользоваться\nботом"]
vals = np.array([share_date, share_trial, share_unused, share_interest]) * 100
fig, ax = plt.subplots()
bars = ax.bar(labels, vals, color=[RED, RED, RED, GREEN])
ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=3, fontweight="bold")
ax.set(title="Проблема массовая — и решение востребовано", ylabel="% респондентов", ylim=(0, 100))
fig.tight_layout(); fig.savefig("charts/4_problem_scale.png"); plt.close(fig)

# ---------- 5. Какие подписки чаще всего «мёртвые» ----------
svc = subs.groupby("service").agg(total=("used_last_30d", "size"),
                                  unused=("used_last_30d", lambda s: (~s).sum()))
svc = svc[svc["total"] >= 20]
svc["unused_pct"] = svc["unused"] / svc["total"] * 100
top = svc.sort_values("unused_pct").tail(10)
fig, ax = plt.subplots()
ax.barh(top.index, top["unused_pct"], color=RED)
ax.set(title="Доля подписчиков, не пользовавшихся сервисом 30 дней",
       xlabel="% подписчиков сервиса")
fig.tight_layout(); fig.savefig("charts/5_unused_by_service.png"); plt.close(fig)

# ---------- 6. Сколько бот может сэкономить ----------
# Сценарий: бот помогает отменить неиспользуемые подписки и не пропустить пробные периоды
trial_loss_year = 34  # € в год на человека: забытые пробные периоды (по данным Self Financial 2026)
months = np.arange(0, 13)
monthly_unused = resp["unused_cost_month_eur"].mean()
without_bot = months * monthly_unused + months / 12 * trial_loss_year * share_trial
with_bot = without_bot * np.where(months == 0, 1, 0.2)  # предполагаем, что бот убирает ~80% потерь
fig, ax = plt.subplots()
ax.plot(months, without_bot, color=RED, lw=2.5, marker="o", label="Без бота")
ax.plot(months, with_bot, color=GREEN, lw=2.5, marker="o", label="С ботом (−80% потерь)")
ax.fill_between(months, with_bot, without_bot, color=GREEN, alpha=0.12)
saved = without_bot[-1] - with_bot[-1]
ax.annotate(f"Экономия ≈ {saved:.0f} € в год\nна одного пользователя",
            xy=(12, (without_bot[-1] + with_bot[-1]) / 2), xytext=(6.2, without_bot[-1] * 0.8),
            arrowprops=dict(arrowstyle="->", color=GREY), fontsize=11)
ax.set(title="Накопленные потери на ненужных подписках за год",
       xlabel="Месяц", ylabel="€ потеряно (накопительно)")
ax.legend(frameon=False)
fig.tight_layout(); fig.savefig("charts/6_savings.png"); plt.close(fig)

# ---------- 7. Интерес к боту vs число подписок ----------
pivot = resp.groupby("n_subscriptions")["bot_interest_1to5"].mean()
corr = np.corrcoef(resp["n_subscriptions"], resp["bot_interest_1to5"])[0, 1]
fig, ax = plt.subplots()
ax.plot(pivot.index, pivot.values, color=BLUE, lw=2.5, marker="o")
ax.set(title=f"Чем больше подписок, тем нужнее бот (r = {corr:.2f})",
       xlabel="Количество подписок", ylabel="Средний интерес к боту (1–5)", ylim=(1, 5.2))
fig.tight_layout(); fig.savefig("charts/7_interest_vs_subs.png"); plt.close(fig)

# ---------- Сводная таблица ----------
summary = resp.groupby("group").agg(
    респондентов=("respondent_id", "count"),
    подписок_в_среднем=("n_subscriptions", "mean"),
    оценка_евро_мес=("estimated_month_eur", "mean"),
    реально_евро_мес=("actual_month_eur", "mean"),
    потери_евро_год=("unused_cost_month_eur", lambda s: s.mean() * 12),
).round(1)
summary.to_csv("charts/summary_by_group.csv")
print("\n=== По группам ===")
print(summary.to_string())
print(f"\nЭкономия с ботом: ≈ {saved:.0f} € в год на пользователя")
print(f"Корреляция «число подписок — интерес к боту»: r = {corr:.2f}")
