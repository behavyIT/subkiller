"""
Генерация синтетического датасета для проекта «Убийца подписок».

Данные сгенерированы, но параметры откалиброваны по реальным опросам:
- C+R Research (2022): люди недооценивают расходы на подписки примерно в 2.5 раза,
  42% забывали о подписке, которой уже не пользуются, 74% считают, что регулярные
  списания легко забыть.
- Self Financial (2025–2026): около 55–60% имеют хотя бы одну неиспользуемую подписку,
  70% хотя бы раз забывали отменить пробный период.
- CNET/YouGov (2025): в среднем ~17 $ в месяц уходит на неиспользуемые подписки.

Результат: два CSV-файла
  respondents.csv    — опрос 600 человек (студенты и молодые работающие)
  subscriptions.csv  — все подписки этих людей построчно
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 600

# Каталог сервисов: название, категория, цена в месяц (€)
CATALOG = [
    ("Netflix", "Видео", 12.99), ("YouTube Premium", "Видео", 11.99),
    ("Disney+", "Видео", 9.99), ("HBO Max", "Видео", 9.99), ("Prime Video", "Видео", 5.99),
    ("Spotify", "Музыка", 10.99), ("Apple Music", "Музыка", 10.99), ("Deezer", "Музыка", 10.99),
    ("ChatGPT Plus", "AI-сервисы", 20.00), ("Claude Pro", "AI-сервисы", 18.00),
    ("Midjourney", "AI-сервисы", 10.00), ("GitHub Copilot", "AI-сервисы", 10.00),
    ("iCloud+", "Облако", 2.99), ("Google One", "Облако", 1.99), ("Dropbox", "Облако", 11.99),
    ("Xbox Game Pass", "Игры", 14.99), ("PlayStation Plus", "Игры", 8.99), ("Nintendo Online", "Игры", 3.99),
    ("Duolingo Super", "Обучение", 12.99), ("Coursera Plus", "Обучение", 49.00),
    ("Telegram Premium", "Соцсети", 4.99), ("Discord Nitro", "Соцсети", 9.99),
    ("Adobe CC", "Софт", 24.99), ("Microsoft 365", "Софт", 7.00), ("Notion Plus", "Софт", 9.50),
    ("Спортзал", "Спорт", 35.00), ("Strava", "Спорт", 7.99),
    ("Bolt Food+", "Доставка", 5.99), ("Wolt+", "Доставка", 7.99),
]
names = np.array([c[0] for c in CATALOG])
cats = np.array([c[1] for c in CATALOG])
prices = np.array([c[2] for c in CATALOG])

# Популярность сервисов (веса выбора)
popularity = np.array([9, 6, 4, 3, 4, 10, 3, 1, 7, 3, 1, 2, 8, 5, 1,
                       3, 3, 1, 4, 1, 5, 3, 2, 4, 2, 4, 1, 2, 2], dtype=float)
popularity /= popularity.sum()

# Вероятность, что подпиской не пользовались 30 дней, зависит от категории:
# обучение и спортзал бросают чаще всего, музыку и облако используют постоянно
UNUSED_P = {"Обучение": 0.42, "Спорт": 0.38, "Видео": 0.22, "AI-сервисы": 0.18,
            "Игры": 0.20, "Софт": 0.15, "Доставка": 0.25, "Соцсети": 0.12,
            "Музыка": 0.06, "Облако": 0.04}

respondents, subs_rows = [], []

for rid in range(1, N + 1):
    group = rng.choice(["Студент", "Работающий"], p=[0.65, 0.35])
    age = int(rng.integers(18, 24)) if group == "Студент" else int(rng.integers(22, 36))

    # Количество подписок: у работающих в среднем больше
    lam = 5.0 if group == "Студент" else 7.0
    n_subs = int(np.clip(rng.poisson(lam), 1, 15))

    chosen = rng.choice(len(CATALOG), size=n_subs, replace=False, p=popularity)

    actual = 0.0
    unused_count, unused_cost = 0, 0.0
    for idx in chosen:
        # Студенческие скидки на часть сервисов
        price = prices[idx] * (0.5 if group == "Студент" and rng.random() < 0.3 else 1.0)
        # Вероятность, что подпиской не пользовались последние 30 дней
        used = rng.random() > UNUSED_P[cats[idx]]
        day = int(rng.integers(1, 29))
        subs_rows.append({
            "respondent_id": rid, "service": names[idx], "category": cats[idx],
            "price_month_eur": round(price, 2), "billing_day": day,
            "used_last_30d": used,
        })
        actual += price
        if not used:
            unused_count += 1
            unused_cost += price

    # Оценка «навскидку»: люди недооценивают в среднем в ~2.5 раза
    underestimate_factor = rng.lognormal(mean=np.log(2.3), sigma=0.35)
    estimated = actual / underestimate_factor

    forgot_trial = rng.random() < 0.70
    missed_date = rng.random() < (0.45 + 0.04 * n_subs)  # чем больше подписок, тем чаще путаются
    # Интерес к боту (1–5) выше у тех, у кого больше подписок и есть «мёртвые» подписки
    interest = 2.2 + 0.18 * n_subs + 0.6 * (unused_count > 0) + 0.4 * forgot_trial + rng.normal(0, 0.8)
    interest = int(np.clip(round(interest), 1, 5))

    respondents.append({
        "respondent_id": rid, "group": group, "age": age,
        "n_subscriptions": n_subs,
        "estimated_month_eur": round(estimated, 2),
        "actual_month_eur": round(actual, 2),
        "unused_subscriptions": unused_count,
        "unused_cost_month_eur": round(unused_cost, 2),
        "forgot_to_cancel_trial": forgot_trial,
        "forgot_payment_date": bool(missed_date),
        "bot_interest_1to5": interest,
    })

pd.DataFrame(respondents).to_csv("respondents.csv", index=False)
pd.DataFrame(subs_rows).to_csv("subscriptions.csv", index=False)
print(f"Сгенерировано: {N} респондентов, {len(subs_rows)} подписок")
