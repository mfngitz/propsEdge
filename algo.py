import json
import math
from itertools import combinations

import numpy as np
from scipy.stats import norm, poisson


with open("projections.json") as f:
    projections = json.load(f)

with open("prop_trends.json") as f:
    trends = json.load(f)


proj_map = {p["id"]: p for p in projections}
trend_map = {p["id"]: p for p in trends}


def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def remove_vig(over_odds, under_odds):
    p_over = american_to_prob(over_odds)
    p_under = american_to_prob(under_odds)
    total = p_over + p_under
    return p_over / total, p_under / total


def expected_value(prob, odds):
    if odds > 0:
        return prob * (odds / 100) - (1 - prob)
    return prob - (1 - prob) * (100 / -odds)


def kelly_fraction(prob, odds):
    if odds > 0:
        b = odds / 100
    else:
        b = 100 / -odds

    f = (prob * (b + 1) - 1) / b
    return max(f, 0)


def dynamic_std(mean, l10_rate, l20_rate):
    dispersion = abs(l10_rate - l20_rate) / 100
    base = max(1.5, mean * 0.18)
    return base * (1 + dispersion)


def poisson_over_prob(mean, line):
    k = math.floor(line)
    return 1 - poisson.cdf(k, mean)


def normal_over_prob(mean, line, std):
    z = (mean - line) / std
    return norm.cdf(z)


def choose_model(prop_name, mean, line, std):
    low_count = ["points", "assists", "rebounds"]

    if prop_name in low_count and mean < 25:
        return poisson_over_prob(mean, line)

    return normal_over_prob(mean, line, std)


def trend_probability(rate, l5, l10, l20, opp_rank):
    base = 0.4 * rate + 0.25 * l5 + 0.2 * l10 + 0.15 * l20

    matchup_adj = 1 + ((15 - opp_rank) / 40)
    momentum_adj = 1 + ((l5 - l10) + (l10 - l20)) * 0.01

    return base * matchup_adj * momentum_adj


def blend_probability(model_p, trend_p, market_p):
    blended = 0.5 * model_p + 0.3 * trend_p + 0.2 * market_p

    edge = blended - market_p
    calibrated = 1 / (1 + math.exp(-edge * 10))

    return 0.5 * calibrated + 0.5 * blended


edges = []

for pid, proj in proj_map.items():
    if pid not in trend_map:
        continue

    player_trend = trend_map[pid]
    player_proj = proj["projections"]

    for prop_name, prop_data in player_trend.items():
        if prop_name in ["id", "name", "team", "gameId", "position"]:
            continue

        if prop_name not in player_proj:
            continue

        line = prop_data.get("line")
        over_odds = prop_data.get("over")
        under_odds = prop_data.get("under")

        if line is None or over_odds is None or under_odds is None:
            continue

        mean = player_proj[prop_name]

        market_over, _ = remove_vig(over_odds, under_odds)

        rate = prop_data.get("rate", 50) / 100
        l5 = prop_data.get("l5Rate", 50) / 100
        l10 = prop_data.get("l10Rate", 50) / 100
        l20 = prop_data.get("l20Rate", 50) / 100
        opp = prop_data.get("oppDef", 15)

        std = dynamic_std(
            mean,
            prop_data.get("l10Rate", 50),
            prop_data.get("l20Rate", 50),
        )

        model_prob = choose_model(prop_name, mean, line, std)
        trend_prob = trend_probability(rate, l5, l10, l20, opp)

        final_prob = blend_probability(model_prob, trend_prob, market_over)

        ev = expected_value(final_prob, over_odds)
        kelly = kelly_fraction(final_prob, over_odds)

        z_score = (mean - line) / std
        stability = 1 - abs(z_score) / 8
        liquidity_adj = 1 - abs(final_prob - market_over)

        score = ev * stability * liquidity_adj

        edges.append({
            "player": proj["name"],
            "team": proj["team"],
            "prop": prop_name,
            "line": line,
            "projection": round(mean, 2),
            "probability": round(final_prob, 3),
            "ev": round(ev, 3),
            "kelly": round(kelly, 3),
            "score": round(score, 3),
        })


edges.sort(key=lambda x: x["score"], reverse=True)

print("\nTop edges\n")
for e in edges[:25]:
    print(e)


def correlation_adjustment(combo):
    teams = [leg["team"] for leg in combo]
    duplicates = len(teams) - len(set(teams))
    return 1 - duplicates * 0.1


def parlay_probability(combo):
    p = 1
    for leg in combo:
        p *= leg["probability"]

    return p * correlation_adjustment(combo)


top_candidates = edges[:15]
parlays = []

for combo in combinations(top_candidates, 3):
    p = parlay_probability(combo)
    parlays.append((combo, p))

parlays.sort(key=lambda x: x[1], reverse=True)

print("\nTop 3-leg combinations\n")
for combo, p in parlays[:5]:
    print("\nProbability:", round(p, 3))
    for leg in combo:
        print(leg["player"], leg["prop"], "over", leg["line"])
