import json
import math
import os
from datetime import datetime, UTC
from scipy.stats import norm, poisson


BASE = os.path.dirname(__file__)
JDIR = os.path.join(BASE, "jsons")


def build():

    with open(os.path.join(JDIR, "projections.json")) as f:
        proj = json.load(f)

    with open(os.path.join(JDIR, "prop_trends.json")) as f:
        trnd = json.load(f)

    with open(os.path.join(JDIR, "schedule.json")) as f:
        sch = json.load(f)

    today = datetime.now(UTC).strftime("%Y-%m-%d")

    team = set()
    games = []

    # collect only today's matchups
    for g in sch:
        if g.get("date") != today:
            continue

        h = g.get("home")
        a = g.get("away")

        if h and a:
            team.add(h)
            team.add(a)
            games.append((h, a))

    allowed = {
        "points",
        "rebounds",
        "assists",
        "pointsRebounds",
        "pointsAssists",
        "reboundsAssists",
        "pointsReboundsAssists",
    }

    pmap = {p["id"]: p for p in proj}
    tmap = {p["id"]: p for p in trnd}

    def num(x, d=0):
        return x if isinstance(x, (int, float)) else d

    def pct(x):
        return num(x, 50) / 100

    # simple bayesian smoothing so small samples don't swing too hard
    def bay(p, n=20, m=0.5, w=15):
        s = p * n
        a = s + m * w
        b = (n - s) + (1 - m) * w
        return a / (a + b)

    # blend poisson + normal depending on dispersion
    def mix(mu, ln, sd):
        p1 = 1 - poisson.cdf(math.floor(ln), mu)
        sd = max(sd, 0.01)
        z = (mu - ln) / sd
        p2 = norm.cdf(z)
        d = sd / max(mu, 1)
        w = min(max(d, 0), 1)
        return w * p2 + (1 - w) * p1

    def adj(p, rk):
        s = (rk - 15) / 15
        return max(min(p * (1 + s * 0.12), 0.99), 0.01)

    def logi(x):
        return 1 / (1 + math.exp(-x))

    def ent(p):
        if p <= 0 or p >= 1:
            return 0
        return -p * math.log(p) - (1 - p) * math.log(1 - p)

    def am2p(o):
        o = num(o)
        if o > 0:
            return 100 / (o + 100)
        return -o / (-o + 100)

    def nov(o1, o2):
        p1 = am2p(o1)
        p2 = am2p(o2)
        t = p1 + p2
        return p1 / t if t else 0.5

    def evl(p, o):
        o = num(o)
        if o > 0:
            return p * (o / 100) - (1 - p)
        return p - (1 - p) * (100 / -o)

    def kel(p, o):
        o = num(o)
        b = o / 100 if o > 0 else 100 / -o
        f = (p * (b + 1) - 1) / b if b else 0
        return max(f, 0)

    def stdv(mu, l1, l2):
        v = abs(l1 - l2) / 100
        b = max(1.5, mu * 0.16)
        return b * (1 + v)

    best = {}

    for pid, pr in pmap.items():

        if pid not in tmap:
            continue

        tm = pr.get("team")
        if tm not in team:
            continue

        nm = pr.get("name")
        pj = pr.get("projections", {})
        tr = tmap[pid]

        top = None

        for prop, dt in tr.items():

            if prop not in allowed:
                continue
            if prop not in pj:
                continue
            if not isinstance(dt, dict):
                continue

            ln = num(dt.get("line"))
            ov = dt.get("over")
            un = dt.get("under")

            if ln is None or ov is None or un is None:
                continue

            mu = num(pj.get(prop))
            mp = nov(ov, un)

            rt = bay(pct(dt.get("rate")), 20)
            l5 = bay(pct(dt.get("l5Rate")), 5)
            l10 = bay(pct(dt.get("l10Rate")), 10)
            l20 = bay(pct(dt.get("l20Rate")), 20)

            sd = stdv(mu, num(dt.get("l10Rate"), 50), num(dt.get("l20Rate"), 50))

            mp1 = mix(mu, ln, sd)
            mp1 = adj(mp1, num(dt.get("oppDef"), 15))

            el = (mu - ln) / ln if ln else 0
            trp = (0.45 * rt + 0.25 * l5 + 0.2 * l10 + 0.1 * l20)
            trp *= (1 + el)
            trp = adj(trp, num(dt.get("oppDef"), 15))

            dis = abs(mp1 - mp)
            mw = 0.35 + dis * 0.6
            tw = 0.35
            bw = 1 - mw - tw

            # weighted blend of model, trends, and market
            bl = mw * mp1 + tw * trp + bw * mp
            bl = max(min(bl, 0.99), 0.01)

            cf = 1 - abs(mp1 - trp)
            bl = bl * cf + mp * (1 - cf)

            cal = logi((bl - 0.5) * 6)

            mmt = (l5 - l20)
            cal *= (1 + mmt * 0.15)
            cal = max(min(cal, 0.99), 0.01)

            edg = mp1 - mp
            cal *= (1 + edg * 0.4)
            cal = max(min(cal, 0.99), 0.01)

            # evaluate both sides and take whichever has better EV
            p_over = cal
            p_under = 1 - cal

            e_over = evl(p_over, ov)
            e_under = evl(p_under, un)

            if e_over >= e_under:
                side = "over"
                final_p = p_over
                final_ev = e_over
                final_k = kel(p_over, ov)
            else:
                side = "under"
                final_p = p_under
                final_ev = e_under
                final_k = kel(p_under, un)

            var = 1 - abs(l10 - l20) * 0.4
            rev = 1 - abs(rt - l10) * 0.5
            con = 1 - (abs(rt - l10) + abs(l10 - l20)) * 0.3

            sc = final_ev * cf * var * rev * con
            sc *= (1 + abs(mu - ln) / ln if ln else 1)
            sc *= 1 - (ent(final_p) / 0.69)

            row = {
                "player": nm,
                "team": tm,
                "prop": prop,
                "side": side,
                "line": ln,
                "projection": round(mu, 2),
                "prob": round(final_p, 3),
                "ev": round(final_ev, 3),
                "kelly": round(final_k, 3),
                "score": sc,
            }

            if top is None or row["score"] > top["score"]:
                top = row

        if top:
            best.setdefault(tm, []).append(top)

    return games, allowed, best
