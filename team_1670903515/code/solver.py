"""AI Concrete solver: true-capacity aware, cost-minimising chronological planner on the official simulator.

Pipeline per day
  1. fix_capacity(): plants.csv lines are unreliable -> infer real lines from plant_load_history.csv
  2. build(): chronological greedy. For every load pick (plant, truck, load_start) so the truck arrives exactly when
     the previous truck finishes pouring (zero gap), minimising material + transport + reposition + waiting cost.
  3. search over order_changes (flexible customers) and production boosts, scored by the real simulator.
"""
from __future__ import annotations

import heapq
import math
import os

import fastsim  # noqa: F401  (speed-ups, must come before the aiconcrete imports)
from aiconcrete import (Sim, kpis, leg, m2t, read_csv, simulate, t2m, transport_cost)

BIG_PEN = 1e6


# ------------------------------------------------------------------ data repair
def fix_capacity(day):
    """Real line count = peak hourly loads in history x (prep+mix) / 60 (system value in plants.csv is stale)."""
    path = os.path.join(os.path.dirname(day.folder), "common", "plant_load_history.csv")
    if not os.path.exists(path):
        return {}
    peak = {}
    for r in read_csv(path):
        peak[r["plant_id"]] = max(peak.get(r["plant_id"], 0), int(r["loads"]))
    fixed = {}
    for pid, p in day.plants.items():
        if pid in peak and peak[pid] > 0:
            est = max(1, int(round(peak[pid] * (p["prep_min"] + p["mix_load_min"]) / 60.0)))
            if est != p["lines"]:
                fixed[pid] = (p["lines"], est)
                p["lines"] = est
    return fixed


# ------------------------------------------------------------------ greedy construction
class Builder:
    def __init__(self, day, changes=None, production=None, buf=None, bias=None, wait_w=1.0, trucks_k=3, co2_w=0.0, free_on=None, free_gr=None):
        self.co2_w = co2_w
        # streams of orders that may miss on-time (first load) / GREEN (later loads): the slack of their SLA grade
        self.free_on, self.free_gr = free_on or set(), free_gr or set()
        self.day, self.changes, self.production = day, changes, production
        self.buf = buf or {}
        self.bias = bias or {}
        self.K = trucks_k
        self.wait_w = wait_w
        self.plants_for = {}
        for sid, s in day.streams.items():
            self.plants_for[sid] = [p for p in day.plants if (p, s["product_id"]) in day.plant_product]

    # desired planned load_start so that truck arrives at `target`
    def solve_planned(self, sim, sid, k, pid, tid, target, floor):
        best = None
        d = self.day
        s = d.streams[sid]
        p = d.plants[pid]
        site = d.sites[s["site_id"]]
        _, m = leg(d, pid, s["site_id"], target - 40)
        planned = target - m - site["access_min"] - p["mix_load_min"] - p["qc_min"]
        for _ in range(4):
            pl = max(floor, int(math.floor(planned)))
            r = sim.preview(sid, k, pid, tid, pl)
            err = target - r["arrive"]
            if best is None or abs(err) < abs(best[1]):
                best = (r, err)
            if abs(err) <= 0.6 or (err < 0 and r["load_start"] > pl + 0.01 and pl == floor):
                break
            planned = pl + err
        return best[0]

    def run(self):
        d = self.day
        sim = Sim(d, self.production, self.changes)
        state = {sid: dict(k=1, prev_end=None, prev_planned=0) for sid in d.streams}
        heap = []

        def push(sid):
            st = state[sid]
            s = d.streams[sid]
            if st["k"] > s["loads"]:
                return
            if st["k"] == 1:
                t = self.first_target(sim, sid)
            else:
                t = st["prev_end"] - 1 - self.buf.get(sid, 0)
            heapq.heappush(heap, (t, sid, st["k"]))

        for sid in d.streams:
            push(sid)
        rows, results = [], {}
        while heap:
            target, sid, k = heapq.heappop(heap)
            s = d.streams[sid]
            st = state[sid]
            best = None
            floor = st["prev_planned"]
            for pid in self.plants_for[sid]:
                for tid in self.pick_trucks(sim, sid, pid, target):
                    r = self.solve_planned(sim, sid, k, pid, tid, target, floor)
                    sc = self.score(sim, r, s, target, k) + self.bias.get((sid, pid), 0)
                    if best is None or sc < best[0]:
                        best = (sc, r)
            if best is None:
                continue
            sc, r = best
            planned = int(max(floor, math.floor(r["planned"])))
            sim.commit(r)
            st["prev_end"] = r["pour_end"]
            st["prev_planned"] = planned
            st["k"] += 1
            rows.append(dict(stream_id=sid, seq=k, plant_id=r["plant_id"], truck_id=r["truck_id"], load_start=planned))
            push(sid)
        return rows

    def first_target(self, sim, sid):
        s = self.day.streams[sid]
        c = self.day.customers[s["customer_id"]]
        first = sim.first_arrival[sid]
        return min(first + c["site_delay_min"], first + 12)

    def pick_trucks(self, sim, sid, pid, target):
        d = self.day
        s = d.streams[sid]
        p = d.plants[pid]
        site = d.sites[s["site_id"]]
        _, m = leg(d, pid, s["site_id"], target - 40)
        need = target - m - site["access_min"] - p["mix_load_min"] - p["qc_min"]
        feas, late = [], []
        for tid, t in d.trucks.items():
            if t["size"] != s["truck_size"] or (t["allowed"] is not None and pid not in t["allowed"]):
                continue
            ts = sim.trucks[tid]
            if ts.loc == pid:
                ready, repo = ts.ready, 0.0
            else:
                km, mins = leg(d, ts.loc, pid, ts.ready)
                ready = ts.ready + mins
                repo = km * d.cost["reposition_baht_per_km"] if ts.loc in d.plants else 0.0
            if ready <= need + 2:
                # prefer free repositioning, then the truck that has been waiting the shortest (best fit)
                feas.append((repo, -ready, tid))
            else:
                late.append((ready, tid))
        feas.sort()
        out = [t for _, _, t in feas[: self.K]]
        if len(out) < self.K:
            late.sort()
            out += [t for _, t in late[: self.K - len(out)]]
        return out

    def score(self, sim, r, s, target, k):
        d = self.day
        p = d.plants[r["plant_id"]]
        if r["problems"] or r["rejected"] or r["load_start"] + p["mix_load_min"] > p["close_m"]:
            return BIG_PEN
        c = d.cost
        mat = d.plant_product[(r["plant_id"], s["product_id"])] * s["load_m3"]
        tr = transport_cost(d, r["km_loaded"], s["load_m3"])
        rep = r["km_reposition"] * c["reposition_baht_per_km"]
        wait = r["wait_site"] * c["truck_wait_baht_per_min"] * self.wait_w
        sc = mat + tr + rep + wait
        sc += self.co2_w * (1.05 * r["km_loaded"] + 0.75 * r["km_empty_to_plant"])
        late = r["arrive"] - target + self.buf.get(s["stream_id"], 0) * (k > 1)
        slack = s["max_gap_min"] if k > 1 else 12
        if s["stream_id"] in (self.free_on if k == 1 else self.free_gr):
            return sc
        if k == 1:
            first = sim.first_arrival[s["stream_id"]]
            dev = abs(r["arrive"] - first)
            if dev > 14:
                sc += 5000 + 200 * (dev - 14)
        else:
            if late > 1.5:
                sc += 30 * (late - 1.5)
            if late > slack * 0.7:
                sc += 3000 + 300 * (late - slack * 0.7)
        # keep capacity of cheap plants free: lateness of the plant queue is already in arrive
        return sc


def evaluate_rows(day, rows, production=None, changes=None):
    plan = [dict(stream_id=r["stream_id"], seq=r["seq"], plant_id=r["plant_id"], truck_id=r["truck_id"],
                 load_start=r["load_start"]) for r in rows]
    sim, errors = simulate(day, plan, production, changes)
    return kpis(day, sim, errors)


def objective(res):
    """lexicographic-ish: passing the SLA dominates, then contribution."""
    return (res["sla_checks_passed"] * 1e7 + res["contribution_baht"]) - 1e9 * res["errors"]


# ------------------------------------------------------------------ search
import random
import time


def shift_options(day):
    """order_id -> list of allowed new first_arrival minutes (customer flexibility)."""
    out = {}
    for oid, sids in day.orders.items():
        s0 = day.streams[sids[0]]
        c = day.customers[s0["customer_id"]]
        if c["flex_type"] == "later_ok":
            out[oid] = [f for f in (s0["first_arrival"] + x for x in range(15, c["flex_max_shift_min"] + 1, 15)) if 0 <= f < 1440]
        elif c["flex_type"] == "earlier_ok":
            out[oid] = [f for f in (s0["first_arrival"] - x for x in range(15, c["flex_max_shift_min"] + 1, 15)) if 0 <= f < 1440]
    return out


def optimize(day, budget=60.0, seed=1, log=None, co2_cap=None, strict=False, patience=None, init=None):
    rnd = random.Random(seed)
    shifts = shift_options(day)
    boosts = [(pid, p) for pid, p in day.plants.items() if p["boost_max_lines"] > 0 and p.get("boost_from")]
    st = dict(buf={}, bias={}, chg=auto_shifts(day), boost={}, co2w={"w": 0.0})
    if init:
        st["buf"], st["bias"] = dict(init[0]), dict(init[1])
        if len(init) > 2:
            st["chg"] = dict(init[2])

    def build(st):
        changes = [dict(order_id=o, new_first_arrival=m2t(t), reason="ขอลูกค้าเลื่อนเวลา") for o, t in st["chg"].items()]
        prod = [dict(plant_id=pid, extra_lines=1, **{"from": p["boost_from"], "to": p["boost_to"]})
                for pid, p in boosts if st["boost"].get(pid)]
        rows = Builder(day, changes, prod, st["buf"], st["bias"], co2_w=st["co2w"].get("w", 0.0)).run()
        res = evaluate_rows(day, rows, prod, changes)
        return rows, changes, prod, res

    def obj(res):
        if strict:
            nbad = sum(1 for o in res["orders"] if o["colour"] != "GREEN" or not o["on_time"])
            return -1e8 * nbad + res["contribution_baht"] - 1e9 * res["errors"]
        pen = 200.0 * max(0.0, res["co2_kg"] - co2_cap) if co2_cap else 0.0
        return objective(res) + res["sla_score"] * 1e5 - pen

    cur = build(st)
    best_o = obj(cur[3])
    best = (cur, {k: dict(v) for k, v in st.items()})
    t0 = time.monotonic()
    it = 0
    last_imp = 0
    while time.monotonic() - t0 < budget:
        it += 1
        if patience and it - last_imp > patience and best[0][3]["sla_pass"]:
            break  # converged: SLA passes and no improvement for `patience` evaluations
        cand = {k: dict(v) for k, v in best[1].items()}
        res = best[0][3]
        bad = [o for o in res["orders"] if o["colour"] != "GREEN" or not o["on_time"]]
        m = rnd.random()
        if bad and m < 0.45:
            o = rnd.choice(bad)
            for sid in day.orders[o["order_id"]]:
                cand["buf"][sid] = max(0, cand["buf"].get(sid, 0) + rnd.choice([3, 5, 8, 12]))
        elif m < 0.6:
            sid = rnd.choice(list(day.streams))
            cand["buf"][sid] = max(0, cand["buf"].get(sid, 0) + rnd.choice([-6, -3, 3]))
        elif m < 0.8:
            sid = rnd.choice(list(day.streams))
            pid = rnd.choice(list(day.plants))
            cand["bias"][(sid, pid)] = cand["bias"].get((sid, pid), 0) + rnd.choice([-400, -150, 150, 400, 1500])
        elif m < 0.88:
            sid = rnd.choice(list(day.streams))
            prod_id = day.streams[sid]["product_id"]
            opts = sorted([p for p in day.plants if (p, prod_id) in day.plant_product], key=lambda p: day.plant_product[(p, prod_id)])
            pid = rnd.choice(opts[:3])
            cand["bias"][(sid, pid)] = cand["bias"].get((sid, pid), 0) - rnd.choice([60, 120, 250])
        elif m < 0.93 and shifts:
            oid = rnd.choice(list(shifts))
            if oid in cand["chg"] and rnd.random() < 0.4:
                del cand["chg"][oid]
            else:
                cand["chg"][oid] = rnd.choice(shifts[oid])
        elif m < 0.97 or not boosts:
            cand["co2w"]["w"] = rnd.choice([0.0, 2.0, 4.0, 6.0, 9.0, 14.0, 20.0])
        elif boosts:
            pid = rnd.choice(boosts)[0]
            cand["boost"][pid] = not cand["boost"].get(pid)
        out = build(cand)
        o = obj(out[3])
        if o >= best_o:
            if o > best_o and log:
                log(f"  it{it} {time.monotonic()-t0:.0f}s sla={out[3]['sla_checks_passed']}/{out[3]['sla_checks_total']} contrib={out[3]['contribution_baht']:,}")
            if o > best_o:
                last_imp = it
            best_o, best = o, (out, cand)
    return best[0]


# ------------------------------------------------------------------ plan-level refinement (single-load mutations)
REF_LAM = 1.0      # baht per kg CO2 in the refine objective (ranking is Contribution first, CO2 second)
REF_SLACK = 150.0  # kg CO2 per day the refine may add on top of the starting plan


def refine_plan(day, rows, changes, prod, deadline, seed=1, patience=6000, lam=None, co2_slack=None):
    lam = REF_LAM if lam is None else lam
    co2_slack = REF_SLACK if co2_slack is None else co2_slack
    """Hill-climb on an existing plan with single-load mutations (change plant / truck / load_start).
    If the plan does not pass the SLA yet this is a repair: SLA first, then profit. Once it passes,
    CO2 may not exceed the passing plan's CO2 and (contribution - lam*CO2) must improve."""
    rnd = random.Random(seed)
    plan = [dict(r) for r in rows]
    inc = fastsim.Incremental(day, prod, changes)
    cur = inc.eval(plan)
    inc.accept()
    stale = 0
    passing = cur["sla_pass"] and cur["errors"] == 0
    cap = cur["co2_kg"] + co2_slack if passing else None
    key = lambda r: r["sla_checks_passed"] * 1e7 + r["sla_score"] * 1e5 + r["contribution_baht"] - lam * r["co2_kg"] - 1e9 * r["errors"]
    def bad_idx(r, pl):
        bad = {o["order_id"] for o in r["orders"] if o["colour"] != "GREEN" or not o["on_time"]}
        sim, _ = simulate(day, [dict(x) for x in pl], prod, changes)
        idx = []
        for i, p in enumerate(pl):
            d = sim.done.get((p["stream_id"], p["seq"]))
            if day.streams[p["stream_id"]]["order_id"] in bad or (d and d["plant_delay"] > 4):
                idx.append(i)
        return idx
    focus = [] if passing else bad_idx(cur, plan)
    cost_cache = {}

    def opt_cost(pid, s):
        k = (pid, s["stream_id"])
        if k not in cost_cache:
            km = leg(day, pid, s["site_id"], 480)[0]
            cost_cache[k] = day.plant_product[(pid, s["product_id"])] * s["load_m3"] + transport_cost(day, km, s["load_m3"])
        return cost_cache[k]

    n = last = 0
    while time.monotonic() < deadline and n - last <= patience:
        n += 1
        i = rnd.choice(focus) if (focus and rnd.random() < 0.7) else rnd.randrange(len(plan))
        r = plan[i]
        s = day.streams[r["stream_id"]]
        new = [dict(x) for x in plan]
        c = new[i]
        m = rnd.random()
        if m < 0.4:
            opts = [p for p in day.plants if (p, s["product_id"]) in day.plant_product and p != r["plant_id"]]
            if not opts:
                continue
            if passing and rnd.random() < 0.8:      # profit mode: prefer cheaper (material + haul) plants
                base = min(opt_cost(p, s) for p in opts)
                pid = rnd.choices(opts, weights=[math.exp(-(opt_cost(p, s) - base) / 250.0) for p in opts])[0]
            else:
                pid = rnd.choice(opts)
            t = day.trucks[c["truck_id"]]
            if t["allowed"] is not None and pid not in t["allowed"]:
                continue
            old_m = leg(day, r["plant_id"], s["site_id"], r["load_start"])[1]
            new_m = leg(day, pid, s["site_id"], r["load_start"])[1]
            c["plant_id"] = pid
            c["load_start"] = max(0, int(round(r["load_start"] + old_m - new_m)) + rnd.choice([-3, -2, 0, 2]))
        elif m < 0.7:
            cands = [tid for tid, t in day.trucks.items() if t["size"] == s["truck_size"] and tid != r["truck_id"]
                     and (t["allowed"] is None or r["plant_id"] in t["allowed"])]
            if not cands:
                continue
            c["truck_id"] = rnd.choice(cands)
        else:
            c["load_start"] = max(0, r["load_start"] + rnd.choice([-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]))
        if not 0 <= c["load_start"] < 1440:
            continue
        res = inc.eval(new)
        if res["errors"]:
            continue
        if passing and not (res["sla_pass"] and res["co2_kg"] <= cap):
            continue
        if key(res) >= key(cur):
            stale += 1
            if key(res) > key(cur):
                last = n
                stale = 50
            if stale >= 50:                      # refresh snapshots on improvements (and now and then on plateaus)
                inc.accept()
                stale = 0
            plan, cur = new, res
            if not passing:
                if cur["sla_pass"]:
                    passing, cap, focus = True, cur["co2_kg"] + co2_slack, []
                elif last == n:
                    focus = bad_idx(cur, plan)
    return plan, cur


def auto_shifts(day):
    """Deterministic start: if an order's pour window hits a no-entry window (school hours / road closure) at its site
    and the customer is flexible, move it by the smallest allowed shift that clears the window."""
    opts = shift_options(day)
    out = {}
    for oid, sids in day.orders.items():
        if oid not in opts:
            continue
        s0 = day.streams[sids[0]]
        wins = day.sites[s0["site_id"]]["no_entry"]
        if not wins:
            continue
        span = max(day.streams[s]["loads"] * day.streams[s]["interval_min"] for s in sids) * 1.15 + 12
        first = s0["first_arrival"]

        def clash(f):
            return any(f < b and f + span > a for a, b in wins)
        if not clash(first):
            continue
        ok = [f for f in opts[oid] if not clash(f)]
        if ok:
            out[oid] = min(ok, key=lambda f: abs(f - first))
    return out


def repair_build(day, changes, prod, deadline, max_iter=80):
    """Deterministic repair loop: rebuild, then for every failing order give its streams more arrival buffer and
    push loads that waited for a plant line away from that plant. Returns best (rows, res)."""
    buf, bias = {}, {}
    best = None
    for it in range(max_iter):
        rows = Builder(day, changes, prod, buf, bias).run()
        res = evaluate_rows(day, rows, prod, changes)
        key = (res["sla_checks_passed"], res["sla_score"], res["contribution_baht"])
        if best is None or key > best[0]:
            best = (key, rows, res, ({k: v for k, v in buf.items()}, {k: v for k, v in bias.items()}))
        if res["sla_pass"] or time.monotonic() > deadline:
            break
        bad = {o["order_id"] for o in res["orders"] if o["colour"] != "GREEN" or not o["on_time"]}
        sim, _ = simulate(day, [dict(r) for r in rows], prod, changes)
        for (sid, k), d in sim.done.items():
            if day.streams[sid]["order_id"] not in bad:
                continue
            buf[sid] = buf.get(sid, 0) + (2 if k == 1 else 0.3)
            if d["plant_delay"] > 5:
                bias[(sid, d["plant_id"])] = bias.get((sid, d["plant_id"]), 0) + 120
    return best[1], best[2], best[3]


def improve_shifts(day, start, prod, deadline, profit=False, bkw=None):
    """Coordinate descent over flexible orders: try every allowed shift of each failing order with a fresh greedy
    build and keep the best (SLA first, then contribution). Deterministic and cheap (~0.1 s per trial)."""
    opts = shift_options(day)
    chg = dict(start)

    def run(c):
        changes = [dict(order_id=o, new_first_arrival=m2t(t), reason="ขอลูกค้าเลื่อนเวลา") for o, t in c.items()]
        rows = Builder(day, changes, prod, **(bkw or {})).run()
        res = evaluate_rows(day, rows, prod, changes)
        return (res["sla_checks_passed"], res["sla_score"], res["contribution_baht"]), res

    cur_key, cur_res = run(chg)
    improved = True
    while improved and time.monotonic() < deadline and (profit or not cur_res["sla_pass"]):
        improved = False
        bad = [o["order_id"] for o in cur_res["orders"] if o["colour"] != "GREEN" or not o["on_time"]]
        # failing orders first, then the orders sharing their plants' rush hour (all flexible orders)
        order = [o for o in bad if o in opts] + [o for o in opts if o not in bad]
        for oid in order:
            if time.monotonic() > deadline:
                break
            best = (cur_key, chg.get(oid), cur_res)
            for f in [None] + opts[oid]:
                if time.monotonic() >= deadline:
                    break
                if f == chg.get(oid):
                    continue
                trial = dict(chg)
                if f is None:
                    trial.pop(oid, None)
                else:
                    trial[oid] = f
                k, r = run(trial)
                if k > best[0]:
                    best = (k, f, r)
            if best[0] > cur_key:
                cur_key, cur_res = best[0], best[2]
                if best[1] is None:
                    chg.pop(oid, None)
                else:
                    chg[oid] = best[1]
                improved = True
                if cur_res["sla_pass"] and not profit:
                    break
    return chg, cur_res


def _pworker(day, start, changes, prod, end, seed, q):
    p, _ = refine_plan(day, start, changes, prod, end, seed=seed)
    q.put(p)


def parallel_refine(day, rows, changes, prod, deadline, nproc=None, rounds=2, alt=None):
    """Run refine_plan in several processes (fork) with different seeds, keep the best, and repeat from it."""
    import multiprocessing as mp
    try:
        nproc = nproc or len(os.sched_getaffinity(0))
    except Exception:
        nproc = nproc or (os.cpu_count() or 1)
    nproc = max(1, min(nproc, 4))
    best_rows = [dict(r) for r in rows]
    best_res = evaluate_rows(day, best_rows, prod, changes)
    if nproc == 1:
        return refine_plan(day, best_rows, changes, prod, deadline)
    method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"   # Windows has no fork
    key = lambda r: (r["sla_checks_passed"], r["sla_score"], r["contribution_baht"] - REF_LAM * r["co2_kg"])
    try:
        ctx = mp.get_context(method)
        for rd in range(rounds):
            now = time.monotonic()
            end = now + (deadline - now) / (rounds - rd)
            q = ctx.Queue()

            starts = [alt if (alt is not None and rd == 0 and sd % 2 == 0) else best_rows for sd in range(1, nproc + 1)]
            procs = [ctx.Process(target=_pworker, args=(day, st, changes, prod, end, sd + 100 * rd, q))
                     for sd, st in zip(range(1, nproc + 1), starts)]
            for p in procs:
                p.start()
            got = []
            for _ in procs:
                try:
                    got.append(q.get(timeout=max(1.0, end - time.monotonic() + 4)))
                except Exception:
                    break
            for p in procs:
                p.join(timeout=1)
                if p.is_alive():
                    p.terminate()
            for plan in got:
                res = evaluate_rows(day, plan, prod, changes)
                if res["errors"] == 0 and res["sla_pass"] and key(res) > key(best_res):
                    best_rows, best_res = plan, res
    except Exception:
        pass
    if time.monotonic() < deadline - 2:   # parallel stage failed/ended early: spend the rest serially
        try:
            p, _ = refine_plan(day, best_rows, changes, prod, deadline)
            r = evaluate_rows(day, p, prod, changes)
            if r["errors"] == 0 and r["sla_pass"] and key(r) > key(best_res):
                best_rows, best_res = p, r
        except Exception:
            pass
    return best_rows, best_res
