"""Behaviour-identical speed-ups for the official simulator (memoised leg(), cheaper Line.earliest()).
Only the solver imports this; results are bit-for-bit the same as the original functions."""
from bisect import bisect_right

import aiconcrete as _ac

_orig_leg = _ac.leg
_orig_earliest = _ac.Line.earliest


def _leg(day, a, b, depart):
    if a == b:
        return 0.0, 0.0
    try:
        cache = day._legc
    except AttributeError:
        cache = day._legc = {}
    k = (a, b, depart)
    r = cache.get(k)
    if r is None:
        r = cache[k] = _orig_leg(day, a, b, depart)
    return r


def _earliest(self, a, length):
    # Lines append reservations and blocks. Many preview calls query the same calendar.
    sizes = (len(self.busy), len(self.blocks))
    if length <= 0 or sum(sizes) >= 500:
        return _orig_earliest(self, a, length)
    calendar = getattr(self, '_calendar', None)
    if calendar is None or calendar[0] != sizes:
        starts, ends = [], []
        for start, end in sorted(self.busy + self.blocks):
            if start > end:
                return _orig_earliest(self, a, length)
            if ends and start <= ends[-1]:
                ends[-1] = max(ends[-1], end)
            else:
                starts.append(start)
                ends.append(end)
        self._calendar = calendar = (sizes, starts, ends)
    _, starts, ends = calendar
    i = bisect_right(ends, a)
    while i < len(starts) and a + length > starts[i]:
        a = ends[i]
        i += 1
    return a


def _line_start(self, pid, desired):
    plant = self.day.plants[pid]
    prep = plant['prep_min']
    first = desired - prep
    duration = prep + plant['mix_load_min']
    best = None
    for i, line in enumerate(self.lines[pid]):
        start = line.earliest(first, duration)
        if start is not None:
            load = start + prep
            if start == first:
                return (load, i) if best is None or load < best[0] else best
            if best is None or load < best[0]:
                best = (load, i)
    return best


def eligible_trucks(day):
    """Index vendors by station and truck size, in the original fleet order."""
    cached = getattr(day, '_eligible_trucks', None)
    if cached is None:
        cached = {}
        for tid, truck in day.trucks.items():
            allowed = truck['allowed'] if truck['allowed'] is not None else day.plants
            for pid in allowed:
                cached.setdefault((pid, truck['size']), []).append(tid)
        day._eligible_trucks = cached
    return cached


def _queue_violations(day, sim):
    visits = {}
    for (sid, _), trip in sim.done.items():
        visits.setdefault(day.streams[sid]['site_id'], []).append((trip['arrive'], trip['pour_start']))
    violations = 0
    for site, trips in visits.items():
        limit = day.sites[site]['max_queue']
        waiting_trips = [(arrive, pour) for arrive, pour in trips if arrive < pour]
        arrivals = sorted(arrive for arrive, _ in waiting_trips)
        pours = sorted(pour for _, pour in waiting_trips)
        for arrive, pour in trips:
            waiting = bisect_right(arrivals, arrive) - bisect_right(pours, arrive)
            violations += waiting - (arrive < pour) >= limit
    return violations


_ac.leg = _leg
_ac.Line.earliest = _earliest
_ac.Sim.line_start = _line_start
_ac.queue_violations = _queue_violations


# ---------------------------------------------------------------------------------------------
# Incremental evaluator: the plan is simulated in load_start order, so a change at index j leaves the
# state after the first j loads untouched. We keep snapshots of the *current* plan and re-simulate only the suffix.
import copy as _copy
from aiconcrete import Sim, simulate, kpis, TruckState, Line

STEP = 6


def _clone(sim):
    n = Sim.__new__(Sim)
    n.__dict__.update(sim.__dict__)
    n.notes = list(sim.notes)
    n.lines = {pid: [Line(busy=list(l.busy), blocks=l.blocks, boost=l.boost) for l in ls] for pid, ls in sim.lines.items()}
    n.trucks = {tid: TruckState(t.loc, t.ready, t.loads, t.km_loaded, t.km_empty, t.km_reposition) for tid, t in sim.trucks.items()}
    n.site_free = dict(sim.site_free)
    n.done = dict(sim.done)
    return n


class Incremental:
    def __init__(self, day, production, changes):
        self.day, self.prod, self.chg = day, production, changes
        self.base = None          # sorted plan tuples of the current plan
        self.snaps = {}           # idx -> (sim, errors, seen, last_seq)

    @staticmethod
    def _sorted(rows):
        plan = [dict(stream_id=r["stream_id"], seq=r["seq"], plant_id=r["plant_id"], truck_id=r["truck_id"],
                     load_start=r["load_start"]) for r in rows]
        plan.sort(key=lambda r: (r["load_start"], r["stream_id"], r["seq"]))
        return plan

    def _run(self, plan, sim, errors, seen, last_seq, start, snap):
        day = self.day
        for idx in range(start, len(plan)):
            if snap is not None and idx % STEP == 0:
                snap[idx] = (_clone(sim), list(errors), set(seen), dict(last_seq))
            r = plan[idx]
            key = (r["stream_id"], r["seq"])
            if r["stream_id"] not in day.streams:
                errors.append(f"ไม่มีสายเท {r['stream_id']}")
                continue
            if key in seen:
                errors.append(f"เที่ยวซ้ำ {key}")
                continue
            if r["truck_id"] in day.trucks_off:
                errors.append("truck off")
                continue
            if r["plant_id"] not in day.plants or r["truck_id"] not in day.trucks:
                errors.append("bad ids")
                continue
            if r["seq"] != last_seq.get(r["stream_id"], 0) + 1:
                errors.append("seq order")
            last_seq[r["stream_id"]] = r["seq"]
            seen.add(key)
            res = sim.preview(r["stream_id"], r["seq"], r["plant_id"], r["truck_id"], r["load_start"])
            for p in res["problems"]:
                errors.append(p)
            sim.commit(res)

    def eval(self, rows):
        plan = self._sorted(rows)
        tup = [(r["stream_id"], r["seq"], r["plant_id"], r["truck_id"], r["load_start"]) for r in plan]
        start = 0
        if self.base is not None:
            f = 0
            b = self.base
            m = min(len(b), len(tup))
            while f < m and b[f] == tup[f]:
                f += 1
            start = (f // STEP) * STEP
            while start > 0 and start not in self.snaps:
                start -= STEP
        if start and start in self.snaps:
            s0, e0, seen0, ls0 = self.snaps[start]
            sim, errors, seen, last_seq = _clone(s0), list(e0), set(seen0), dict(ls0)
        else:
            start = 0
            sim, errors, seen, last_seq = Sim(self.day, self.prod, self.chg), [], set(), {}
        self._run(plan, sim, errors, seen, last_seq, start, None)
        self._last = (plan, tup)
        return kpis(self.day, sim, errors)

    def accept(self, rows=None):
        """make the plan last passed to eval() (or `rows`) the new base; rebuild snapshots."""
        if rows is not None:
            plan = self._sorted(rows)
            tup = [(r["stream_id"], r["seq"], r["plant_id"], r["truck_id"], r["load_start"]) for r in plan]
        else:
            plan, tup = self._last
        snaps = {}
        sim = Sim(self.day, self.prod, self.chg)
        self._run(plan, sim, [], set(), {}, 0, snaps)
        self.base, self.snaps = tup, snaps
