"""Rebuild expensive orders or contiguous groups of loads around a fixed plan."""
from bisect import bisect_right
from collections import Counter, defaultdict
from itertools import count
from random import Random
from time import monotonic

from aiconcrete import TRUCK_DAY_START, WASH_MIN, kpis, leg, m2t, simulate, transport_cost
from dispatch import dispatch
from solver import shift_options


def replay(day, rows, changes, production):
    sim, errors = simulate(day, rows, production, changes)
    closed = sum(r['load_start'] + day.plants[r['plant_id']]['mix_load_min'] > day.plants[r['plant_id']]['close_m']
                 for r in sim.done.values())
    res = kpis(day, sim, errors)
    res['after_close'] = closed
    return sim, res


def ranking(res):
    return (-res['errors'], -res.get('after_close', 0), res['sla_checks_passed'],
            res['sla_score'], res['contribution_baht'], -res['km_total'])


def parameters(rng):
    return dict(reuse_weight=rng.choice([0.45, 1, 2.5, 5, 12]),
                empty_weight=rng.choice([0, 0.3, 1]), first_slack=rng.choice([0.8, 2, 4]),
                tightness=rng.choice([0.15, 0.65, 1]), shift_penalty=rng.choice([0.15, 0.6, 1.5, 4]),
                grade_weight=rng.choice([0, 0.5, 1.5]), lead_weight=rng.choice([0, 0.5, 1, 1.5]),
                pace=rng.choice([0, 0.3, 0.65, 0.85]), wait_weight=rng.choice([0.5, 1, 2]))


def rebuild(day, rows, changes, production, deadline, offer, seed=15, partial=True):
    rng = Random(seed)
    flex = shift_options(day)
    sim, current = replay(day, rows, changes, production)
    if current['errors']:
        return
    counts = Counter(r['stream_id'] for r in rows)
    unit = {}
    for sid, s in day.streams.items():
        unit[sid] = {
            pid: (day.plant_product[pid, s['product_id']] * s['load_m3']
                  + transport_cost(day, leg(day, pid, s['site_id'], 480)[0], s['load_m3']))
            for pid, p in day.plants.items() if (pid, s['product_id']) in day.plant_product
            and not any(st['plant_id'] == pid and st['lines'] == 0 and st['start'] <= p['open_m']
                        and st['end'] >= p['close_m'] for st in day.status)
        }
    cheapest = {sid: min(cost.values(), default=0) for sid, cost in unit.items()}

    def priorities():
        margins = dict.fromkeys(day.orders, 10)
        costs = []
        for row in rows:
            sid, seq, pid = row['stream_id'], row['seq'], row['plant_id']
            trip = sim.done[sid, seq]
            cost = max(0, unit[sid].get(pid, 20000) - cheapest[sid]) + trip['km_reposition'] * 15 + 20
            p = day.plants[pid]
            if trip['rejected'] or trip['load_start'] + p['mix_load_min'] > p['close_m']:
                cost += 20000
            margins[day.streams[sid]['order_id']] += cost
            costs.append(cost)
        for sid, s in day.streams.items():
            margins[s['order_id']] += 3000 * max(0, s['loads'] - counts[sid])
        return sorted(margins, key=lambda oid: -margins[oid]), margins, costs

    orders, margins, costs = priorities()
    for trial in count():
        if not orders or monotonic() >= deadline:
            return
        whole = not partial or trial < len(orders) or trial % 3 != 0 or not rows
        if whole:
            group_sizes = [1, 1, 1, 2, 3] if partial else [1, 1, 1, 1, 2, 3]
            chosen = ([orders[trial]] if trial < len(orders) else rng.choices(
                orders, weights=[margins[oid] ** 0.75 for oid in orders], k=rng.choice(group_sizes)))
            bounds = {sid: (1, day.streams[sid]['loads']) for oid in chosen for sid in day.orders[oid]}
        else:
            row = rng.choices(rows, weights=costs, k=1)[0]
            sid, seq = row['stream_id'], row['seq']
            bounds = {sid: (max(1, seq - rng.randrange(3)), min(counts[sid], seq + rng.randrange(5)))}
            chosen = []
        new_changes = list(changes)
        if trial % 3 != 1:
            for oid in sorted(set(chosen) & flex.keys()):
                shift = rng.choice([None, *flex[oid]])
                new_changes = [r for r in new_changes if r['order_id'] != oid]
                if shift is not None:
                    new_changes.append(dict(order_id=oid, new_first_arrival=m2t(shift), reason='กระจายการผลิต'))
        fixed = []
        for row in rows:
            sid, seq = row['stream_id'], row['seq']
            lo, hi = bounds.get(sid, (1, 0))
            if lo <= seq <= hi:
                continue
            actual = sim.done[sid, seq]
            plan_row = row
            if trial % 3 and actual['load_start'] < 1440:
                plan_row = dict(row, load_start=int(actual['load_start']))
            fixed.append((plan_row, actual))
        built = dispatch(day, new_changes, production, only=bounds, fixed=fixed, **parameters(rng))
        if len(built) < len(rows):
            continue
        new_sim, result = replay(day, built, new_changes, production)
        if ranking(result) > ranking(current):
            sim, rows, changes, current = new_sim, built, new_changes, result
            counts = Counter(row['stream_id'] for row in rows)
            if partial:
                orders, margins, costs = priorities()
            else:
                # The whole-order sweep keeps its original order as a stable search baseline.
                _, margins, costs = priorities()
            offer(rows, changes, production, result)


def eject(day, rows, changes, production, deadline, offer, seed):
    """Try cheaper plants first, including the slot-ejection move from 1670903911."""
    rng = Random(seed)
    sim, current = replay(day, rows, changes, production)
    if current['errors'] or not rows:
        return
    streams = [day.streams[r['stream_id']] for r in rows]

    def price(i, pid):
        s = streams[i]
        if (pid, s['product_id']) not in day.plant_product:
            return float('inf')
        km = leg(day, pid, s['site_id'], 480)[0]
        return day.plant_product[pid, s['product_id']] * s['load_m3'] + transport_cost(day, km, s['load_m3'])

    prices = [{pid: price(i, pid) for pid in day.plants} for i in range(len(rows))]

    def move(i, pid, tid, offset):
        row = rows[i]
        trip = sim.done[row['stream_id'], row['seq']]
        plant = day.plants[pid]
        site = streams[i]['site_id']
        depart = trip['depart']
        desired_arrival = trip['arrive'] - trip['wait_ban'] + offset
        for _ in range(3):
            depart = desired_arrival - day.sites[site]['access_min'] - leg(day, pid, site, depart)[1]
        lo, hi = plant['open_m'] + plant['prep_min'], plant['close_m'] - plant['mix_load_min']
        start = max(lo, min(hi, round(depart - plant['mix_load_min'] - plant['qc_min'])))
        return dict(row, plant_id=pid, truck_id=tid, load_start=start)

    def allowed(i, pid):
        return [tid for tid, t in day.trucks.items() if t['size'] == streams[i]['truck_size']
                and (t['allowed'] is None or pid in t['allowed'])]

    while monotonic() < deadline:
        options = sorted((prices[i][row['plant_id']] - cost + rng.uniform(-70, 70), i, pid)
                         for i, row in enumerate(rows) for pid, cost in prices[i].items()
                         if cost < prices[i][row['plant_id']] - 1)
        if not options:
            return
        for _, i, pid in reversed(options):
            if monotonic() >= deadline:
                return
            old = rows[i]['plant_id']
            if prices[i][pid] >= prices[i][old]:
                continue
            trucks = allowed(i, pid)
            rng.shuffle(trucks)
            tid = rows[i]['truck_id']
            if tid in trucks:
                trucks.remove(tid)
                trucks.insert(0, tid)
            for tid in trucks[:3]:
                base = move(i, pid, tid, 0)
                proposals = [{i: move(i, pid, tid, d)} for d in (0, -4, 4, -9, 9)]
                # Free a near-term line at the cheap plant by displacing a lower-value load.
                nearby = sorted((abs(q['load_start'] - base['load_start']), j)
                                for j, q in enumerate(rows) if j != i and q['plant_id'] == pid)
                for distance, j in nearby[:2]:
                    if distance > 15:
                        break
                    saving = prices[i][old] - prices[i][pid]
                    for extra, alt in sorted((p - prices[j][pid], alt) for alt, p in prices[j].items() if alt != pid)[:2]:
                        j_tid = rows[j]['truck_id']
                        if extra < saving and j_tid in allowed(j, alt):
                            proposals.append({i: base, j: move(j, alt, j_tid, 0)})
                for proposal in proposals:
                    if monotonic() >= deadline:
                        return
                    trial = [proposal.get(j, row) for j, row in enumerate(rows)]
                    new_sim, result = replay(day, trial, changes, production)
                    if ranking(result) > ranking(current):
                        sim, rows, current = new_sim, trial, result
                        offer(rows, changes, production, current)
                        break
                else:
                    continue
                break


def tune_trucks(day, rows, changes, production, deadline, offer, seed):
    """Reassign promising loads to nearby trucks to cut morning moves and waiting."""
    rng = Random(seed)
    sim, current = replay(day, rows, changes, production)
    streams = [day.streams[r['stream_id']] for r in rows]
    sites = [s['site_id'] for s in streams]
    trucks_by_size = defaultdict(list)
    for tid, truck in day.trucks.items():
        trucks_by_size[truck['size']].append(tid)

    def routes():
        grouped = defaultdict(list)
        actual = [sim.done[r['stream_id'], r['seq']] for r in rows]
        for i, row in enumerate(rows):
            grouped[row['truck_id']].append((row['load_start'], i))
        for route in grouped.values():
            route.sort()
        repositioned = [i for i, trip in enumerate(actual) if trip['km_reposition']]
        return grouped, actual, repositioned

    grouped, actual, repositioned = routes()
    while monotonic() < deadline:
        if repositioned and rng.random() < 0.45:
            i = rng.choice(repositioned)
        else:
            i = rng.randrange(len(rows))
        row, trip = rows[i], actual[i]
        pid = row['plant_id']
        start = max(0, min(1439, round(trip['load_start']) + rng.choice((0, 0, -6, -3, -1, 1, 3, 6))))
        ready_after = trip['pour_end'] + WASH_MIN + start - trip['load_start']
        estimates = []
        for tid in trucks_by_size[streams[i]['truck_size']]:
            truck = day.trucks[tid]
            if truck['allowed'] is not None and pid not in truck['allowed']:
                continue
            schedule = grouped.get(tid, ())
            if tid == row['truck_id']:
                schedule = [pair for pair in schedule if pair[1] != i]
            pos = bisect_right(schedule, (start, i))
            if pos:
                previous = schedule[pos - 1][1]
                origin, ready = sites[previous], actual[previous]['pour_end'] + WASH_MIN
            else:
                origin, ready = truck['home_plant'], TRUCK_DAY_START
            km, travel = leg(day, origin, pid, ready)
            cost = km if pos else km * day.cost['reposition_baht_per_km']
            cost += 45 * max(0, ready + travel - start)
            if pos < len(schedule):
                following = schedule[pos][1]
                next_plant = rows[following]['plant_id']
                _, travel = leg(day, sites[i], next_plant, ready_after)
                cost += 45 * max(0, ready_after + travel - actual[following]['load_start'])
                if not pos:
                    cost -= leg(day, origin, next_plant, ready)[0] * day.cost['reposition_baht_per_km']
            estimates.append((cost, tid))
        for tid in [t for _, t in sorted(estimates)[:3]] + [row['truck_id']]:
            if monotonic() >= deadline:
                return
            replacement = dict(row, truck_id=tid, load_start=start)
            if replacement == row:
                continue
            trial = rows[:]
            trial[i] = replacement
            new_sim, result = replay(day, trial, changes, production)
            if ranking(result) > ranking(current):
                sim, rows, current = new_sim, trial, result
                grouped, actual, repositioned = routes()
                offer(rows, changes, production, current)
                break
