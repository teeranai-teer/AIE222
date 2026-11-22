"""Dispatch with insertion into the truck and plant calendars of a working plan."""
from bisect import bisect_right
from collections import defaultdict
from heapq import heappop, heappush
from itertools import count
from math import ceil, floor

from fastsim import eligible_trucks
from aiconcrete import GRADES, Sim, TRUCK_DAY_START, WASH_MIN, leg, transport_cost


def dispatch(day, changes, production, *, first_slack=3, tightness=0.60,
             lead_weight=0.9, grade_weight=0.0, reuse_weight=1.5, empty_weight=0.2,
             shift_penalty=1.0, wait_weight=1.0, pace=0, fixed=(), only=None):
    sim = Sim(day, production, changes)
    costs = day.cost
    by_plant_size = eligible_trucks(day)
    reposition_rate = costs['reposition_baht_per_km']
    plans = {tid: [] for tid in day.trucks}
    starts = {tid: [] for tid in day.trucks}
    available = {}
    heap = []
    next_job = count()
    weights = {g: grade_weight * params[4] for g, params in GRADES.items()}

    bounds = (only if isinstance(only, dict) else
              {sid: (1, s['loads']) for sid, s in day.streams.items() if only is None or sid in only})
    for sid, s in day.streams.items():
        if sid not in bounds:
            continue
        site_id = s['site_id']
        site = day.sites[site_id]
        volume = s['load_m3']
        candidates = []
        for pid, plant in day.plants.items():
            if (pid, s['product_id']) not in day.plant_product or (pid, s['truck_size']) not in by_plant_size:
                continue
            if any(st['plant_id'] == pid and st['lines'] == 0 and st['start'] <= plant['open_m']
                   and st['end'] >= plant['close_m'] for st in day.status):
                continue
            km, mins = leg(day, pid, site_id, sim.first_arrival[sid] - 45)
            spend = day.plant_product[pid, s['product_id']] * volume + transport_cost(day, km, volume)
            lead = plant['mix_load_min'] + plant['qc_min'] + mins + site['access_min']
            candidates.append((spend, pid, lead))
        available[sid] = sorted(candidates)

    def push(sid, seq):
        s = day.streams[sid]
        if seq > bounds[sid][1] or not available[sid]:
            return
        customer = day.customers[s['customer_id']]
        if seq == 1:
            ideal = sim.first_arrival[sid] + min(customer['site_delay_min'], 15 - first_slack)
        else:
            ideal = sim.site_free[sid] + s['max_gap_min'] * pace
        lead = min(t for _, _, t in available[sid])
        priority = ideal - lead * lead_weight - weights[s['grade']]
        heappush(heap, (priority, next(next_job), sid, seq, ideal))

    rows = []
    fixed_by_seq = {}
    for row, trip in sorted(fixed, key=lambda pair: (pair[1]['load_start'], pair[0]['stream_id'], pair[0]['seq'])):
        tid, pid = row['truck_id'], row['plant_id']
        start = trip['load_start']
        if trip['line'] < 0:
            return []
        plant = day.plants[pid]
        sim.lines[pid][trip['line']].busy.append((start - plant['prep_min'], start + plant['mix_load_min']))
        starts[tid].append(start)
        plans[tid].append(trip)
        rows.append(row)
        fixed_by_seq[row['stream_id'], row['seq']] = (row, trip)
    last_time = defaultdict(int)
    for sid, (start, _) in bounds.items():
        if start > 1:
            previous, trip = fixed_by_seq[sid, start - 1]
            sim.site_free[sid] = trip['pour_end']
            last_time[sid] = previous['load_start']
        push(sid, start)
    while heap:
        _, _, sid, seq, ideal = heappop(heap)
        s = day.streams[sid]
        site_id = s['site_id']
        site = day.sites[site_id]
        customer = day.customers[s['customer_id']]
        original = sim.first_arrival[sid]
        earliest = original + customer['site_delay_min'] if seq == 1 else sim.site_free[sid]
        pour_time = s['interval_min'] * customer['pour_factor']
        following = fixed_by_seq.get((sid, seq + 1))
        gap = s['max_gap_min']
        life = day.products[s['product_id']]['concrete_life_min']
        access = site['access_min']
        best = None
        for spend, pid, lead in available[sid]:
            if best is not None and spend > best[0] + 600:
                break
            plant = day.plants[pid]
            offset = plant['mix_load_min'] + plant['qc_min']
            last_slot = plant['close_m'] - plant['mix_load_min']
            target = ideal - lead
            for _ in range(3):
                target = ideal - offset - access - leg(day, pid, site_id, target + offset)[1]
            target = floor(target)
            slots, pours, predecessors, examined = {}, {}, {}, set()

            def next_slot(desired):
                if desired not in slots:
                    slot = sim.line_start(pid, desired)
                    slots[desired] = ceil(slot[0] - 1e-8) if slot is not None else last_slot + 1
                return slots[desired]

            starts_to_try = {max(last_time[sid], target + d) for d in (-7, -2, 0, 2, 6)}
            for planned in sorted(starts_to_try):
                if planned > last_slot:
                    continue
                low = next_slot(planned)
                if low > last_slot or low in examined:
                    continue
                examined.add(low)
                for tid in by_plant_size[pid, s['truck_size']]:
                    times, route = starts[tid], plans[tid]
                    index = bisect_right(times, low)
                    predecessor = tid, index
                    if predecessor not in predecessors:
                        if index:
                            prev = route[index - 1]
                            location, ready = day.streams[prev['stream_id']]['site_id'], prev['pour_end'] + WASH_MIN
                        else:
                            location, ready = day.trucks[tid]['home_plant'], TRUCK_DAY_START
                        km, travel = leg(day, location, pid, ready)
                        first_cost = km * reposition_rate if not index else 0
                        if index == 0 and route:
                            first_cost -= leg(day, location, route[0]['plant_id'], ready)[0] * reposition_rate
                        cost_floor = spend + first_cost * reuse_weight + empty_weight * km
                        predecessors[predecessor] = (location, ready, cost_floor, ceil(ready + travel - 1e-8))
                    location, ready, cost_floor, arrival_at_plant = predecessors[predecessor]
                    if best is not None and cost_floor >= best[0]:
                        continue
                    actual = max(low, arrival_at_plant)
                    if actual != low:
                        if best is not None and actual > target + 35 + gap:
                            continue
                        actual = next_slot(actual)
                    if actual > last_slot or index < len(times) and actual >= times[index]:
                        continue
                    if following and actual > following[0]['load_start']:
                        continue
                    if actual not in pours:
                        arrive = actual + offset + leg(day, pid, site_id, actual + offset)[1] + access
                        stop = 0
                        for start, end in site['no_entry']:
                            if start <= arrive < end:
                                stop = end - arrive
                                arrive = end
                        begin = max(earliest, arrive)
                        end = begin + pour_time
                        wait_cost = (begin - arrive + stop) * costs['truck_wait_baht_per_min'] * wait_weight
                        if seq == 1:
                            violation = max(0, abs(arrive - original) - 14.7)
                            rhythm = max(0, begin - earliest - 6) * 0.3
                        else:
                            violation = max(0, begin - earliest - gap + 0.3)
                            rhythm = max(0, begin - earliest - gap * max(tightness, pace))
                        if following:
                            next_trip = following[1]
                            violation += max(0, next_trip['arrive'] - end - max(gap, next_trip['gap'] or 0))
                            violation += max(0, end - next_trip['pour_start'] - 1)
                        pours[actual] = (end, wait_cost, 500 * violation, 7 * rhythm * shift_penalty)
                    end, wait_cost, violation_cost, rhythm_cost = pours[actual]
                    if end - actual > life:
                        continue
                    if index < len(route):
                        ready_after = end + WASH_MIN
                        time_to_next = leg(day, site_id, route[index]['plant_id'], ready_after)[1]
                        if ready_after + time_to_next > times[index] + 1e-8:
                            continue
                    rank = cost_floor + wait_cost + violation_cost + rhythm_cost
                    if best is None or rank < best[0]:
                        best = (rank, pid, tid, actual, index, location, ready)
        if best is None:
            continue
        _, pid, tid, planned, index, location, ready = best
        truck = sim.trucks[tid]
        truck.loc, truck.ready = location, ready
        trip = sim.preview(sid, seq, pid, tid, planned)
        if trip['problems'] or trip['rejected']:
            continue
        sim.commit(trip)
        starts[tid].insert(index, planned)
        plans[tid].insert(index, trip)
        rows.append(dict(stream_id=sid, seq=seq, plant_id=pid, truck_id=tid, load_start=planned))
        last_time[sid] = planned
        push(sid, seq+1)
    return rows
