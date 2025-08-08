# dqn_planner.py
import math, random, collections, os
import torch
import torch.nn as nn
import torch.optim as optim
from planners import Planner
from problems import ResourceType, HealthcareElements
from simulator import EventType

# --------------------------------------------------------------------- #
#  1. Replay buffer                                                     #
# --------------------------------------------------------------------- #
Transition = collections.namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done', 'mask', 'next_mask')
)

class ReplayMemory:
    def __init__(self, capacity=200_000):
        self.capacity = capacity
        self.buffer   = collections.deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self): return len(self.buffer)

# --------------------------------------------------------------------- #
#  2. Neural network (dueling, mask-aware)                              #
# --------------------------------------------------------------------- #
class Net(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU())
        self.V = nn.Linear(128, 1)
        self.A = nn.Linear(128, action_dim)

    def forward(self, x, mask=None):
        z  = self.backbone(x)
        V  = self.V(z)
        A  = self.A(z)
        if mask is not None:
            A[mask == 0] = -1e9        # forbid illegal moves
        return V + (A - A.mean(dim=1, keepdim=True))

# --------------------------------------------------------------------- #
#  3. The Planner                                                       #
# --------------------------------------------------------------------- #
class DQNPlanner(Planner):
    """
    A Planner that can *train* (off-line) and later *act* (inference-only)
    using the same definition.  Switch by `train=True/False`.
    
    Enhanced with per-diagnosis queues and structured state representation.
    """
    ## hyper-parameters
    GAMMA      = 0.99
    LR         = 1e-4
    BATCH      = 256
    TARGET_UPD = 2_000
    EPS_START  = 1.0
    EPS_END    = 0.05
    EPS_DECAY  = 400_000     # steps

    def __init__(self,
                 model_path="weights.pt",
                 train=False,
                 max_queue_len=50):
        super().__init__()
        self.model_path     = model_path
        self.train_mode     = train
        self.max_queue_len  = max_queue_len   # cut tail to keep state small
        self.step_counter   = 0

        # ---- diagnosis tracking and per-diagnosis queues --------------
        self.diagnosis_dict = {}  # case_id -> diagnosis
        self.arrival_times = {}   # case_id -> arrival_time
        self.diagnosis_encoding = {
            'A1': 0, 'A2': 1, 'A3': 2, 'A4': 3,
            'B1': 4, 'B2': 5, 'B3': 6, 'B4': 7,
            'No diagnosis': 8
        }

        self.diagnosis_queues = {
            'A1': collections.deque(), 'A2': collections.deque(),
            'A3': collections.deque(), 'A4': collections.deque(),
            'B1': collections.deque(), 'B2': collections.deque(),
            'B3': collections.deque(), 'B4': collections.deque(),
            'No diagnosis': collections.deque()
        }

        self.slots_per_diagnosis = {
            'A1': 3, 'A2': 1, 'A3': 1, 'A4': 1,
            'B1': 2, 'B2': 1, 'B3': 1, 'B4': 1,
            'No diagnosis': 1
        }

        self.total_slots = sum(self.slots_per_diagnosis.values())  # 12 slots

        # ---- pending actions & clocks ---------------------------------
        self.pending_actions = []  # list of (ts, state, action, mask, meta, typ)
        self.arrival_ts = {}       # case_id -> arrival ts
        self.wta_open = set()      # case_ids awaiting intake
        self.wth_start = {}        # case_id -> activation ts
        self.schedule_history = []

        # ---- resources -------------------------------------------------
        self.resources_now   = {rt:0 for rt in ResourceType}
        self.future_plan     = collections.defaultdict(dict)   # t -> {rt:n}

        # ---- RL parts --------------------------------------------------
        self.state_dim = (self.total_slots * 2) + 9 + len(ResourceType) + 9
        self.action_dim = (self.total_slots * 24) + 27
        self.net   = Net(self.state_dim, self.action_dim)
        self.tgt   = Net(self.state_dim, self.action_dim)
        self.tgt.load_state_dict(self.net.state_dict())
        self.mem   = ReplayMemory()
        self.opt   = optim.Adam(self.net.parameters(), lr=self.LR)

        if os.path.exists(model_path):
            self.net.load_state_dict(torch.load(model_path))
            self.tgt.load_state_dict(self.net.state_dict())
            print("Loaded weights:", model_path)

    # ================================================================ #
    #  REPORT  – bind simulator stream to our private counters         #
    # ================================================================ #
    def report(self, case_id, element, ts, resource, lifecycle, data=None):
        # Arrival of a new patient: start WTA clock
        if element and element.label in (
            HealthcareElements.PATIENT_REFERAL,
            HealthcareElements.EMERGENCY_PATIENT,
        ):
            self.arrival_ts[case_id] = ts
            self.wta_open.add(case_id)
            if hasattr(element, 'data') and 'diagnosis' in element.data:
                self.diagnosis_dict[case_id] = element.data.get('diagnosis', 'No diagnosis') or 'No diagnosis'
                self.arrival_times[case_id] = ts
                self._add_case_to_queue(case_id, self.diagnosis_dict[case_id], ts)

        # Intake start: compute WTA and credit last plan
        if (
            lifecycle == EventType.START_TASK
            and element
            and element.label == HealthcareElements.INTAKE
            and case_id in self.arrival_ts
        ):
            wta = ts - self.arrival_ts[case_id]
            r = -self._norm_time(wta, cap=336)
            self._credit_last_plan(case_id, r)
            self.wta_open.discard(case_id)

        # Nervousness: replanning close to execution
        if lifecycle == EventType.PLAN_UPDATED and data and 'old_tp' in data:
            old_tp = data['old_tp']
            nerv = max(0.0, 336 - (old_tp - ts)) / 336.0
            self._credit_last_plan(case_id, -nerv)

        # In-hospital waiting time tracking
        is_hosp_task = element and element.label in (
            HealthcareElements.SURGERY,
            HealthcareElements.NURSING,
        )
        if lifecycle == EventType.ACTIVATE_TASK and is_hosp_task:
            self.wth_start[case_id] = ts

        if lifecycle == EventType.START_TASK and is_hosp_task:
            t0 = self.wth_start.pop(case_id, None)
            if t0 is not None:
                wth = ts - t0
                r = -self._norm_time(wth, cap=168)
                self._credit_schedule_in_force(at_ts=t0, reward=r)

        # Hourly cost credit
        if lifecycle == EventType.SCHEDULE_RESOURCES and data and 'cost' in data:
            cost_rate = data['cost'] / 264.0
            self._credit_latest_schedule(-3.0 * cost_rate)

    # ================================================================ #
    #  PLAN  – always called at arbitrary sim events                   #
    # ================================================================ #
    def plan(self, to_plan, to_replan, now):
        self._update_diagnosis_queues(to_plan, to_replan, now)
        state, mask, plan_actions, sched_actions = self._build_structured_state(now)
        action = self._select_action(state, mask)
        plan_count = len(plan_actions)
        if action < plan_count:
            case_id, adm_time = plan_actions[action]
            self._remember_pending(now, state, action, mask, {'case_id': case_id}, 'plan')
            diagnosis = self.diagnosis_dict.get(case_id, 'No diagnosis')
            if case_id in self.diagnosis_queues[diagnosis]:
                self.diagnosis_queues[diagnosis].remove(case_id)
            return [(case_id, adm_time)]
        else:
            s_idx = action - plan_count
            (new_or, new_ab, new_bb), t_eff = sched_actions[s_idx]
            self.resources_now[ResourceType.OR] = new_or
            self.resources_now[ResourceType.A_BED] = new_ab
            self.resources_now[ResourceType.B_BED] = new_bb
            self.future_plan[t_eff][ResourceType.OR] = new_or
            self.future_plan[t_eff][ResourceType.A_BED] = new_ab
            self.future_plan[t_eff][ResourceType.B_BED] = new_bb
            self._remember_pending(
                now,
                state,
                action,
                mask,
                {'t_eff': t_eff, 'triple': (new_or, new_ab, new_bb)},
                'schedule',
            )
            return [
                (ResourceType.OR, t_eff, new_or),
                (ResourceType.A_BED, t_eff, new_ab),
                (ResourceType.B_BED, t_eff, new_bb),
            ]

    # ================================================================ #
    #  SCHEDULE – called daily 18:00                                   #
    # ================================================================ #
    def schedule(self, now):
        state, mask, plan_actions, sched_actions = self._build_structured_state(now)
        action = self._select_action(state, mask)
        plan_count = len(plan_actions)
        if action < plan_count:
            case_id, adm_time = plan_actions[action]
            self._remember_pending(now, state, action, mask, {'case_id': case_id}, 'plan')
            diagnosis = self.diagnosis_dict.get(case_id, 'No diagnosis')
            if case_id in self.diagnosis_queues[diagnosis]:
                self.diagnosis_queues[diagnosis].remove(case_id)
            return [(case_id, adm_time)]
        s_idx = action - plan_count
        (new_or, new_ab, new_bb), t_eff = sched_actions[s_idx]
        self.resources_now[ResourceType.OR] = new_or
        self.resources_now[ResourceType.A_BED] = new_ab
        self.resources_now[ResourceType.B_BED] = new_bb
        self.future_plan[t_eff][ResourceType.OR] = new_or
        self.future_plan[t_eff][ResourceType.A_BED] = new_ab
        self.future_plan[t_eff][ResourceType.B_BED] = new_bb
        self._remember_pending(
            now,
            state,
            action,
            mask,
            {'t_eff': t_eff, 'triple': (new_or, new_ab, new_bb)},
            'schedule',
        )
        return [
            (ResourceType.OR, t_eff, new_or),
            (ResourceType.A_BED, t_eff, new_ab),
            (ResourceType.B_BED, t_eff, new_bb),
        ]

    # ----------------------- helpers ---------------------------------
    def _remember_pending(self, ts, state, action, mask, meta, typ):
        self.pending_actions.append((ts, state, action, mask, meta, typ))

    def _credit_last_plan(self, case_id, r):
        for i in range(len(self.pending_actions) - 1, -1, -1):
            ts, s, a, m, meta, typ = self.pending_actions[i]
            if typ == 'plan' and meta and meta.get('case_id') == case_id:
                self._push_experience(s, a, m, r)
                self.pending_actions.pop(i)
                return

    def _credit_latest_schedule(self, r):
        for i in range(len(self.pending_actions) - 1, -1, -1):
            ts, s, a, m, meta, typ = self.pending_actions[i]
            if typ == 'schedule':
                self._push_experience(s, a, m, r)
                return

    def _credit_schedule_in_force(self, at_ts, reward):
        chosen = None
        for i in range(len(self.pending_actions) - 1, -1, -1):
            ts, s, a, m, meta, typ = self.pending_actions[i]
            if typ == 'schedule' and meta and meta.get('t_eff') <= at_ts:
                chosen = i
                break
        if chosen is not None:
            _, s, a, m, _, _ = self.pending_actions[chosen]
            self._push_experience(s, a, m, reward)

    def _push_experience(self, s, a, m, r, s2=None, done=False, m2=None):
        if self.train_mode:
            self.mem.push(s, a, r, s2, done, m, m2)
            if len(self.mem) >= self.BATCH:
                self._learn()

    def _norm_time(self, dt_hours, cap):
        return min(max(dt_hours, 0), cap) / cap

    def _next_effective_time(self, now, dr_or, dr_ab, dr_bb):
        t_eff = now + 14
        if dr_or < 0 or dr_ab < 0 or dr_bb < 0:
            t_eff = now + 168
        return t_eff

    def _slot_features(self, case_id, diagnosis, now):
        waiting = now - self.arrival_times.get(case_id, now)
        w_norm = min(waiting, 336) / 336.0
        pr = self.diagnosis_encoding.get(diagnosis, 8) / 8.0
        return [w_norm, pr]

    # ================================================================ #
    #  INTERNALS                                                       #
    # ================================================================ #
    def _update_diagnosis_queues(self, to_plan, to_replan, now):
        """Update per-diagnosis queues with new cases, maintaining FIFO order"""
        all_cases = to_plan + to_replan
        
        for case_id in all_cases:
            diagnosis = self.diagnosis_dict.get(case_id, 'No diagnosis')
            arrival_time = self.arrival_times.get(case_id, now)
            
            # Add to appropriate queue if not already there
            if case_id not in self.diagnosis_queues[diagnosis]:
                # Insert in order of arrival time (FIFO)
                self.diagnosis_queues[diagnosis].append(case_id)
                
                # Sort by arrival time (maintain FIFO)
                sorted_queue = sorted(self.diagnosis_queues[diagnosis], 
                                    key=lambda x: self.arrival_times.get(x, 0))
                self.diagnosis_queues[diagnosis] = collections.deque(sorted_queue)
    
    def _add_case_to_queue(self, case_id, diagnosis, arrival_time):
        """Add a case to its appropriate diagnosis queue"""
        if case_id not in self.diagnosis_queues[diagnosis]:
            self.diagnosis_queues[diagnosis].append(case_id)
            
            # Sort by arrival time (maintain FIFO)
            sorted_queue = sorted(self.diagnosis_queues[diagnosis], 
                                key=lambda x: self.arrival_times.get(x, 0))
            self.diagnosis_queues[diagnosis] = collections.deque(sorted_queue)
    
    def _build_structured_state(self, now):
        """Create structured state vector with fixed slots per diagnosis"""
        
        # Initialize state components
        slot_data = []  # features for each slot
        diagnosis_counts = [0] * 9  # count of cases per diagnosis
        
        # Fill slots for each diagnosis
        for diagnosis, num_slots in self.slots_per_diagnosis.items():
            queue = self.diagnosis_queues[diagnosis]
            diagnosis_idx = self.diagnosis_encoding[diagnosis]
            
            # Fill available slots for this diagnosis
            for i in range(num_slots):
                if i < len(queue):
                    case_id = queue[i]
                    slot_data.extend(self._slot_features(case_id, diagnosis, now))
                    diagnosis_counts[diagnosis_idx] += 1
                else:
                    slot_data.extend([0.0, 0.0])
        
        # Resource information
        fut = [self.resources_now[rt] for rt in ResourceType]
        
        # Time features
        hour = now % 24
        dow  = (now//24)%7
        clock = [math.sin(2*math.pi*hour/24), math.cos(2*math.pi*hour/24)]
        dow_onehot = [1 if i==dow else 0 for i in range(7)]
        
        # Construct state vector
        state = torch.tensor(slot_data + diagnosis_counts + fut + clock + dow_onehot,
                           dtype=torch.float32).unsqueeze(0)

        # ------------------- ACTION ENCODING -------------------------
        plan_actions  = []   # [(cid, adm_time)]
        sched_actions = []   # [((or,a,b), t_eff)]
        mask          = torch.zeros(self.action_dim, dtype=torch.uint8).unsqueeze(0)

        # 1. Plan actions: one action per slot per time window
        idx = 0
        for diagnosis, num_slots in self.slots_per_diagnosis.items():
            queue = self.diagnosis_queues[diagnosis]
            for slot_idx in range(num_slots):
                if slot_idx < len(queue):
                    case_id = queue[slot_idx]
                    earliest = now + 24     # ≥ one day ahead
                    for h in range(24):     # 24 hourly slots
                        adm = earliest + h
                        plan_actions.append((case_id, adm))
                        if adm >= earliest: mask[0, idx] = 1
                        idx += 1
                else:
                    # Empty slot - no valid actions
                    idx += 24

        # 2. Schedule actions: set OR, A_BED, B_BED together
        for dr_or in (-1, 0, 1):
            for dr_ab in (-1, 0, 1):
                for dr_bb in (-1, 0, 1):
                    new_or = min(5, max(0, self.resources_now[ResourceType.OR] + dr_or))
                    new_ab = min(30, max(0, self.resources_now[ResourceType.A_BED] + dr_ab))
                    new_bb = min(40, max(0, self.resources_now[ResourceType.B_BED] + dr_bb))
                    t_eff = self._next_effective_time(now, dr_or, dr_ab, dr_bb)
                    sched_actions.append(((new_or, new_ab, new_bb), t_eff))
                    mask[0, idx] = 1
                    idx += 1

        return state, mask, plan_actions, sched_actions

    # ε-greedy --------------------------------------------------------
    def _select_action(self, state, mask):
        if self.train_mode:
            eps = self.EPS_END + (self.EPS_START-self.EPS_END) * \
                  math.exp(-1.*self.step_counter/self.EPS_DECAY)
            self.step_counter += 1
            if random.random() < eps:
                valid = torch.nonzero(mask[0]).squeeze().tolist()
                return random.choice(valid) if valid else 0
        with torch.no_grad():
            q = self.net(state, mask)
            return int(torch.argmax(q).item())

    # learning --------------------------------------------------------
    def _learn(self):
        batch = self.mem.sample(self.BATCH)
        non_final_mask = [i for i, s in enumerate(batch.next_state) if s is not None]
        state = torch.cat(batch.state)
        action = torch.tensor(batch.action).unsqueeze(1)
        reward = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        mask = torch.cat(batch.mask)

        q_sa = self.net(state, mask).gather(1, action)
        q_next = torch.zeros_like(q_sa)
        if non_final_mask:
            non_next = torch.cat([batch.next_state[i] for i in non_final_mask], dim=0)
            next_masks = torch.cat([batch.next_mask[i] for i in non_final_mask], dim=0)
            with torch.no_grad():
                q_online = self.net(non_next, None)
                q_online[next_masks == 0] = -1e9
                a_star = q_online.argmax(dim=1, keepdim=True)
                q_target = self.tgt(non_next, None)
                q_next_vals = q_target.gather(1, a_star)
                for idx, bf in enumerate(non_final_mask):
                    q_next[bf] = q_next_vals[idx]
        done = torch.tensor(batch.done).unsqueeze(1).float()
        target = reward + self.GAMMA * q_next * (1 - done)
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

        if self.step_counter % self.TARGET_UPD == 0:
            self.tgt.load_state_dict(self.net.state_dict())

    # call this at the end of a long training run
    def save(self):
        if self.train_mode:
            torch.save(self.net.state_dict(), self.model_path)
