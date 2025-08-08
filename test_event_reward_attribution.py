#!/usr/bin/env python3
"""Unit tests for event-driven reward attribution in DQNPlanner"""

import math
import torch

from dqn_planner import DQNPlanner
from problems import HealthcareElements
from simulator import EventType

# Ensure PLAN_UPDATED exists for compatibility with the planner's report logic
if not hasattr(EventType, "PLAN_UPDATED"):
    EventType.PLAN_UPDATED = EventType.PLAN_EVENTS


class MockElement:
    def __init__(self, label, diagnosis=None):
        self.label = label
        self.data = {}
        if diagnosis is not None:
            self.data['diagnosis'] = diagnosis


def _dummy_state(planner):
    state = torch.zeros((1, planner.state_dim))
    mask = torch.ones((1, planner.action_dim))
    return state, mask


def test_wta_reward_attribution():
    """Intake START should credit the last plan for that case"""
    planner = DQNPlanner(train=True)

    # Arrival of case C1
    arr_elem = MockElement(HealthcareElements.PATIENT_REFERAL, 'A1')
    planner.report('C1', arr_elem, 0.0, None, EventType.ACTIVATE_EVENT)

    # Remember a pending plan for C1
    state, mask = _dummy_state(planner)
    planner._remember_pending(0.0, state, 0, mask, {'case_id': 'C1'}, 'plan')

    # Intake start after 48 hours
    intake_elem = MockElement(HealthcareElements.INTAKE)
    planner.report('C1', intake_elem, 48.0, None, EventType.START_TASK)

    assert len(planner.mem.buffer) == 1, "Expected one experience for WTA"
    reward = planner.mem.buffer[0].reward
    assert math.isclose(reward, -48 / 336.0, rel_tol=1e-3)


def test_nerv_reward_attribution():
    """PLAN_UPDATED should credit the latest plan with NERV penalty"""
    planner = DQNPlanner(train=True)

    state, mask = _dummy_state(planner)
    planner._remember_pending(0.0, state, 1, mask, {'case_id': 'C1'}, 'plan')

    planner.report('C1', None, 100.0, None, EventType.PLAN_UPDATED, data={'old_tp': 120.0})

    assert len(planner.mem.buffer) == 1
    reward = planner.mem.buffer[0].reward
    expected = -(336 - 20) / 336.0
    assert math.isclose(reward, expected, rel_tol=1e-3)


def test_wth_reward_attribution():
    """ACTIVATE->START for hospital tasks should credit schedule in force"""
    planner = DQNPlanner(train=True)

    state, mask = _dummy_state(planner)
    planner._remember_pending(0.0, state, 5, mask, {'t_eff': 150.0, 'triple': (1, 1, 1)}, 'schedule')

    surg_elem = MockElement(HealthcareElements.SURGERY)
    planner.report('C2', surg_elem, 200.0, None, EventType.ACTIVATE_TASK)
    planner.report('C2', surg_elem, 212.0, None, EventType.START_TASK)

    assert len(planner.mem.buffer) == 1
    reward = planner.mem.buffer[0].reward
    assert math.isclose(reward, -12 / 168.0, rel_tol=1e-3)


def test_cost_reward_attribution():
    """Hourly cost events should credit the latest schedule"""
    planner = DQNPlanner(train=True)
    state, mask = _dummy_state(planner)
    planner._remember_pending(0.0, state, 2, mask, {'t_eff': 0.0, 'triple': (1, 1, 1)}, 'schedule')

    planner.report(None, None, 1.0, None, EventType.SCHEDULE_RESOURCES, data={'cost': 264.0})

    assert len(planner.mem.buffer) == 1
    reward = planner.mem.buffer[0].reward
    assert math.isclose(reward, -3.0, rel_tol=1e-3)

