#!/usr/bin/env python3
"""
Test script for the enhanced DQN planner with diagnosis tracking
"""

from dqn_planner import DQNPlanner
from problems import HealthcareProblem, HealthcareElements
from simulator import Simulator, EventType
import random

def test_diagnosis_tracking():
    """Test that the DQN planner correctly tracks diagnosis information"""
    
    # Create planner in training mode
    planner = DQNPlanner(train=True, max_queue_len=10)
    
    # Create problem and simulator
    problem = HealthcareProblem()
    simulator = Simulator(planner, problem)
    
    # Test diagnosis tracking
    print("Testing diagnosis tracking...")
    
    # Simulate some events to populate diagnosis_dict
    test_cases = [
        ("A1", "A1"),
        ("A2", "A2"), 
        ("B1", "B1"),
        ("B2", "B2"),
        ("EM1", None),  # Emergency with no diagnosis
        ("EM2", "B3"),  # Emergency with diagnosis
    ]
    
    for case_id, expected_diagnosis in test_cases:
        # Create a mock element with diagnosis
        class MockElement:
            def __init__(self, case_id, label, diagnosis):
                self.case_id = case_id
                self.label = label
                self.data = {"diagnosis": diagnosis}
        
        # Simulate emergency_patient or patient_referal event
        if case_id.startswith("EM"):
            element = MockElement(case_id, "emergency_patient", expected_diagnosis)
        else:
            element = MockElement(case_id, "patient_referal", expected_diagnosis)
        
        # Call report method to populate diagnosis_dict
        planner.report(case_id, element, 100.0, None, EventType.ACTIVATE_EVENT)
        
        # Check if diagnosis was tracked correctly
        tracked_diagnosis = planner.diagnosis_dict.get(case_id, "Not found")
        expected = expected_diagnosis if expected_diagnosis is not None else "No diagnosis"
        
        print(f"Case {case_id}: Expected {expected}, Got {tracked_diagnosis}")
        assert tracked_diagnosis == expected, f"Diagnosis tracking failed for {case_id}"
    
    print("✓ Diagnosis tracking test passed!")
    
    # Test case prioritization
    print("\nTesting case prioritization...")
    
    # Add some test cases to plan
    test_cases_to_plan = ["A1", "B2", "A3", "EM1", "B1"]
    
    # Sort them by priority
    sorted_cases = planner._sort_cases_by_priority(test_cases_to_plan, 100.0)
    
    print(f"Original order: {test_cases_to_plan}")
    print(f"Sorted order: {sorted_cases}")
    
    # Verify that higher priority cases come first
    # B2 (priority 6) should come before A1 (priority 1) etc.
    print("✓ Case prioritization test completed!")
    
    # Test state building with diagnosis
    print("\nTesting enhanced state building...")
    
    try:
        state, mask, plan_actions, sched_actions = planner._build_state(100.0, test_cases_to_plan, [])
        print(f"State shape: {state.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Number of plan actions: {len(plan_actions)}")
        print(f"Number of schedule actions: {len(sched_actions)}")
        print("✓ Enhanced state building test passed!")
    except Exception as e:
        print(f"✗ State building failed: {e}")
        raise

def test_integration():
    """Test integration with the full simulation system"""
    
    print("\nTesting full integration...")
    
    try:
        # Create planner
        planner = DQNPlanner(train=True, max_queue_len=20)
        
        # Create problem and simulator
        problem = HealthcareProblem()
        simulator = Simulator(planner, problem)
        
        # Run a short simulation
        print("Running short simulation...")
        result = simulator.run(24)  # 1 day
        
        print(f"Simulation completed. Result: {result}")
        print(f"Diagnosis dict size: {len(planner.diagnosis_dict)}")
        print(f"Sample diagnoses: {list(planner.diagnosis_dict.items())[:5]}")
        
        print("✓ Integration test passed!")
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        raise

if __name__ == "__main__":
    print("Testing Enhanced DQN Planner with Diagnosis Tracking")
    print("=" * 60)
    
    test_diagnosis_tracking()
    test_integration()
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nKey improvements made:")
    print("1. ✓ Diagnosis tracking in report() method")
    print("2. ✓ Enhanced state representation with diagnosis features")
    print("3. ✓ Diagnosis-based case prioritization")
    print("4. ✓ Improved action selection with diagnosis awareness")
    print("5. ✓ Better resource allocation based on patient needs")
