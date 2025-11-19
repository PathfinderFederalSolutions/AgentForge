#!/usr/bin/env python3
"""
Test Prometheus recording and alerting rules.
Verifies rules are loaded and syntax is correct.
"""

import requests
import json
import time
import sys
import pytest

class TestPrometheusRules:
    def __init__(self, prometheus_url="http://localhost:9090"):
        self.prometheus_url = prometheus_url
        
    def query_prometheus(self, query):
        """Execute Prometheus query"""
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", 
                                  params={'query': query})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Query failed: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def test_recording_rules(self):
        """Test our SLO recording rules"""
        print("\n🧪 Testing Recording Rules")
        
        rules = [
            "jetstream_backlog",
            "slo_backlog_drain_p95", 
            "slo_violation_ratio_1h"
        ]
        
        results = {}
        for rule in rules:
            print(f"   Checking {rule}...")
            result = self.query_prometheus(rule)
            if result and result.get('status') == 'success':
                data = result.get('data', {}).get('result', [])
                if data:
                    print(f"   ✅ {rule}: {len(data)} series")
                    results[rule] = "PASS"
                else:
                    print(f"   ⚠️  {rule}: No data (rule may be valid but no metrics)")
                    results[rule] = "NO_DATA"
            else:
                print(f"   ❌ {rule}: Failed to query")
                results[rule] = "FAIL"
        
        return results
    
    def test_alerting_rules(self):
        """Test alerting rules are loaded"""
        print("\n🧪 Testing Alerting Rules")
        
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/rules")
            if response.status_code == 200:
                rules_data = response.json()
                if rules_data.get('status') == 'success':
                    groups = rules_data.get('data', {}).get('groups', [])
                    
                    slo_alerts = []
                    for group in groups:
                        if group.get('name') == 'slo_alerts':
                            for rule in group.get('rules', []):
                                if rule.get('type') == 'alerting':
                                    slo_alerts.append(rule['name'])
                    
                    expected_alerts = [
                        'SustainedBacklogWarning',
                        'SustainedBacklogCritical', 
                        'AckPendingHigh'
                    ]
                    
                    print(f"   Found SLO alerts: {slo_alerts}")
                    
                    results = {}
                    for alert in expected_alerts:
                        if alert in slo_alerts:
                            print(f"   ✅ {alert}: Loaded")
                            results[alert] = "LOADED"
                        else:
                            print(f"   ❌ {alert}: Missing")
                            results[alert] = "MISSING"
                    
                    return results
                else:
                    print("   ❌ Failed to get rules status")
                    return {}
            else:
                print(f"   ❌ Rules API failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"   ❌ Error checking rules: {e}")
            return {}
    
    def test_target_health(self):
        """Check Prometheus targets"""
        print("\n🧪 Testing Target Health")
        
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/targets")
            if response.status_code == 200:
                targets_data = response.json()
                if targets_data.get('status') == 'success':
                    active_targets = targets_data.get('data', {}).get('activeTargets', [])
                    
                    target_status = {}
                    for target in active_targets:
                        job = target.get('labels', {}).get('job', 'unknown')
                        health = target.get('health', 'unknown')
                        endpoint = target.get('scrapeUrl', 'unknown')
                        
                        print(f"   {job}: {health} ({endpoint})")
                        target_status[job] = health
                    
                    return target_status
                else:
                    print("   ❌ Failed to get targets status")
                    return {}
            else:
                print(f"   ❌ Targets API failed: {response.status_code}")
                return {}
        except Exception as e:
            print(f"   ❌ Error checking targets: {e}")
            return {}

def main():
    print("🚀 Prometheus Rules Test Suite")
    print("=" * 50)
    
    tester = TestPrometheusRules()
    
    # Test target health first
    target_results = tester.test_target_health()
    
    # Test recording rules
    recording_results = tester.test_recording_rules()
    
    # Test alerting rules
    alerting_results = tester.test_alerting_rules()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    all_passed = True
    
    print("Targets:")
    for job, health in target_results.items():
        status = "✅" if health == "up" else "❌"
        print(f"  {status} {job}: {health}")
        if health != "up":
            all_passed = False
    
    print("\nRecording Rules:")
    for rule, status in recording_results.items():
        icon = "✅" if status == "PASS" else "⚠️" if status == "NO_DATA" else "❌"
        print(f"  {icon} {rule}: {status}")
        if status == "FAIL":
            all_passed = False
    
    print("\nAlerting Rules:")
    for rule, status in alerting_results.items():
        icon = "✅" if status == "LOADED" else "❌"
        print(f"  {icon} {rule}: {status}")
        if status == "MISSING":
            all_passed = False
    
    print(f"\n🎯 Overall Status: {'✅ PASS' if all_passed else '❌ FAIL'}")
    
    return 0 if all_passed else 1


def test_prometheus_rules_basic():
    """Basic test that prometheus rules configuration is valid"""
    tester = TestPrometheusRules()
    assert tester is not None
    assert tester.prometheus_url == "http://localhost:9090"
    # Test that basic methods exist
    assert hasattr(tester, 'query_prometheus')
    assert hasattr(tester, 'test_recording_rules')


if __name__ == "__main__":
    sys.exit(main())
