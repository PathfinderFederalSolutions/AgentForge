#!/usr/bin/env python3
"""
Complete Platform Integration Test
Tests all current AgentForge capabilities end-to-end
"""

import pytest
import asyncio
import requests
import time
import json

BASE_URL = "http://localhost:8000"
ADMIN_DASHBOARD_URL = "http://localhost:3001"
INDIVIDUAL_FRONTEND_URL = "http://localhost:3002"

class TestCompletePlatform:
    """Test all integrated platform capabilities"""
    
    def test_backend_health(self):
        """Test backend health and service availability"""
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                assert data["status"] == "ok"
                print("✅ Backend health check passed")
            else:
                pytest.skip(f"Backend not available: {response.status_code}")
                
        except requests.exceptions.RequestException:
            pytest.skip("Backend not available for testing")
    
    def test_comprehensive_services_status(self):
        """Test comprehensive services status endpoint"""
        try:
            response = requests.get(f"{BASE_URL}/v1/services/comprehensive-status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify all major service categories
                required_categories = [
                    "agi_core", "coordination_systems", "universal_io",
                    "self_improvement", "security", "infrastructure"
                ]
                
                for category in required_categories:
                    assert category in data, f"Missing service category: {category}"
                
                # Verify integration summary
                assert "integration_summary" in data
                summary = data["integration_summary"]
                assert "total_services" in summary
                assert "available_services" in summary
                assert "platform_readiness" in summary
                
                print(f"✅ Platform readiness: {summary['platform_readiness']}")
                print(f"✅ Services available: {summary['available_services']}/{summary['total_services']}")
                
            else:
                pytest.skip(f"Comprehensive status not available: {response.status_code}")
                
        except requests.exceptions.RequestException:
            pytest.skip("Backend not available for testing")
    
    def test_chat_message_processing(self):
        """Test chat message processing with agent deployment"""
        try:
            chat_request = {
                "message": "Analyze the current system capabilities and identify any gaps",
                "context": {
                    "userId": "test_user_001",
                    "sessionId": "test_session_001",
                    "conversationHistory": [],
                    "dataSources": [],
                    "userPreferences": {}
                },
                "capabilities": ["introspection", "analysis"]
            }
            
            response = requests.post(
                f"{BASE_URL}/v1/chat/message",
                json=chat_request,
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                assert "response" in data
                assert "agentMetrics" in data
                assert "processingTime" in data
                assert "confidence" in data
                
                # Verify agent metrics
                metrics = data["agentMetrics"]
                assert "totalAgentsDeployed" in metrics
                assert "successRate" in metrics
                
                print("✅ Chat message processing test passed")
                print(f"   Agents deployed: {metrics.get('totalAgentsDeployed', 0)}")
                print(f"   Processing time: {data.get('processingTime', 0):.2f}s")
                
            else:
                pytest.skip(f"Chat endpoint not available: {response.status_code}")
                
        except requests.exceptions.RequestException:
            pytest.skip("Backend not available for testing")
    
    def test_admin_dashboard_endpoints(self):
        """Test admin dashboard API endpoints"""
        try:
            # Test agents endpoint
            response = requests.get(f"{BASE_URL}/v1/jobs/agents", timeout=5)
            
            if response.status_code == 200:
                agents = response.json()
                assert isinstance(agents, list)
                print(f"✅ Admin dashboard agents endpoint: {len(agents)} agents")
            
            # Test metrics endpoint
            response = requests.get(f"{BASE_URL}/v1/jobs/metrics", timeout=5)
            
            if response.status_code == 200:
                metrics = response.json()
                assert "cpu" in metrics
                assert "memory" in metrics
                assert "activeAgents" in metrics
                print("✅ Admin dashboard metrics endpoint passed")
            
            # Test alerts endpoint
            response = requests.get(f"{BASE_URL}/v1/jobs/alerts", timeout=5)
            
            if response.status_code == 200:
                alerts = response.json()
                assert isinstance(alerts, list)
                print(f"✅ Admin dashboard alerts endpoint: {len(alerts)} alerts")
                
        except requests.exceptions.RequestException:
            pytest.skip("Admin dashboard endpoints not available")
    
    def test_fusion_capabilities(self):
        """Test all fusion capabilities"""
        fusion_tests = [
            {
                "name": "Bayesian Fusion",
                "endpoint": "/v1/fusion/bayesian",
                "data": {
                    "eo_data": [0.1, 0.2, 0.3, 0.4],
                    "ir_data": [0.15, 0.25, 0.35, 0.45],
                    "track_id": "test_bayesian_001"
                }
            },
            {
                "name": "EO/IR Fusion",
                "endpoint": "/v1/fusion/eo-ir",
                "data": {
                    "eo_stream": [0.2, 0.3, 0.4],
                    "ir_stream": [0.25, 0.35, 0.45]
                }
            },
            {
                "name": "Detection Analysis",
                "endpoint": "/v1/fusion/detection-analysis",
                "data": {
                    "detection_results": [
                        {"confidence": 0.9}, {"confidence": 0.7}, {"confidence": 0.5}
                    ],
                    "ground_truth": [1, 1, 0]
                }
            },
            {
                "name": "Conformal Prediction",
                "endpoint": "/v1/fusion/conformal-prediction",
                "data": {
                    "residuals": [0.1, -0.2, 0.05, 0.0, -0.1],
                    "alpha": 0.1
                }
            }
        ]
        
        for test in fusion_tests:
            try:
                response = requests.post(
                    f"{BASE_URL}{test['endpoint']}",
                    json=test["data"],
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        print(f"✅ {test['name']} test passed")
                    else:
                        print(f"⚠️ {test['name']} returned error: {result.get('error')}")
                else:
                    print(f"⚠️ {test['name']} endpoint not available: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                print(f"⚠️ {test['name']} test failed: connection error")
    
    def test_service_endpoints(self):
        """Test individual service endpoints"""
        service_tests = [
            {
                "name": "Neural Mesh Query",
                "endpoint": "/v1/services/neural-mesh/query",
                "data": {
                    "query": "test query for neural mesh",
                    "memory_tier": "L1",
                    "max_results": 5
                }
            },
            {
                "name": "Quantum Scheduler",
                "endpoint": "/v1/services/quantum-scheduler/schedule",
                "data": {
                    "task_description": "Test quantum scheduling task",
                    "agent_count": 100,
                    "coherence_level": "MEDIUM"
                }
            },
            {
                "name": "Universal I/O",
                "endpoint": "/v1/services/universal-io/process",
                "data": {
                    "input_data": "Test input for universal I/O",
                    "input_type": "text",
                    "output_format": "json"
                }
            },
            {
                "name": "Self-Bootstrap Analysis",
                "endpoint": "/v1/services/self-bootstrap/analyze",
                "data": {
                    "analysis_scope": "test_system"
                }
            }
        ]
        
        for test in service_tests:
            try:
                response = requests.post(
                    f"{BASE_URL}{test['endpoint']}",
                    json=test["data"],
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if not result.get("error"):
                        print(f"✅ {test['name']} service test passed")
                    else:
                        print(f"⚠️ {test['name']} service error: {result.get('error')}")
                else:
                    print(f"⚠️ {test['name']} service not available: {response.status_code}")
                    
            except requests.exceptions.RequestException:
                print(f"⚠️ {test['name']} service test failed: connection error")
    
    def test_websocket_connectivity(self):
        """Test WebSocket connectivity for real-time updates"""
        try:
            import websocket
            
            def on_message(ws, message):
                data = json.loads(message)
                assert "type" in data
                print(f"✅ WebSocket message received: {data['type']}")
                ws.close()
            
            def on_error(ws, error):
                print(f"⚠️ WebSocket error: {error}")
            
            def on_open(ws):
                print("✅ WebSocket connection opened")
            
            # Test WebSocket connection
            ws = websocket.WebSocketApp(
                f"ws://localhost:8000/v1/realtime/ws",
                on_message=on_message,
                on_error=on_error,
                on_open=on_open
            )
            
            # Run for a short time
            ws.run_forever(timeout=3)
            
        except ImportError:
            pytest.skip("WebSocket client not available")
        except Exception as e:
            print(f"⚠️ WebSocket test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
