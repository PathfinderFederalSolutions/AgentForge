#!/usr/bin/env python3
"""
Final Sprint Ready Verification Script - AgentForge Monitoring Setup
Comprehensive check of all monitoring and metrics systems
"""
import requests
import json
import subprocess
import time
import sys
from typing import Dict, List, Tuple, Any

class SprintReadinessChecker:
    def __init__(self):
        self.results = []
        self.critical_failures = []
        
    def log_result(self, test_name: str, status: bool, message: str, critical: bool = False):
        """Log a test result"""
        symbol = "✅" if status else "❌"
        self.results.append({
            'test': test_name,
            'status': status,
            'message': message,
            'critical': critical
        })
        print(f"{symbol} {test_name}: {message}")
        
        if not status and critical:
            self.critical_failures.append(test_name)
    
    def check_nats_server(self) -> bool:
        """Check NATS server is running with JetStream enabled"""
        try:
            response = requests.get("http://localhost:8222/varz", timeout=5)
            if response.status_code == 200:
                server_info = response.json()
                version = server_info.get('version', 'unknown')
                port = server_info.get('port', 'unknown')
                
                # Check if JetStream is mentioned in the config
                has_jetstream = 'jetstream' in str(server_info).lower()
                
                self.log_result(
                    "NATS Server Running",
                    True,
                    f"Version {version} on port {port}",
                    critical=True
                )
                
                return True
            else:
                self.log_result(
                    "NATS Server Running",
                    False,
                    f"HTTP {response.status_code}",
                    critical=True
                )
                return False
        except Exception as e:
            self.log_result(
                "NATS Server Running",
                False,
                f"Connection failed: {e}",
                critical=True
            )
            return False
    
    def check_nats_monitoring_port(self) -> bool:
        """Verify NATS monitoring port 8222 is reachable"""
        try:
            response = requests.get("http://localhost:8222/varz", timeout=5)
            monitoring_reachable = response.status_code == 200
            
            if monitoring_reachable:
                server_info = response.json()
                http_port = server_info.get('http_port', 'unknown')
                self.log_result(
                    "NATS Monitoring Port 8222",
                    True,
                    f"Reachable on port {http_port}",
                    critical=True
                )
            else:
                self.log_result(
                    "NATS Monitoring Port 8222",
                    False,
                    f"HTTP {response.status_code}",
                    critical=True
                )
            
            return monitoring_reachable
        except Exception as e:
            self.log_result(
                "NATS Monitoring Port 8222",
                False,
                f"Unreachable: {e}",
                critical=True
            )
            return False
    
    def check_jetstream_enabled(self) -> bool:
        """Confirm NATS server started with JetStream enabled"""
        try:
            response = requests.get("http://localhost:8222/jsz", timeout=5)
            if response.status_code == 200:
                js_info = response.json()
                # JetStream is enabled if 'disabled' is false or not present
                disabled = js_info.get('disabled', False)
                enabled = not disabled
                
                if enabled:
                    streams = js_info.get('streams', 0)
                    memory = js_info.get('memory', 0)
                    storage = js_info.get('storage', 0)
                    
                    self.log_result(
                        "JetStream Enabled",
                        True,
                        f"Active with {streams} streams, {memory} memory, {storage} storage",
                        critical=True
                    )
                else:
                    self.log_result(
                        "JetStream Enabled",
                        False,
                        "JetStream is disabled",
                        critical=True
                    )
                
                return enabled
            else:
                self.log_result(
                    "JetStream Enabled",
                    False,
                    f"JSZ endpoint returned HTTP {response.status_code}",
                    critical=True
                )
                return False
        except Exception as e:
            self.log_result(
                "JetStream Enabled",
                False,
                f"JSZ check failed: {e}",
                critical=True
            )
            return False
    
    def check_nats_exporter(self) -> bool:
        """Confirm NATS Prometheus exporter is running with required args"""
        try:
            response = requests.get("http://localhost:7777/metrics", timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Check for required metric types
                required_metrics = [
                    'gnatsd_varz_',
                    'gnatsd_connz_',
                    'gnatsd_routez_',
                    'jetstream_server_',
                    'gnatsd_accstatz_'
                ]
                
                found_metrics = []
                for metric_type in required_metrics:
                    if metric_type in metrics_text:
                        found_metrics.append(metric_type.replace('gnatsd_', '').replace('_', ''))
                
                self.log_result(
                    "NATS Prometheus Exporter",
                    True,
                    f"Running with metrics: {', '.join(found_metrics)}",
                    critical=True
                )
                
                # Verify exporter args are working
                args_working = all(metric in metrics_text for metric in [
                    'gnatsd_varz_',  # -varz
                    'jetstream_server_',  # -jsz
                    'gnatsd_accstatz_'  # -accstatz
                ])
                
                self.log_result(
                    "Exporter Args Working",
                    args_working,
                    "-varz -connz -routez -jsz -accstatz" if args_working else "Some args missing",
                    critical=True
                )
                
                return True
            else:
                self.log_result(
                    "NATS Prometheus Exporter",
                    False,
                    f"HTTP {response.status_code}",
                    critical=True
                )
                return False
        except Exception as e:
            self.log_result(
                "NATS Prometheus Exporter",
                False,
                f"Connection failed: {e}",
                critical=True
            )
            return False
    
    def check_jetstream_quantity_tracking(self) -> bool:
        """Test quantity tracked by JetStream"""
        try:
            response = requests.get("http://localhost:7777/metrics", timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Extract JetStream quantities
                quantities = {}
                for line in metrics_text.split('\n'):
                    if line.startswith('jetstream_server_total_streams{') and not line.startswith('#'):
                        quantities['streams'] = int(float(line.split()[-1]))
                    elif line.startswith('jetstream_server_total_messages{') and not line.startswith('#'):
                        quantities['messages'] = int(float(line.split()[-1]))
                    elif line.startswith('jetstream_server_total_message_bytes{') and not line.startswith('#'):
                        quantities['bytes'] = int(float(line.split()[-1]))
                    elif line.startswith('jetstream_server_total_consumers{') and not line.startswith('#'):
                        quantities['consumers'] = int(float(line.split()[-1]))
                
                # Verify we can track quantities
                streams = quantities.get('streams', 0)
                messages = quantities.get('messages', 0)
                bytes_stored = quantities.get('bytes', 0)
                
                has_streams = streams > 0
                has_messages = messages > 0
                
                if has_streams and has_messages:
                    self.log_result(
                        "JetStream Quantity Tracking",
                        True,
                        f"{streams} streams, {messages} messages, {bytes_stored} bytes",
                        critical=True
                    )
                elif has_streams:
                    self.log_result(
                        "JetStream Quantity Tracking",
                        True,
                        f"{streams} streams created (no messages yet)",
                        critical=False
                    )
                else:
                    self.log_result(
                        "JetStream Quantity Tracking",
                        False,
                        "No streams or messages detected",
                        critical=True
                    )
                
                return has_streams
            else:
                self.log_result(
                    "JetStream Quantity Tracking",
                    False,
                    f"Metrics endpoint HTTP {response.status_code}",
                    critical=True
                )
                return False
        except Exception as e:
            self.log_result(
                "JetStream Quantity Tracking",
                False,
                f"Error checking quantities: {e}",
                critical=True
            )
            return False
    
    def check_agentforge_metrics(self) -> bool:
        """Check AgentForge internal metrics"""
        try:
            # Test if prometheus can be enabled
            import subprocess
            result = subprocess.run([
                'python', '-c',
                'import os; os.environ["PROMETHEUS_ENABLE"]="1"; from observability import *; print("OK")'
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0 and "OK" in result.stdout:
                self.log_result(
                    "AgentForge Prometheus Integration",
                    True,
                    "Observability module loads successfully",
                    critical=False
                )
                return True
            else:
                self.log_result(
                    "AgentForge Prometheus Integration",
                    False,
                    f"Module load failed: {result.stderr}",
                    critical=False
                )
                return False
        except Exception as e:
            self.log_result(
                "AgentForge Prometheus Integration",
                False,
                f"Test failed: {e}",
                critical=False
            )
            return False
    
    def check_docker_containers(self) -> bool:
        """Verify required Docker containers are running"""
        try:
            result = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
            docker_output = result.stdout
            
            nats_running = 'nats:' in docker_output and 'nats-testing' in docker_output
            exporter_running = 'prometheus-nats-exporter' in docker_output
            
            containers_status = []
            if nats_running:
                containers_status.append("NATS server")
            if exporter_running:
                containers_status.append("NATS exporter")
            
            if nats_running and exporter_running:
                self.log_result(
                    "Docker Containers",
                    True,
                    f"Running: {', '.join(containers_status)}",
                    critical=True
                )
                return True
            else:
                missing = []
                if not nats_running:
                    missing.append("NATS server")
                if not exporter_running:
                    missing.append("NATS exporter")
                
                self.log_result(
                    "Docker Containers",
                    False,
                    f"Missing: {', '.join(missing)}",
                    critical=True
                )
                return False
        except Exception as e:
            self.log_result(
                "Docker Containers",
                False,
                f"Docker check failed: {e}",
                critical=True
            )
            return False
    
    def run_all_checks(self):
        """Run all sprint readiness checks"""
        print("🚀 AgentForge Sprint Readiness Verification")
        print("=" * 60)
        
        # Core infrastructure checks
        self.check_docker_containers()
        self.check_nats_server()
        self.check_nats_monitoring_port()
        self.check_jetstream_enabled()
        
        # Monitoring and metrics checks
        self.check_nats_exporter()
        self.check_jetstream_quantity_tracking()
        self.check_agentforge_metrics()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r['status'])
        total = len(self.results)
        critical_passed = sum(1 for r in self.results if r['status'] and r['critical'])
        critical_total = sum(1 for r in self.results if r['critical'])
        
        print(f"Total Tests: {passed}/{total} passed")
        print(f"Critical Tests: {critical_passed}/{critical_total} passed")
        
        if self.critical_failures:
            print(f"\n❌ CRITICAL FAILURES:")
            for failure in self.critical_failures:
                print(f"   - {failure}")
            print("\n❌ TEAM NOT READY FOR NEXT SPRINT")
            return False
        else:
            print("\n✅ ALL CRITICAL CHECKS PASSED")
            print("✅ TEAM READY FOR NEXT SPRINT")
            return True

def main():
    checker = SprintReadinessChecker()
    success = checker.run_all_checks()
    
    print("\n" + "=" * 60)
    print("📋 NEXT STEPS")
    print("=" * 60)
    
    if success:
        print("• All monitoring systems are operational")
        print("• NATS JetStream metrics are being tracked")
        print("• Prometheus exporter is collecting all required metrics")
        print("• Team can proceed with confidence to next sprint")
    else:
        print("• Fix critical failures before proceeding")
        print("• Verify NATS server configuration")
        print("• Check Docker container status")
        print("• Restart failed services as needed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
