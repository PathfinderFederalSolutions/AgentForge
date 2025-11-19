# AgentForge Consolidated Test Suite

This directory contains the complete test suite for AgentForge, organized by capabilities and features.

## Test Organization

### Core AI Services (`core_ai_services/`)
- Neural mesh coordination tests
- AGI introspection and evolution tests  
- Self-coding AGI tests
- Enhanced backend API tests

### Advanced Services (`advanced_services/`)
- Quantum scheduler tests (million-scale coordination)
- Universal I/O tests (70+ inputs, 45+ outputs)
- Mega-swarm coordination tests
- Self-bootstrap system tests

### Fusion Capabilities (`fusion_capabilities/`)
- Bayesian sensor fusion tests
- Conformal prediction tests
- EO/IR sensor fusion tests
- ROC/DET analysis tests
- Advanced fusion pipeline tests

### Infrastructure (`infrastructure/`)
- Import and dependency tests
- Router and orchestrator tests
- API endpoint tests
- Database and storage tests

### Libraries (`libraries/`)
- AF-Common library tests (types, logging, config, tracing)
- AF-Schemas tests (agent and event schemas)
- AF-Messaging tests (NATS integration)

### Integration (`integration/`)
- Complete platform integration tests
- Cross-service communication tests
- End-to-end workflow tests

### E2E (`e2e/`)
- Full system end-to-end tests
- Performance and load tests
- Real-world scenario tests

## Running Tests

### Run All Tests
```bash
cd tests_consolidated
pytest -v
```

### Run Specific Categories
```bash
# Core AI services
pytest core_ai_services/ -v

# Fusion capabilities
pytest fusion_capabilities/ -v

# Integration tests
pytest integration/ -v
```

### Environment Variables

- `ENABLE_INTEGRATION=1` - Enable integration tests (require running backend)
- `ENABLE_BACKEND_TESTS=1` - Enable backend API tests (default: enabled)
- `ENABLE_FUSION_TESTS=1` - Enable fusion capability tests (default: enabled)
- `ENABLE_SERVICE_TESTS=1` - Enable individual service tests (default: enabled)

### Test Requirements

1. **Backend Running**: Integration tests require backend on port 8000
2. **Dependencies**: All test dependencies in requirements.dev.txt
3. **Environment**: Test environment variables in conftest.py

## Test Coverage

This test suite covers:
- ✅ All 15+ AgentForge services
- ✅ Advanced fusion capabilities (Bayesian, EO/IR, ROC/DET)
- ✅ Database and analytics systems
- ✅ Enhanced logging and configuration
- ✅ AF-Common, AF-Schemas, AF-Messaging libraries
- ✅ Admin dashboard and individual frontend APIs
- ✅ Real-time WebSocket communication
- ✅ Error handling and retry logic
- ✅ Performance and observability
