# STL Texturizer Test Suite

This directory contains automated tests for the STL Texturizer application.

## Test Files

### `test_production.py`
Production deployment tests for https://fiuncholabs.com

**Key Tests:**
1. **Server Reachability** - Verifies the production server is online
2. **Health Check** - Tests the `/health` endpoint
3. **Estimate Accuracy** - Tests that estimates are accurate for fuzzy skin parameters
4. **Processing Capability** - Verifies that files the estimate says can be processed actually can be
5. **Time Estimate Accuracy** - Compares estimated vs actual processing time
6. **Error Handling** - Tests invalid inputs and edge cases
7. **Consistency** - Verifies estimates are consistent across multiple calls

**Critical Test:**
- `test_04_estimate_accuracy_fuzzy_skin` - Verifies the estimate correctly identifies that simple-corner.stl can be processed with fuzzy skin settings (point_distance=2.0)
- `test_05_actual_processing_fuzzy_skin` - Verifies actual processing works and compares timing with estimate

## Running Tests

### Run all production tests:
```bash
python3 -m pytest tests/test_production.py -v
```

Or using unittest:
```bash
python3 tests/test_production.py
```

### Run specific test:
```bash
python3 -m pytest tests/test_production.py::TestProductionDeployment::test_04_estimate_accuracy_fuzzy_skin -v
```

### Run with more detailed output:
```bash
python3 tests/test_production.py
```

## Requirements

Install test dependencies:
```bash
pip3 install requests pytest
```

## Test Data

Tests use `simple-corner.stl` from the project root directory. This file must exist for tests to run.

## Expected Results

For `simple-corner.stl` with fuzzy skin parameters (point_distance=2.0):
- **Can process**: Should be `True`
- **Estimated triangles**: Should be well under the 20M limit
- **Estimated time**: Should be reasonable (typically < 60 seconds)
- **Actual processing**: Should complete successfully
- **Time accuracy**: Should be within reasonable margin (ideally within 2x of estimate)

## Interpreting Results

### Success Indicators
- ✓ All tests pass
- ✓ `can_process` is `True` for fuzzy skin variant
- ✓ Actual processing completes without timeout
- ✓ Time estimate is reasonably accurate (within 100% margin)

### Warning Signs
- ⚠ Time estimate significantly off (> 100% difference)
- ⚠ Estimate says cannot process when it should be able to
- ⚠ Processing times out despite estimate saying it's possible

### Failures
- ✗ Server unreachable
- ✗ Estimate says can process but actual processing fails
- ✗ Estimate consistently wrong about processability
- ✗ Processing produces invalid output

## Continuous Integration

These tests can be run as part of CI/CD pipeline to verify production deployment health after updates.

## Future Tests

Planned additions:
- Load testing with multiple concurrent requests
- Large file handling tests
- Different noise type variations
- Rotation parameter tests
- Memory usage verification
- Output STL file validation (triangle count, file structure)
