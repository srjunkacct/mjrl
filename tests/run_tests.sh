#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MJRL Environment Testing Suite ===${NC}\n"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}pytest not found. Installing...${NC}"
    pip install pytest pytest-cov
fi

# Function to run tests and check status
run_test() {
    echo -e "\n${BLUE}Running: $1${NC}"
    $1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Test passed${NC}"
    else
        echo -e "${RED}✗ Test failed${NC}"
        exit 1
    fi
}

# Basic environment tests
echo -e "\n${BLUE}=== Basic Environment Tests ===${NC}"
run_test "pytest tests/test_environments.py -v"

# Policy integration tests
echo -e "\n${BLUE}=== Policy Integration Tests ===${NC}"
run_test "pytest tests/test_policy_integration.py -v"

# Run specific test cases
echo -e "\n${BLUE}=== Specific Test Cases ===${NC}"
run_test "pytest tests/test_environments.py::test_env_creation -v"
run_test "pytest tests/test_environments.py::test_full_episode -v"
run_test "pytest tests/test_policy_integration.py::test_policy_action -v"

# Run with coverage report
echo -e "\n${BLUE}=== Test Coverage Report ===${NC}"
run_test "pytest --cov=mjrl tests/ --cov-report term-missing"

echo -e "\n${GREEN}All tests completed successfully!${NC}" 