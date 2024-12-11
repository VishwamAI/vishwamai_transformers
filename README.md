# Vishwamai Transformers

## Project Overview
This repository contains an implementation of transformer models for natural language processing tasks.

## Test Coverage
The project includes comprehensive tests covering the following components:
- Multi-head Attention Mechanisms
- Encoder Architecture
- Decoder Architecture

### Test Results
All tests are passing successfully:
- ✅ Multi-head Attention Output Shape
- ✅ Decoder Layer Output Shape
- ✅ Decoder Output Shape
- ✅ Encoder Layer Output Shape
- ✅ Encoder Output Shape

## Project Structure
```
vishwamai_transformers/
├── tests/
│   └── test_models/
│       ├── test_attention_mechanisms.py
│       ├── test_decoder.py
│       └── test_encoder.py
```

## Running Tests
To run the tests:
```bash
pytest  # For basic test execution
pytest -v  # For verbose test output
```

## Environment
- Python 3.10.12
- pytest 8.3.4
- pluggy 1.5.0