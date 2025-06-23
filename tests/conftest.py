import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration object for testing."""
    config = Mock()
    config.batch_size = 32
    config.learning_rate = 0.001
    config.epochs = 10
    config.hidden_size = 768
    config.num_attention_heads = 12
    config.num_hidden_layers = 12
    config.vocab_size = 30522
    config.max_position_embeddings = 512
    config.intermediate_size = 3072
    config.hidden_act = "gelu"
    config.hidden_dropout_prob = 0.1
    config.attention_probs_dropout_prob = 0.1
    return config


@pytest.fixture
def mock_model():
    """Mock model object for testing."""
    model = MagicMock()
    model.predict = MagicMock(return_value=[0.1, 0.9])
    model.train = MagicMock()
    model.evaluate = MagicMock(return_value={"loss": 0.5, "accuracy": 0.85})
    return model


@pytest.fixture
def sample_text_data():
    """Sample text data for testing."""
    return [
        "This is a sample sentence for testing.",
        "Another example text for our tests.",
        "BERT models are powerful for NLP tasks.",
    ]


@pytest.fixture
def sample_tokenized_data():
    """Sample tokenized data for testing."""
    return {
        "input_ids": [[101, 2023, 2003, 1037, 6876, 102], 
                      [101, 2178, 2742, 3793, 102],
                      [101, 14324, 2819, 2024, 3928, 102]],
        "attention_mask": [[1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1]],
        "token_type_ids": [[0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0]]
    }


@pytest.fixture
def mock_data_loader():
    """Mock data loader for testing."""
    loader = Mock()
    loader.__iter__ = Mock(return_value=iter([
        {"inputs": "sample1", "labels": 0},
        {"inputs": "sample2", "labels": 1},
    ]))
    loader.__len__ = Mock(return_value=2)
    return loader


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    yield captured_output
    captured_output.close()


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds for reproducible tests."""
    import random
    
    random.seed(42)
    
    try:
        import numpy as np
        np.random.seed(42)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
    except ImportError:
        pass