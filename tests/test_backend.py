"""
DeployMate AI — Unit Tests
Testing core backend functions
"""

import pytest
from chatbot_backend_database_SystemPrompt import (
    is_error_message,
    is_deploy_message,
    is_code_review_message,
    is_dangerous,
    get_thread_id_from_config,
)


# ─── TESTS: is_error_message() ───────────────────────────

def test_error_message_detects_docker():
    """Should detect docker keyword as error message."""
    assert is_error_message("my docker container is failing") == True

def test_error_message_detects_traceback():
    """Should detect traceback as error message."""
    assert is_error_message("Traceback most recent call last") == True

def test_error_message_detects_exception():
    """Should detect exception keyword."""
    assert is_error_message("ModuleNotFoundError: No module named") == True

def test_error_message_normal_question():
    """Normal question should not be detected as error."""
    assert is_error_message("how are you doing today") == False

def test_error_message_case_insensitive():
    """Should work regardless of case."""
    assert is_error_message("DOCKER ERROR occurred") == True


# ─── TESTS: is_deploy_message() ──────────────────────────

def test_deploy_message_detects_deploy():
    """Should detect deploy keyword."""
    assert is_deploy_message("I want to deploy my app") == True

def test_deploy_message_detects_aws():
    """Should detect AWS keyword."""
    assert is_deploy_message("how to host on aws ec2") == True

def test_deploy_message_detects_railway():
    """Should detect Railway keyword."""
    assert is_deploy_message("deploy on railway platform") == True

def test_deploy_message_normal_question():
    """Normal question should not be detected as deploy."""
    assert is_deploy_message("what is python") == False


# ─── TESTS: is_code_review_message() ─────────────────────

def test_code_review_detects_review():
    """Should detect code review request."""
    assert is_code_review_message("review my code please") == True

def test_code_review_detects_backtick():
    """Should detect code block."""
    assert is_code_review_message("```python print('hello')```") == True

def test_code_review_detects_def():
    """Should detect function definition."""
    assert is_code_review_message("def my_function():") == True

def test_code_review_normal_question():
    """Normal question should not be detected as code review."""
    assert is_code_review_message("what is docker") == False


# ─── TESTS: is_dangerous() ───────────────────────────────

def test_dangerous_detects_rm_rf():
    """Should detect rm -rf as dangerous."""
    assert is_dangerous("run rm -rf /var/log") == True

def test_dangerous_detects_drop_database():
    """Should detect drop database as dangerous."""
    assert is_dangerous("drop database mydb") == True

def test_dangerous_detects_sudo_rm():
    """Should detect sudo rm as dangerous."""
    assert is_dangerous("sudo rm se files delete karo") == True

def test_dangerous_safe_command():
    """Safe command should not be detected as dangerous."""
    assert is_dangerous("restart the nginx server") == False

def test_dangerous_case_insensitive():
    """Should work regardless of case."""
    assert is_dangerous("RM -RF /logs") == True


# ─── TESTS: get_thread_id_from_config() ──────────────────

def test_thread_id_extracted_correctly():
    """Should extract thread_id from config correctly."""
    config = {"configurable": {"thread_id": "test-123"}}
    assert get_thread_id_from_config(config) == "test-123"

def test_thread_id_empty_config():
    """Should return empty string for empty config."""
    config = {}
    assert get_thread_id_from_config(config) == ""

def test_thread_id_missing_thread():
    """Should return empty string when thread_id missing."""
    config = {"configurable": {}}
    assert get_thread_id_from_config(config) == ""