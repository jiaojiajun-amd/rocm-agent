"""Tests for observation compression in MiniAgent."""

import pytest
from unittest.mock import Mock, MagicMock
from minisweagent.agents.mini import MiniAgent, AgentConfig


class TestObservationCompression:
    """Test observation compression functionality."""
    
    def test_short_observation_not_compressed(self):
        """Short observations should not trigger compression."""
        model = Mock()
        env = Mock()
        
        agent = MiniAgent(model, env, max_observation_tokens=1000)
        
        short_text = "Hello world" * 10  # ~120 chars, ~30 tokens
        agent.add_message("user", short_text)
        
        assert len(agent.messages) == 1
        assert agent.messages[0]["content"] == short_text
        assert "full_content" not in agent.messages[0]
    
    def test_long_observation_compressed(self):
        """Long observations should trigger compression."""
        model = Mock()
        model.query = Mock(return_value={"content": "This is a summary of the long output."})
        model.get_template_vars = Mock(return_value={})
        env = Mock()
        env.get_template_vars = Mock(return_value={})
        
        agent = MiniAgent(model, env, max_observation_tokens=100)
        
        # Create a long observation (>400 chars = >100 tokens)
        long_text = "x" * 500
        
        reasoning = agent._reason_about_observation(long_text)
        
        assert model.query.called
        assert "<observation_summary>" in reasoning
        assert "summary" in reasoning.lower()
    
    def test_get_full_messages_restores_content(self):
        """get_full_messages should restore full_content to content."""
        model = Mock()
        env = Mock()
        
        agent = MiniAgent(model, env)
        
        # Add message with full_content
        agent.add_message("user", "short summary", full_content="very long original content")
        
        full_messages = agent.get_full_messages()
        
        assert len(full_messages) == 1
        assert full_messages[0]["content"] == "very long original content"
        assert "full_content" not in full_messages[0]
    
    def test_get_full_messages_preserves_normal_messages(self):
        """get_full_messages should preserve messages without full_content."""
        model = Mock()
        env = Mock()
        
        agent = MiniAgent(model, env)
        
        agent.add_message("user", "normal message")
        agent.add_message("assistant", "response")
        
        full_messages = agent.get_full_messages()
        
        assert len(full_messages) == 2
        assert full_messages[0]["content"] == "normal message"
        assert full_messages[1]["content"] == "response"
    
    def test_count_tokens(self):
        """Token counting should work correctly."""
        model = Mock()
        env = Mock()
        
        agent = MiniAgent(model, env)
        
        text = "a" * 400  # 400 chars
        tokens = agent.count_tokens(text)
        
        assert tokens == 100  # 400 / 4 = 100 tokens
    
    def test_mixed_messages_full_messages(self):
        """Test get_full_messages with mixed message types."""
        model = Mock()
        env = Mock()
        
        agent = MiniAgent(model, env)
        
        agent.add_message("system", "System prompt")
        agent.add_message("user", "Task description")
        agent.add_message("assistant", "Action")
        agent.add_message("user", "Summary", full_content="Long observation")
        agent.add_message("assistant", "Next action")
        
        full_messages = agent.get_full_messages()
        
        assert len(full_messages) == 5
        assert full_messages[0]["content"] == "System prompt"
        assert full_messages[1]["content"] == "Task description"
        assert full_messages[2]["content"] == "Action"
        assert full_messages[3]["content"] == "Long observation"
        assert full_messages[4]["content"] == "Next action"
        
        # Ensure no full_content keys remain
        for msg in full_messages:
            assert "full_content" not in msg


class TestHistorySummarization:
    """Test history summarization functionality."""
    
    def test_no_summarization_when_disabled(self):
        """History should not be summarized when max_context_tokens is 0."""
        model = Mock()
        model.get_template_vars = Mock(return_value={})
        env = Mock()
        env.get_template_vars = Mock(return_value={})
        
        agent = MiniAgent(model, env, max_context_tokens=0)
        
        for i in range(10):
            agent.add_message("user", f"Message {i}" * 100)
        
        assert len(agent.messages) == 10
        assert not agent._history_summarized
    
    def test_summarization_triggered_when_context_too_long(self):
        """History should be summarized when context exceeds max_context_tokens."""
        model = Mock()
        model.query = Mock(return_value={"content": "Summary of previous conversation."})
        model.get_template_vars = Mock(return_value={})
        model.n_calls = 0
        model.cost = 0
        env = Mock()
        env.get_template_vars = Mock(return_value={})
        
        agent = MiniAgent(model, env, max_context_tokens=100, keep_recent_messages=2)
        
        agent.add_message("system", "System prompt")
        for i in range(6):
            agent.add_message("user" if i % 2 == 0 else "assistant", "x" * 100)
        
        agent._check_and_summarize_history()
        
        assert agent._history_summarized
        assert model.query.called
        # Should have: system + summary + 2 recent messages = 4
        assert len(agent.messages) == 4
    
    def test_full_messages_preserved_after_summarization(self):
        """Full messages should contain all original messages after summarization."""
        model = Mock()
        model.query = Mock(return_value={"content": "Summary"})
        model.get_template_vars = Mock(return_value={})
        model.n_calls = 0
        model.cost = 0
        env = Mock()
        env.get_template_vars = Mock(return_value={})
        
        agent = MiniAgent(model, env, max_context_tokens=100, keep_recent_messages=2)
        
        agent.add_message("system", "System")
        for i in range(5):
            agent.add_message("user", f"Message{i}" * 50)
        
        original_full_count = len(agent.full_messages)
        agent._check_and_summarize_history()
        
        assert len(agent.full_messages) == original_full_count
        assert len(agent.messages) < original_full_count
    
    def test_all_model_calls_tracked(self):
        """All model calls should be tracked for training data."""
        model = Mock()
        model.query = Mock(return_value={"content": "Response"})
        model.get_template_vars = Mock(return_value={})
        model.n_calls = 0
        model.cost = 0
        env = Mock()
        env.get_template_vars = Mock(return_value={})
        
        agent = MiniAgent(model, env, max_observation_tokens=10)
        
        # Trigger observation reasoning
        agent._reason_about_observation("x" * 500)
        
        assert len(agent.all_model_calls) == 1
        assert agent.all_model_calls[0]["type"] == "observation_reasoning"
    
    def test_get_all_model_calls(self):
        """get_all_model_calls should return a copy of all model calls."""
        model = Mock()
        model.query = Mock(return_value={"content": "Response"})
        model.get_template_vars = Mock(return_value={})
        env = Mock()
        env.get_template_vars = Mock(return_value={})
        
        agent = MiniAgent(model, env)
        agent.all_model_calls.append({"type": "test", "data": "value"})
        
        calls = agent.get_all_model_calls()
        
        assert len(calls) == 1
        assert calls[0]["type"] == "test"
        # Ensure it's a copy
        calls.append({"type": "new"})
        assert len(agent.all_model_calls) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

