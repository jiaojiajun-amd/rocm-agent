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
        env = Mock()
        
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

