"""Example usage of observation compression in MiniAgent.

This example demonstrates how the new compression feature works:
1. Long observations are automatically compressed
2. Full observations are preserved for training data
"""

from minisweagent.agents.mini import MiniAgent, AgentConfig
from minisweagent.models.litellm_amd_model import LiteLLMAMDModel
from minisweagent.environments.docker_remote import RemoteDockerEnvironment


def example_with_compression():
    """Example showing automatic observation compression."""
    
    # Initialize model
    model = LiteLLMAMDModel(
        model_name="gpt-5.1-codex",
        api_key="your-api-key",
        temperature=0.7,
        max_tokens=4000
    )
    
    # Initialize environment
    env = RemoteDockerEnvironment(
        docker_server_url="10.67.77.184:9527",
        container_name="test-container"
    )
    
    # Initialize agent with compression enabled
    agent = MiniAgent(
        model=model,
        env=env,
        max_observation_tokens=1000,  # Compress if > 1000 tokens
    )
    
    # Run agent on task
    problem = "Optimize the GPU kernel implementation..."
    exit_status, result = agent.run(problem)
    
    # Get messages for API calls (with compressed observations)
    context_messages = agent.messages
    print(f"Messages in context: {len(context_messages)}")
    for msg in context_messages:
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        has_full = "✓ has full_content" if "full_content" in msg else ""
        print(f"  {msg['role']}: {content_preview} {has_full}")
    
    # Get full messages for training data (with original observations)
    training_messages = agent.get_full_messages()
    print(f"\nFull messages for training: {len(training_messages)}")
    for msg in training_messages:
        content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  {msg['role']}: {content_preview}")
    
    return training_messages


def example_custom_config():
    """Example with custom compression configuration."""
    
    # Custom config with different threshold
    config = AgentConfig(
        max_observation_tokens=500,  # More aggressive compression
        observation_reasoning_template="""
        Analyze this observation and extract:
        1. Critical errors or failures
        2. Success metrics
        3. Required next actions
        
        Observation:
        {{observation}}
        
        Be extremely concise.
        """
    )
    
    # Use config with agent
    # agent = MiniAgent(model, env, config_class=lambda **kw: config)
    print("Custom config created with max_observation_tokens=500")


def analyze_compression_benefits():
    """Demonstrate the benefits of compression."""
    
    # Example: Long compilation output
    long_observation = """
    Observation: <returncode>0</returncode>
    <output>
    [1/245] Building CXX object ...
    [2/245] Building CXX object ...
    ...
    [245/245] Linking CXX executable benchmark_block_adjacent_difference
    </output>
    """ * 20  # Simulate very long output
    
    # Without compression: Would add full 50KB to context
    # With compression: Adds ~200-500 tokens of reasoning
    
    compressed_size = 500  # tokens
    original_size = len(long_observation) // 4  # rough token estimate
    
    print(f"Original observation: ~{original_size} tokens")
    print(f"Compressed reasoning: ~{compressed_size} tokens")
    print(f"Space saved: ~{original_size - compressed_size} tokens ({100*(original_size-compressed_size)/original_size:.1f}%)")
    print(f"\nBenefits:")
    print(f"  - Prevents ContextWindowExceededError")
    print(f"  - Reduces API costs")
    print(f"  - Improves agent reasoning (focused summary vs raw output)")
    print(f"  - Preserves full data for training")


if __name__ == "__main__":
    print("=== Observation Compression Examples ===\n")
    
    print("Example 1: Compression Benefits")
    print("-" * 50)
    analyze_compression_benefits()
    
    print("\n\nExample 2: Custom Configuration")
    print("-" * 50)
    example_custom_config()
    
    print("\n\nExample 3: Usage in training data generation")
    print("-" * 50)
    print("See: src/agent_v2/generate_training_data.py")
    print("Line 157: messages=agent.get_full_messages()")

