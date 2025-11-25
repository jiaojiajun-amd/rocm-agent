# Observation Compression for Long Outputs

## Problem

When generating training data for agent continuous pretraining, long command outputs (observations) can exceed the model's context window limit, causing `ContextWindowExceededError`.

## Solution

The `MiniAgent` now automatically compresses long observations using a two-stage approach:

1. **Detection**: Check if observation exceeds `max_observation_tokens` (default: 1000 tokens)
2. **Reasoning**: Use the model to analyze and summarize the long observation
3. **Dual Storage**: 
   - Store compressed reasoning in context for subsequent agent steps
   - Store full observation in message metadata for training data

## Architecture

### Message Structure

Messages now support dual content storage:
```python
{
    "role": "user",
    "content": "<observation_summary>...compressed reasoning...</observation_summary>",
    "full_content": "Observation: <very long original output>"
}
```

### Key Methods

- `count_tokens(text)`: Estimate token count for text
- `_reason_about_observation(observation)`: Call model to summarize long observation
- `get_full_messages()`: Retrieve messages with full content for training data

## Configuration

In `mini.yaml`:

```yaml
agent:
  # Maximum tokens before triggering compression
  max_observation_tokens: 1000
  
  # Template for reasoning prompt
  observation_reasoning_template: |
    The following observation is very long. Please analyze it and provide a concise summary...
```

## Usage

### For Agent Execution

No changes needed - compression happens automatically:

```python
agent = MiniAgent(model, env, **config)
exit_status, result = agent.run(problem_statement)
# Observations are automatically compressed if too long
```

### For Training Data Generation

Use `get_full_messages()` to retrieve complete message history:

```python
# During data generation
agent.run(problem)
full_messages = agent.get_full_messages()  # Contains full observations

# Save to training data
training_example = TrainingExample(
    messages=full_messages,  # Full content for CPT
    ...
)
```

## Benefits

1. **Context Window Management**: Prevents context overflow by compressing long outputs
2. **Preserved Information**: Full observations saved for training data completeness
3. **Better Reasoning**: Agent gets focused summaries instead of raw long outputs
4. **Token Efficiency**: Reduces API costs by using compressed context

## Example

Before compression:
```
Observation: [50,000 character compilation output with thousands of lines]
```

After compression (in context):
```
<observation_summary>
The compilation completed successfully for the rocPRIM library. All 245 source files 
were compiled without errors. A few deprecation warnings were noted for C++11 features,
but these don't affect functionality. The benchmark binary was generated successfully.

Next steps: Run the benchmark to measure performance improvements.
</observation_summary>
```

Full observation preserved in `full_content` for training data.

## Token Counting

Current implementation uses a simple heuristic (4 chars ≈ 1 token). For more accurate counting, consider integrating `tiktoken`:

```python
def count_tokens(self, text: str) -> int:
    import tiktoken
    encoding = tiktoken.encoding_for_model("gpt-4")
    return len(encoding.encode(text))
```

## Trade-offs

**Pros:**
- Prevents context window errors
- Better agent reasoning with focused summaries
- Complete data preservation for training

**Cons:**
- Additional model call for reasoning (increases cost/latency)
- Simple token estimation may not be perfectly accurate
- Reasoning quality depends on model capability

## Future Improvements

1. Use proper tokenizer (tiktoken) for accurate token counting
2. Configurable reasoning strategies (extractive vs abstractive)
3. Caching for similar observations
4. Adaptive compression ratios based on task type

