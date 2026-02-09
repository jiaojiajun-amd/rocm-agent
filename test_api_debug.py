#!/usr/bin/env python3
"""Debug script to test AMD LLM API connectivity"""

import litellm
import os

# Test configuration
api_key = "c1f7f3ee59064fc0a5fad8c2586f1bd9"
api_base = "https://llm-api.amd.com/azure/engines/gpt-5"
model_name = "gpt-5"

print("=" * 60)
print("Testing AMD LLM API Connection")
print("=" * 60)
print(f"API Base: {api_base}")
print(f"Model: {model_name}")
print(f"API Key: {api_key[:20]}...")
print()

# Enable verbose logging
litellm.set_verbose = True

# Test 1: Basic call with reasoning_effort
print("\n[Test 1] Testing with reasoning_effort parameter...")
try:
    response = litellm.completion(
        model=f"openai/{model_name}",
        messages=[{"role": "user", "content": "Say 'test' only"}],
        api_base=api_base,
        api_key=api_key,
        reasoning_effort="high",
        max_tokens=10,
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        }
    )
    print("✓ Test 1 PASSED")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Test 1 FAILED: {type(e).__name__}: {e}")

# Test 2: Without reasoning_effort parameter
print("\n[Test 2] Testing WITHOUT reasoning_effort parameter...")
try:
    response = litellm.completion(
        model=f"openai/{model_name}",
        messages=[{"role": "user", "content": "Say 'test' only"}],
        api_base=api_base,
        api_key=api_key,
        max_tokens=10,
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        }
    )
    print("✓ Test 2 PASSED")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Test 2 FAILED: {type(e).__name__}: {e}")

# Test 3: Different model format (without openai/ prefix)
print("\n[Test 3] Testing with different model format...")
try:
    response = litellm.completion(
        model=model_name,  # Without openai/ prefix
        messages=[{"role": "user", "content": "Say 'test' only"}],
        api_base=api_base,
        api_key=api_key,
        max_tokens=10,
        extra_headers={
            'Ocp-Apim-Subscription-Key': api_key
        }
    )
    print("✓ Test 3 PASSED")
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"✗ Test 3 FAILED: {type(e).__name__}: {e}")

# Test 4: Check if API base is correct by trying different paths
print("\n[Test 4] Testing alternative API paths...")
alternative_bases = [
    "https://llm-api.amd.com/v1",
    "https://llm-api.amd.com/azure",
    "https://llm-api.amd.com",
]

for alt_base in alternative_bases:
    try:
        print(f"  Trying: {alt_base}")
        response = litellm.completion(
            model=f"openai/{model_name}",
            messages=[{"role": "user", "content": "test"}],
            api_base=alt_base,
            api_key=api_key,
            max_tokens=10,
            extra_headers={'Ocp-Apim-Subscription-Key': api_key}
        )
        print(f"  ✓ SUCCESS with {alt_base}")
        print(f"  Response: {response.choices[0].message.content}")
        break
    except Exception as e:
        print(f"  ✗ Failed: {type(e).__name__}")

print("\n" + "=" * 60)
print("Debug test completed")
print("=" * 60)

