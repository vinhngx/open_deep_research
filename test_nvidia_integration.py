#!/usr/bin/env python3
"""
Test script for NVIDIA NIM integration in Open Deep Research.

This script tests the NVIDIA model configuration and integration
to ensure it works properly with the research system.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test basic imports
def test_imports():
    """Test if we can import the functions without dependency issues."""
    try:
        from open_deep_research.utils import get_model_config_for_nvidia, get_api_key_for_model
        print("✅ Successfully imported utility functions")
        return True
    except Exception as e:
        print(f"❌ Failed to import utility functions: {e}")
        return False

def test_nvidia_config():
    """Test NVIDIA model configuration."""
    # Test imports first
    if not test_imports():
        return
        
    from open_deep_research.utils import get_model_config_for_nvidia
    
    print("Testing NVIDIA model configuration...")
    
    # Test NVIDIA model configuration
    model_name = "nvidia:llama-3.3-nemotron-super-49b-v1"
    api_key = "test-key"
    
    config = get_model_config_for_nvidia(
        model_name=model_name,
        api_key=api_key,
        max_tokens=100
    )
    
    print(f"Input model name: {model_name}")
    print(f"Generated config: {config}")
    
    # Verify the configuration
    expected = {
        "model": "llama-3.3-nemotron-super-49b-v1",  # Should strip nvidia: prefix
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": api_key,
        "max_tokens": 100,
        "model_provider": "nvidia"
    }
    
    assert config == expected, f"Config mismatch: {config} != {expected}"
    print("✅ NVIDIA configuration test passed!")

def test_standard_model_config():
    """Test standard model configuration (non-NVIDIA)."""
    from open_deep_research.utils import get_model_config_for_nvidia
    
    print("Testing standard model configuration...")
    
    model_name = "openai:gpt-4"
    api_key = "test-openai-key"
    
    config = get_model_config_for_nvidia(
        model_name=model_name,
        api_key=api_key,
        max_tokens=100
    )
    
    print(f"Input model name: {model_name}")
    print(f"Generated config: {config}")
    
    # Verify the configuration
    expected = {
        "model": model_name,
        "api_key": api_key,
        "max_tokens": 100
    }
    
    assert config == expected, f"Config mismatch: {config} != {expected}"
    print("✅ Standard model configuration test passed!")

def test_api_key_detection():
    """Test API key detection for NVIDIA models."""
    from open_deep_research.utils import get_api_key_for_model
    
    print("Testing NVIDIA API key detection...")
    
    # Mock config for testing
    class MockConfig:
        def get(self, key, default=None):
            return {"configurable": {"apiKeys": {"NVIDIA_API_KEY": "test-nvidia-key"}}}
    
    config = MockConfig()
    
    # Test with environment variable disabled
    os.environ["GET_API_KEYS_FROM_CONFIG"] = "false"
    os.environ["NVIDIA_API_KEY"] = "env-nvidia-key"
    
    api_key = get_api_key_for_model("nvidia:test-model", config)
    assert api_key == "env-nvidia-key", f"Expected env key, got {api_key}"
    
    print("✅ API key detection test passed!")

def test_nvidia_model_live():
    """Test NVIDIA model with actual API call (requires API key)."""    
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    print(f"NVIDIA_API_KEY loaded: {'Yes' if nvidia_api_key else 'No'}")
    print(f"API key (first 10 chars): {nvidia_api_key[:10] if nvidia_api_key else 'None'}...")
    nvidia_api_key="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"
    if not nvidia_api_key:
        print("⚠️  Skipping live test - NVIDIA_API_KEY not set")
        return
    
    print("Testing live NVIDIA API call...")
    
    try:
        # Test with simple requests to avoid dependency issues
        import requests
        
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {nvidia_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
            "messages": [{"role": "user", "content": "Hello! who are you and who made you? what's the humanity future in the face of things like you?"}],
            "temperature": 0.1,
            "max_tokens": 5000
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            message = result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')
            print(f"✅ Live test successful! Response: {message}")
        else:
            print(f"⚠️  API returned status {response.status_code}: {response.text}")
            
    except Exception as e:
        print(f"⚠️  Live test failed: {e}")
        print("This might be due to invalid API key or model availability")

if __name__ == "__main__":
    print("🚀 Starting NVIDIA NIM integration tests...\n")
    
    try:
        test_nvidia_config()
        print()
        
        test_standard_model_config()
        print()
        
        test_api_key_detection()
        print()
        
        test_nvidia_model_live()
        print()
        
        print("🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
