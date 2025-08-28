#!/usr/bin/env python3
"""
Test the message filtering function directly to ensure it prevents empty content errors.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, '/home/vinhn/deep_research/open_deep_research/src')

def test_message_filtering():
    """Test the filter_empty_messages function directly."""
    try:
        # Import the filter function
        from open_deep_research.utils import filter_empty_messages
        print("✅ Successfully imported filter_empty_messages")
        
        # Test messages with some empty content
        test_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},  # Empty content
            {"role": "user", "content": "   "},  # Whitespace only
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "\n\t"},  # Whitespace/newlines only
            {"role": "assistant", "content": "How can I help?"}
        ]
        
        print(f"📝 Original messages: {len(test_messages)}")
        filtered_messages = filter_empty_messages(test_messages)
        print(f"🔍 Filtered messages: {len(filtered_messages)}")
        
        # Should have removed 3 messages with empty/whitespace content
        expected_count = 4  # system, user "Hello", assistant "Hi there!", assistant "How can I help?"
        
        if len(filtered_messages) == expected_count:
            print("✅ Message filtering test passed!")
            print("📋 Filtered messages:")
            for i, msg in enumerate(filtered_messages):
                print(f"  {i+1}. {msg['role']}: {repr(msg['content'][:50])}")
            return True
        else:
            print(f"❌ Expected {expected_count} messages, got {len(filtered_messages)}")
            return False
            
    except Exception as e:
        print(f"❌ Message filtering test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_nvidia_model_config():
    """Test the NVIDIA model configuration function."""
    try:
        from open_deep_research.utils import get_model_config_for_nvidia
        print("✅ Successfully imported get_model_config_for_nvidia")
        
        config = get_model_config_for_nvidia(
            model_name="nvidia:nvidia/llama-3.3-nemotron-super-49b-v1",
            api_key="test-key",
            max_tokens=100
        )
        
        expected_keys = {"model", "base_url", "api_key", "max_tokens", "model_provider"}
        if all(key in config for key in expected_keys):
            print("✅ NVIDIA model config test passed!")
            print(f"🔧 Config: {config}")
            return True
        else:
            print(f"❌ Missing keys in config: {expected_keys - set(config.keys())}")
            return False
            
    except Exception as e:
        print(f"❌ NVIDIA model config test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Testing NVIDIA integration functions...")
    
    message_test_passed = test_message_filtering()
    config_test_passed = test_nvidia_model_config()
    
    if message_test_passed and config_test_passed:
        print("🎉 All function tests passed! NVIDIA integration should work correctly.")
        exit(0)
    else:
        print("💥 Some tests failed!")
        exit(1)
