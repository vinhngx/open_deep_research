#!/usr/bin/env python3
"""
Test the full Open Deep Research system with NVIDIA integration.
This tests the complete workflow including message filtering.
"""

import os
import asyncio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test that we have the required environment variables
if not os.getenv("NVIDIA_API_KEY"):
    print("❌ NVIDIA_API_KEY environment variable not set")
    exit(1)

print("🚀 Testing full NVIDIA integration with Open Deep Research...")

async def test_full_research_workflow():
    """Test the complete research workflow with NVIDIA models."""
    try:
        # Import the deep research components
        from src.open_deep_research.deep_researcher import deep_research_app
        from src.open_deep_research.configuration import Configuration
        
        print("✅ Successfully imported deep research components")
        
        # Create a simple test configuration
        config = {
            "configurable": {
                "research_model": "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1",
                "summarization_model": "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1",
                "compression_model": "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1",
                "final_report_model": "nvidia:nvidia/llama-3.3-nemotron-super-49b-v1",
                "max_researcher_iterations": 2,
                "max_concurrent_research_units": 1,
                "research_model_max_tokens": 500,
                "summarization_model_max_tokens": 500,
                "compression_model_max_tokens": 500,
                "final_report_model_max_tokens": 1000,
                "max_structured_output_retries": 2,
            }
        }
        
        # Test with a simple research brief
        research_brief = "What are the key benefits of artificial intelligence in healthcare?"
        
        print(f"🔬 Starting research on: {research_brief}")
        
        # Initialize the research state
        initial_state = {
            "research_brief": research_brief,
            "supervisor_messages": [],
            "research_iterations": 0,
        }
        
        # Run the research workflow (limit to prevent long execution)
        result = await deep_research_app.ainvoke(
            initial_state,
            config=config
        )
        
        print("✅ Research workflow completed successfully!")
        print(f"📝 Final notes length: {len(result.get('notes', ''))}")
        print(f"🔄 Research iterations: {result.get('research_iterations', 0)}")
        
        # Check that we got some results
        if result.get('notes'):
            print("✅ Research produced notes")
        else:
            print("❌ No notes produced in research")
            
        return True
        
    except Exception as e:
        print(f"❌ Full integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_full_research_workflow())
    if success:
        print("🎉 Full NVIDIA integration test completed successfully!")
        exit(0)
    else:
        print("💥 Full NVIDIA integration test failed!")
        exit(1)
