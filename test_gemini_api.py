#!/usr/bin/env python3
"""
Test Gemini API connectivity and quota
"""

import os
from google import genai

# Get API key from environment
api_key = os.getenv('GOOGLE_API_KEY')

if not api_key:
    print("ERROR: GOOGLE_API_KEY not set")
    print("Please set it in your .env file or export it:")
    print("  export GOOGLE_API_KEY='your-api-key-here'")
    exit(1)

print(f"✓ API Key loaded: {api_key[:20]}...")
print()

# Test connection
try:
    client = genai.Client(api_key=api_key)

    # List available models
    print("Available models:")
    models = client.models.list()

    gemini_models = []
    for model in models:
        if 'gemini' in model.name.lower():
            gemini_models.append(model.name)
            print(f"  - {model.name}")

    if not gemini_models:
        print("  No Gemini models found!")
        print()
        print("This might mean:")
        print("1. Your API key doesn't have access to Gemini models")
        print("2. You need to enable the API in Google Cloud Console")
        exit(1)

    print()

    # Try a simple test with gemini-1.5-flash
    print("Testing gemini-1.5-flash with a simple prompt...")

    test_prompt = "Say 'Hello' in JSON format with a 'message' field."

    response = client.models.generate_content(
        model='gemini-2.5-pro',
        contents=test_prompt,
        config=genai.types.GenerateContentConfig(
            temperature=0.3,
            response_mime_type="application/json"
        )
    )

    print(f"✓ Response received: {response.text[:100]}...")
    print()
    print(f"Token usage:")
    print(f"  - Input tokens: {response.usage_metadata.prompt_token_count}")
    print(f"  - Output tokens: {response.usage_metadata.candidates_token_count}")
    print(f"  - Total tokens: {response.usage_metadata.total_token_count}")
    print()
    print("✓ API is working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    print()
    print("Possible solutions:")
    print("1. Check your API key at: https://aistudio.google.com/app/apikey")
    print("2. Enable Generative Language API in Google Cloud Console")
    print("3. Check your quota at: https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas")
    exit(1)
