import anthropic

# Ensure client can be constructed without calling the API
client = anthropic.Anthropic(api_key="test_key")
print("Anthropic client constructed")