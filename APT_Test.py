# # LLM|1068491247969763|A8H4vjp4lSI879xuNKg_DZJuGDk

# curl -X POST "https://api.llama.com/v1/chat/completions" \
#   -H "Authorization: Bearer LLM|1068491247969763|A8H4vjp4lSI879xuNKg_DZJuGDk" \
#   -H "Content-Type: application/json" \
#   -d '{
#     "model": "Llama-4-Maverick-17B-128E-Instruct-FP8",
#     "messages": [
#       {
#         "role": "system",
#         "content": "You are a friendly assistant."
#       },
#       {
#         "role": "user",
#         "content": "Hello, world!"
#       }
#     ]
#   }'