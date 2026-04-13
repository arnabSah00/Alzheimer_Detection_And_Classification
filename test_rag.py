from rag.pipeline import build_rag, ask_rag


# BUILD DB (FIRST TIME ONLY)
print("Building RAG...")

build_rag()

print("RAG Ready\n")


# TEST QUERY
stage = 'early'
query = 'What should be done now? Give medicinal treatment and non-medicinal treatment.'

result = ask_rag(stage, query)

print("\nQUESTION:")
print(query)

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")
for s in result["sources"]:
    print(s)