import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

print("✓ All imports successful")

documents = [
    {
        "title": "Aspirin for Cardiovascular Disease Prevention",
        "source": "Journal of Cardiology 2024",
        "content": """
        Aspirin is a nonsteroidal anti-inflammatory drug (NSAID) commonly used 
        for pain relief, fever reduction, and inflammation management. Recent 
        studies show that low-dose aspirin (75-100 mg daily) may reduce 
        cardiovascular events in high-risk patients.
        
        Efficacy: 85% reduction in secondary heart attacks when used appropriately.
        Side effects: GI upset (20%), increased bleeding risk (5%), rare severe allergic reactions.
        Contraindications: Active bleeding, severe liver disease, aspirin allergy.
        
        When combined with other medications, particularly other NSAIDs, bleeding 
        risk increases significantly. Always consult with a healthcare provider 
        before starting aspirin therapy.
        """
    },
    {
        "title": "Ibuprofen Drug Interactions and Safety",
        "source": "FDA Safety Update 2024",
        "content": """
        Ibuprofen is an NSAID used for moderate pain and inflammation. It works 
        by inhibiting prostaglandin synthesis. Common dosing is 200-400 mg 
        every 4-6 hours.
        
        Serious interactions: NEVER combine with aspirin (risk of severe GI bleeding),
        warfarin (increased bleeding), or other NSAIDs.
        
        Renal concerns: Ibuprofen can impair kidney function, especially in elderly 
        patients or those with existing renal disease. Baseline renal function 
        testing recommended.
        
        The FDA has strengthened warnings about cardiovascular and gastrointestinal risks.
        Long-term use should be avoided unless under medical supervision.
        """
    },
    {
        "title": "Pain Management Alternatives to NSAIDs",
        "source": "Pain Management Journal 2024",
        "content": """
        Given the risks associated with NSAIDs, alternative pain management 
        strategies are increasingly recommended. Options include:
        
        1. Acetaminophen: Safer for most patients but hepatotoxic at high doses.
        2. Topical NSAIDs: Lower systemic absorption reduces side effects.
        3. Physical therapy: Effective for chronic pain without medication risks.
        4. Newer agents: COX-2 selective inhibitors have improved safety profiles.
        5. Multimodal approach: Combining multiple strategies for synergistic effects.
        
        Evidence suggests that multimodal pain management reduces NSAID 
        dependence by 60%.
        """
    },
    {
        "title": "Emergency Management of NSAID Overdose",
        "source": "Toxicology Review 2024",
        "content": """
        NSAID overdose or toxicity presents with gastrointestinal bleeding, renal 
        dysfunction, and in severe cases, metabolic acidosis and CNS effects.
        
        Treatment: Supportive care, GI decontamination (within 4 hours), monitoring 
        of electrolytes and renal function. Antacids or H2 blockers may reduce 
        GI symptoms.
        
        For severe cases: Hospitalization required. Hemodialysis may be needed 
        for certain NSAIDs with prolonged half-lives.
        
        Prognosis: Most patients recover fully with early intervention. Mortality 
        is rare unless accompanied by other overdoses or severe comorbidities.
        """
    }
]

print(f"✓ Loaded {len(documents)} documents")

# Initialize embedding model
embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
print("✓ Embedding model loaded")

chroma_client = chromadb.EphemeralClient()

collection = chroma_client.create_collection(
    name="medical_documents",
    metadata={"hnsw:space": "cosine"}
)
print("✓ Chroma collection created")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=['\n\n', '.', ' ']
)

chunk_counter = 0
for doc in documents:
    chunks = text_splitter.split_text(doc["content"])
    print(f"  '{doc['title']}' → {len(chunks)} chunks")
    
    for chunk_idx, chunk in enumerate(chunks):
        embedding = embedding_model.encode([chunk])[0].tolist()
        
        collection.add(
            ids=[f"doc_{chunk_counter}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": chunk_idx
            }]
        )
        
        chunk_counter += 1

print(f"✓ Total chunks indexed: {chunk_counter}")

@tool
def search_documents(query: str, top_k: int = 3) -> str:
    """Search the medical document database. Use query parameter for your search term and top_k for number of results (default 3)."""
    try:
        query_embedding = embedding_model.encode([query])[0].tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k if top_k > 0 else 3
        )
        
        if not results['documents'] or len(results['documents'][0]) == 0:
            return "No relevant documents found. Try a different query."
        
        formatted_results = []
        for i, (document, metadata, distance) in enumerate(
            zip(results['documents'][0], results['metadatas'][0], 
                results['distances'][0])
        ):
            similarity_score = 1 - distance
            formatted_results.append(
                f"\n--- Result {i+1} (Relevance: {similarity_score:.2%}) ---\n"
                f"Source: {metadata['title']} ({metadata['source']})\n"
                f"Content: {document}\n"
            )
        
        return "".join(formatted_results)
        
    except Exception as e:
        return f"Error searching documents: {str(e)}. Please try again."

print("✓ Retrieval tool created")

print("\n" + "="*70)
print("TESTING RETRIEVAL TOOL")
print("="*70)

test_result = search_documents.invoke({"query": "Can I take aspirin and ibuprofen together?", "top_k": 3})
print(test_result)

llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=2048
)

system_prompt = """You are a helpful medical information assistant. You have access to a database of medical documents.

When answering questions:
1. ALWAYS search the document database first using the search_documents tool with a query parameter
2. The search_documents tool takes two parameters: query (required) and top_k (optional, default 3)
3. CITE the sources you find
4. If the documents don't contain relevant information, say so
5. NEVER make up medical information
6. Include relevant information from multiple documents when appropriate
7. For drug interactions or safety concerns, be especially careful and cite sources

Your goal is to provide accurate, well-sourced medical information."""

agent = create_react_agent(
    model=llm,
    tools=[search_documents],
    prompt=system_prompt
)

print("✓ Agent created with retrieval tool")

print("\n" + "="*70)
print("DEMO: AGENTIC RAG IN ACTION")
print("="*70)

demo_queries = [
    "Can I take aspirin and ibuprofen together?",
    "What are the alternatives to NSAIDs for pain management?",
    "How should I handle an NSAID overdose?"
]

for i, query in enumerate(demo_queries, 1):
    print(f"\n[Query {i}] {query}")
    print("Agent thinking...\n")
    
    try:
        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })
        
        answer = result["messages"][-1].content
        print(f"✓ Agent: {answer}\n")
        
    except Exception as e:
        print(f"Error: {e}\n")

def run_medical_rag_agent():
    """Run the agentic RAG system in interactive mode."""
    
    print("\n" + "="*70)
    print("AGENTIC RAG SYSTEM - Medical Information Assistant")
    print("="*70)
    print("\nI have medical documents. Ask me anything about medications.")
    print("\nExamples:")
    print("  - 'What are the side effects of ibuprofen?'")
    print("  - 'Can I take aspirin and ibuprofen together?'")
    print("  - 'What are alternatives to NSAIDs for pain management?'")
    print("  - 'How should I handle an NSAID overdose?'")
    print("\nType 'quit' or 'exit' to leave.")
    print("="*70)
    
    query_count = 0
    
    while True:
        try:
            user_input = input("\n📋 Your question: ").strip()
            
            if not user_input:
                print("⚠️  Please enter a question.")
                continue
            
            if user_input.lower() in ["exit", "quit", "bye", "q"]:
                print(f"\n👋 Thank you for using the Medical Information Assistant!")
                print(f"Total questions answered: {query_count}")
                break
            
            query_count += 1
            print("\n🤔 Searching documents and thinking...")
            
            try:
                result = agent.invoke({
                    "messages": [HumanMessage(content=user_input)]
                })
                
                final_answer = result["messages"][-1].content
                print(f"\n✅ Assistant: {final_answer}")
                
            except Exception as agent_error:
                error_msg = f"Error processing query: {str(agent_error)}"
                print(f"\n❌ {error_msg}")
                print("Tip: Try rephrasing your question or use simpler language.")
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Interrupted by user.")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    run_medical_rag_agent()

# DOMAIN = "Climate Science"

# your_documents = [
#     {
#         "title": "Your Document 1",
#         "source": "Journal/Source Name",
#         "content": """Your 300+ word content here"""
#     },
#     {
#         "title": "Your Document 2",
#         "source": "Journal/Source Name",
#         "content": """Your second document"""
#     },
#     {
#         "title": "Your Document 3",
#         "source": "Journal/Source Name",
#         "content": """Your third document"""
#     }
# ]

# embedding_model_ex1 = SentenceTransformer('BAAI/bge-small-en-v1.5')
# chroma_client_ex1 = chromadb.EphemeralClient()

# collection_ex1 = chroma_client_ex1.create_collection(
#     name=f"{DOMAIN.lower().replace(' ', '_')}_documents",
#     metadata={"hnsw:space": "cosine"}
# )

# text_splitter_ex1 = RecursiveCharacterTextSplitter(
#     chunk_size=300,
#     chunk_overlap=50,
#     separators=['\n\n', '.', ' ']
# )

# chunk_counter = 0
# for doc in your_documents:
#     chunks = text_splitter_ex1.split_text(doc["content"])
#     for chunk in chunks:
#         embedding = embedding_model_ex1.encode([chunk])[0].tolist()
#         collection_ex1.add(
#             ids=[f"chunk_{chunk_counter}"],
#             documents=[chunk],
#             embeddings=[embedding],
#             metadatas=[{
#                 "source": doc["source"],
#                 "title": doc["title"]
#             }]
#         )
#         chunk_counter += 1

# print(f"✓ Indexed {chunk_counter} chunks")

# @tool
# def search_my_documents(query: str, top_k: int = 3) -> str:
#     """Search your domain-specific documents."""
#     try:
#         query_embedding = embedding_model_ex1.encode([query])[0].tolist()
#         results = collection_ex1.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
        
#         if not results['documents'] or len(results['documents'][0]) == 0:
#             return f"No results for '{query}'"
        
#         formatted = []
#         for i, (doc, meta, dist) in enumerate(
#             zip(results['documents'][0], results['metadatas'][0], 
#                 results['distances'][0])
#         ):
#             score = 1 - dist
#             formatted.append(
#                 f"\n[Result {i+1} | Relevance: {score:.1%}]\n"
#                 f"From: {meta['title']}\n"
#                 f"Content: {doc}\n"
#             )
#         return "".join(formatted)
        
#     except Exception as e:
#         return f"Error: {str(e)}"

# my_queries = [
#     "Your question 1",
#     "Your question 2",
#     "Your complex question 3",
#     "Your complex question 4",
#     "Your question 5"
# ]

# print("\n" + "="*70)
# print("EXERCISE 1: TESTING YOUR RETRIEVER")
# print("="*70)

# for i, query in enumerate(my_queries, 1):
#     print(f"\n[Query {i}] {query}")
#     result = search_my_documents(query)
#     print(result)
#     print("-"*70)

# system_prompt_ex2 = f"""
# You are an expert AI assistant specialized in {DOMAIN}.
# You have access to a database of authoritative documents.

# When answering questions:
# 1. ALWAYS search the document database using search_my_documents()
# 2. CITE your sources
# 3. Only answer based on what you find in documents
# 4. If information isn't in the database, say so clearly
# 5. For complex questions, search multiple times with different queries
# 6. Synthesize information across documents when relevant

# Your goal: Provide accurate, well-sourced information about {DOMAIN}.
# """

# agent_ex2 = create_react_agent(
#     model=llm,
#     tools=[search_my_documents],
#     prompt=system_prompt_ex2
# )

# print(f"✓ Agent created for {DOMAIN}")

# def run_domain_agent():
#     print(f"\n{'='*70}")
#     print(f"AGENTIC RAG - {DOMAIN}")
#     print(f"{'='*70}\n")
    
#     query_count = 0
    
#     while True:
#         try:
#             question = input(f"[Query {query_count + 1}] Your question: ").strip()
            
#             if not question:
#                 print("Please ask a question.")
#                 continue
            
#             if question.lower() in ["quit", "exit"]:
#                 print(f"\nAnswered {query_count} questions. Thank you!")
#                 break
            
#             query_count += 1
#             print("\n🔍 Searching documents...\n")
            
#             try:
#                 result = agent_ex2.invoke({
#                     "messages": [HumanMessage(content=question)]
#                 })
                
#                 answer = result["messages"][-1].content
#                 print(f"✓ Assistant: {answer}\n")
                
#             except Exception as e:
#                 print(f"❌ Error: {str(e)}\n")
                
#         except KeyboardInterrupt:
#             print("\n\nSession ended.")
#             break

# run_domain_agent()
