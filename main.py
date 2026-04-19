"""SimpleRAG - A lightweight RAG system."""

import sys
from pathlib import Path


def main():
    """Main entry point for SimpleRAG."""
    if len(sys.argv) < 2:
        print("Usage: python main.py <command>")
        print("Commands:")
        print("  ingest          - Ingest documents into vector store")
        print("  chat            - Start interactive chat")
        print("  query <text>    - Search vector store directly")
        print("  help            - Show this help message")
        sys.exit(1)

    command = sys.argv[1]

    # Load configuration (needed for help text and all commands)
    from src.config import load_config
    config = load_config("config.yaml")

    if command == "help":
        print("SimpleRAG - A lightweight RAG system")
        print()
        print("Commands:")
        print("  ingest         - Ingest documents into vector store")
        print("  chat           - Start interactive chat")
        print(f"  query <text>   - Search vector store directly")
        print(f"                   Options: --top N, --no-post-process")
        print()
        print("Configuration:")
        print("  Edit config.yaml to configure LLM, embedding, and storage settings")
        sys.exit(0)

    # Initialize components
    from src.agent import RAGAgent
    from src.embeddings import EmbeddingService
    from src.ingest import IngestPipeline
    from src.vectorstore import VectorStore

    print("Initializing components...")
    embedding_service = EmbeddingService(
        model_path=config.embedding.model_path,
        dimension=config.embedding.dimension,
    )
    vectorstore = VectorStore(
        persist_dir=config.vectorstore.persist_dir,
        embedding_service=embedding_service,
        search_config=config.search,
    )

    if command == "ingest":
        # Run ingestion pipeline
        pipeline = IngestPipeline(config, embedding_service, vectorstore)
        pipeline.run()
        print("Ingestion complete!")

    elif command == "query":
        # Direct vector search
        if len(sys.argv) < 3:
            print("Usage: python main.py query <search_query>")
            sys.exit(1)

        args = sys.argv[2:]
        post_process = True
        top_k = config.search.default_top_k

        # Parse optional flags
        if "--no-post-process" in args:
            post_process = False
            args = [a for a in args if a != "--no-post-process"]
        if "--top" in args:
            idx = args.index("--top")
            if idx + 1 < len(args):
                try:
                    top_k = int(args[idx + 1])
                except ValueError:
                    print("Error: --top must be followed by a number")
                    sys.exit(1)
            args = [a for a in args if a != "--top" and a != args[idx + 1]]

        query_text = " ".join(args) if args else ""
        if not query_text:
            print("Usage: python main.py query <search_query>")
            sys.exit(1)

        results = vectorstore.search(query_text, top_k=top_k, post_process=post_process)

        if not results:
            print("No results found.")
        else:
            print(f"Found {len(results)} results (sorted by relevance):\n")
            for i, r in enumerate(results, 1):
                source = r.metadata.get("source", "unknown")
                print(f"[{i}] (distance: {r.distance:.4f})")
                print(f"    Source: {source}")
                print(f"    Content: {r.content[:200]}{'...' if len(r.content) > 200 else ''}")
                print()

    elif command == "chat":
        # Start chat mode
        agent = RAGAgent(config, vectorstore)

        print("RAG Chat started! (type 'exit' to quit, 'clear' to clear history)")
        print("=" * 50)

        history = []
        while True:
            try:
                query = input("\n> ").strip()
                if not query:
                    continue
                if query.lower() == "exit":
                    print("Goodbye!")
                    break
                if query.lower() == "clear":
                    history = []
                    print("History cleared.")
                    continue

                response = agent.chat(query, history)
                print(f"\n{response}")
                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": response})

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")

    else:
        print(f"Unknown command: {command}")
        print("Run 'python main.py help' for usage information")
        sys.exit(1)


if __name__ == "__main__":
    main()
