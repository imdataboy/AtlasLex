"""Interactive terminal interface. All application logic lives in the package."""

import getpass
from pathlib import Path

from .config import EMBED_MODEL, RERANK_MODEL, Settings
from .generation import NO_EVIDENCE, Gemini, GeminiError, rag_query
from .indexing import IndexErrorBase, IndexManager, discover_pdfs
from .models import model_cache_status
from .retrieval import Retriever


def banner(title):
    print("\n" + "=" * 44)
    print("\n".join(line.center(44) for line in title.splitlines()))
    print("=" * 44)


def pause():
    input("\nPress Enter to return to the main menu.")


def show_status(manager, *, models=True):
    settings = manager.settings
    print(f"Gemini API Key: {'Configured' if settings.api_key else 'Not configured'}")
    print(f"Collection: {settings.collection}")
    try:
        _, manifest, chunks = manager.load()
        print("Qdrant: Connected")
        print(f"Documents indexed: {len(manifest['documents'])}")
        print(
            f"Articles extracted: {sum(d['article_count'] for d in manifest['documents'].values())}"
        )
        print(f"Chunks indexed: {len(chunks)}")
        # Actually construct sparse index before claiming it is ready.
        from .retrieval import build_bm25

        print(f"BM25 index: {'Ready' if build_bm25(chunks) is not None else 'Not built'}")
    except IndexErrorBase as error:
        print(f"Knowledge base: {error}")
    except Exception:
        print("Qdrant: Unavailable. Start it with: docker compose up -d")
    if models:
        print(f"Embedding model: {model_cache_status(EMBED_MODEL)}")
        print(f"Reranking model: {model_cache_status(RERANK_MODEL)}")


def configure_key(settings):
    banner("GEMINI CONFIGURATION")
    # getpass falls back to echoed stdin when no terminal is present. Reject
    # that fallback so this option never exposes a credential while typing.
    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass("Enter your Gemini API key: ")
        settings.save_api_key(key)
        print("API key saved successfully.")
    except getpass.GetPassWarning:
        print("Hidden input is unavailable. Run AtlasLex in an interactive terminal.")
    except ValueError as error:
        print(error)
    except OSError:
        print("Could not save .env. Check directory permissions.")


def indexing_menu(manager):
    settings = manager.settings
    while True:
        banner("DOCUMENT INDEXING")
        print(f"Place your legal PDF files inside:\n\n{settings.raw_dir}\n")
        print(
            "[1] Index all PDFs in data/raw/\n[2] Enter a PDF file or folder path\n"
            "[3] Rebuild existing index\n[4] Return to main menu"
        )
        selection = input("\nSelect an option: ").strip()
        if selection == "4":
            return
        if selection not in {"1", "2", "3"}:
            print("Please select 1, 2, 3 or 4.")
            continue
        try:
            rebuild = selection == "3"
            if rebuild:
                print(
                    "Rebuilding replaces the current index from its source PDFs. "
                    "Missing source files will be removed. The old index stays active until the new one is ready."
                )
                if input("Continue? [y/N]: ").strip().lower() not in {"y", "yes"}:
                    continue
                paths = manager.source_paths()
            else:
                path = settings.raw_dir
                if selection == "2":
                    entered = input("PDF file or folder path: ").strip().strip('"').strip("'")
                    if not entered:
                        print("Enter a PDF file or folder path.")
                        continue
                    path = Path(entered)
                paths = discover_pdfs(path)
            print(f"Found {len(paths)} PDF documents.")
            if not paths:
                print("No PDF files found. Add documents and try again.")
                continue
            result = manager.index(paths, rebuild=rebuild)
            print("\nIndexing completed successfully.")
            print(
                f"Documents processed: {result['processed']}\nUnchanged documents: {result['skipped']}\n"
                f"Documents indexed: {result['documents']}\nArticles extracted: {result['articles']}\n"
                f"Chunks indexed: {result['chunks']}\nVector database: Qdrant\nSparse index: BM25"
            )
            for warning in result["warnings"]:
                print(warning)
        except (ValueError, IndexErrorBase) as error:
            print(error)
        except Exception:
            print(
                "Indexing could not complete. Check Qdrant, model downloads and disk space. "
                "Check System Status before retrying."
            )
        input("\nPress Enter to continue.")


def chat(manager, *, gemini_factory=Gemini, retriever_factory=Retriever):
    if not manager.settings.api_key:
        print("No Gemini API key configured. Please select Option 1 from the main menu.")
        return
    try:
        retriever = retriever_factory(manager)
    except IndexErrorBase as error:
        print(error)
        return
    except Exception:
        print("Qdrant is unavailable. Start it with: docker compose up -d")
        return
    if not retriever.chunks:
        print(
            "No legal documents are indexed.\n\nPlease select Option 2 to add and index legal PDFs."
        )
        return
    gemini = gemini_factory(manager.settings)
    banner("ATLASLEX CHAT")
    documents = sorted({c["source_document"] for c in retriever.chunks if c.get("source_document")})
    if documents:
        print(f"Indexed documents: {', '.join(documents)}")
    print(
        "Ask a question about your indexed legal documents.\nType 'exit' to return to the main menu."
    )
    try:
        while True:
            query = input("\nYou: ").strip()
            if query.lower() == "exit":
                print("Returning to main menu...")
                return
            if not query:
                continue
            try:
                result = rag_query(query, retriever, gemini)
                print(f"\nAtlasLex:\n{result['answer']}")
                if result["answer"] == NO_EVIDENCE:
                    print(
                        "Try a more specific question, or add a relevant legal PDF "
                        "using Option 2 from the main menu."
                    )
                if result["sources"]:
                    print("\nSources:")
                    for source in result["sources"]:
                        article = (
                            f"Article {source['article_number']}"
                            if source.get("article_number")
                            else "Extrait"
                        )
                        pages = ", ".join(map(str, source["page_numbers"]))
                        print(f"- {source['source_document']} — {article} — Pages {pages}")
            except GeminiError as error:
                print(error)
            except IndexErrorBase as error:
                print(error)
            except Exception:
                print(
                    "Unable to complete retrieval. Check Qdrant and model availability; try again."
                )
    finally:
        gemini.close()


def run(settings, manager=None):
    manager = manager or IndexManager(settings)
    try:
        banner("ATLASLEX\nMoroccan Commercial Law RAG")
        show_status(manager, models=False)
        while True:
            print(
                "\nMAIN MENU\n\n[1] Set Gemini API Key\n[2] Add / Index Legal PDFs\n"
                "[3] Start Chatting\n[4] Check System Status\n[5] Exit"
            )
            selection = input("\nSelect an option: ").strip()
            if selection == "1":
                configure_key(settings)
                pause()
            elif selection == "2":
                indexing_menu(manager)
            elif selection == "3":
                chat(manager)
            elif selection == "4":
                banner("SYSTEM STATUS")
                show_status(manager)
                pause()
            elif selection == "5":
                break
            else:
                print("Please select a number from 1 to 5.")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        manager.close()
    print("Thank you for using AtlasLex.\nGoodbye!")


def main():
    try:
        run(Settings.load())
    except ValueError as error:
        print(f"Configuration error: {error}")
    except Exception:
        print("AtlasLex could not start. Check your configuration and local file permissions.")


if __name__ == "__main__":
    main()
