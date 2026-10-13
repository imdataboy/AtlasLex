"""Manual Qdrant example, separate from the AtlasLex knowledge base."""

from qdrant_client import QdrantClient
from atlaslex.config import Settings
from qdrant_client.models import FieldCondition, MatchValue, Filter


def main():
    settings = Settings.load()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    search_result = client.query_points(
        collection_name="test_collection",
        query=[0.2, 0.1, 0.9, 0.7],
        query_filter=Filter(must=[FieldCondition(key="city", match=MatchValue(value="London"))]),
        with_payload=True,
        limit=3,
    ).points

    print(search_result)
    client.close()


if __name__ == "__main__":
    main()
