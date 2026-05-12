#!/usr/bin/env python3
"""
Скрипт для просмотра содержимого базы данных Qdrant.
Использование:
    python inspect_db.py                    # показать все точки
    python inspect_db.py --source <id>      # фильтр по source_id
    python inspect_db.py --stats            # только статистика
    python inspect_db.py --limit 5          # ограничить количество
"""

import argparse
import json
from qdrant_client import QdrantClient
from qdrant_client.http import models


def get_client() -> QdrantClient:
    """Создает подключение к Qdrant."""
    return QdrantClient(host="localhost", port=6333)


def show_stats(client: QdrantClient, collection_name: str = "audio_rag_chunks"):
    """Показывает статистику коллекции."""
    print("=" * 60)
    print("📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
    print("=" * 60)

    # Получаем информацию о коллекции
    collection_info = client.get_collection(collection_name)

    print(f"Коллекция: {collection_name}")
    print(f"Статус: {collection_info.status}")
    print(f"Точек: {collection_info.points_count}")
    print(f"Индексировано векторов: {collection_info.indexed_vectors_count}")
    print(f"Размер вектора: {collection_info.config.params.vectors.size}")
    print(f"Метрика: {collection_info.config.params.vectors.distance}")
    print()

    # Получаем список источников
    sources = set()
    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    for point in points:
        if point.payload and "source_id" in point.payload:
            sources.add(point.payload["source_id"])

    if sources:
        print(f"Источники ({len(sources)}):")
        for source in sorted(sources):
            count = sum(1 for p in points if p.payload.get("source_id") == source)
            print(f"  • {source}: {count} чанков")

    print()


def show_points(
    client: QdrantClient,
    collection_name: str = "audio_rag_chunks",
    source_filter: str = None,
    limit: int = 10,
    show_vector: bool = False
):
    """Показывает точки в коллекции."""
    print("=" * 60)
    print("📝 ТОЧКИ В БАЗЕ ДАННЫХ")
    print("=" * 60)

    # Фильтр по source_id
    query_filter = None
    if source_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="source_id",
                    match=models.MatchValue(value=source_filter)
                )
            ]
        )

    # Получаем точки
    points, next_offset = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=show_vector,
        query_filter=query_filter
    )

    if not points:
        print("❌ Точки не найдены")
        if source_filter:
            print(f"   Фильтр: source_id = '{source_filter}'")
        return

    print(f"Найдено точек: {len(points)}")
    if source_filter:
        print(f"Фильтр: source_id = '{source_filter}'")
    print()

    for i, point in enumerate(points, 1):
        print(f"Точка #{i}")
        print(f"  ID: {point.id}")

        if point.payload:
            # Source ID
            source_id = point.payload.get("source_id", "N/A")
            print(f"  Source: {source_id}")

            # Chunk ID
            chunk_id = point.payload.get("chunk_id", "N/A")
            print(f"  Chunk ID: {chunk_id}")

            # Текст
            text = point.payload.get("text", "")
            if text:
                print(f"  Текст: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"  Длина текста: {len(text)} символов")

            # Временные метки
            start = point.payload.get("start_offset")
            end = point.payload.get("end_offset")
            if start is not None and end is not None:
                print(f"  Временные метки: [{start}s : {end}s]")

            # Metadata
            metadata = point.payload.get("metadata", {})
            if metadata:
                print(f"  Metadata: {json.dumps(metadata, ensure_ascii=False)}")

        # Вектор (если запрошен)
        if show_vector and point.vector:
            print(f"  Вектор: {len(point.vector)} dim, первые 5: {point.vector[:5]}")

        print()

    if next_offset:
        print(f"⚠️  Есть еще точки (показано {limit} из общего количества)")
        print(f"   Используйте --limit {limit * 2} для показа большего количества")


def list_sources(client: QdrantClient, collection_name: str = "audio_rag_chunks"):
    """Показывает список всех источников."""
    print("=" * 60)
    print("📚 ИСТОЧНИКИ")
    print("=" * 60)

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1000,
        with_payload=True,
        with_vectors=False
    )

    sources = {}
    for point in points:
        if point.payload and "source_id" in point.payload:
            source_id = point.payload["source_id"]
            sources[source_id] = sources.get(source_id, 0) + 1

    if not sources:
        print("❌ Источники не найдены")
        return

    for source_id, count in sorted(sources.items()):
        print(f"  • {source_id}: {count} чанков")

    print(f"\nВсего источников: {len(sources)}")
    print(f"Всего чанков: {sum(sources.values())}")


def main():
    parser = argparse.ArgumentParser(
        description="Просмотр содержимого базы данных Qdrant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python inspect_db.py                      # показать все точки
  python inspect_db.py --stats              # только статистика
  python inspect_db.py --source my-podcast  # фильтр по источнику
  python inspect_db.py --limit 5            # показать 5 точек
  python inspect_db.py --vector             # показать векторы
  python inspect_db.py --sources            # список источников
        """
    )

    parser.add_argument(
        "--collection",
        default="audio_rag_chunks",
        help="Имя коллекции (default: audio_rag_chunks)"
    )
    parser.add_argument(
        "--source",
        help="Фильтр по source_id"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Максимальное количество точек для показа (default: 10)"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Показать только статистику"
    )
    parser.add_argument(
        "--sources",
        action="store_true",
        help="Показать только список источников"
    )
    parser.add_argument(
        "--vector",
        action="store_true",
        help="Показать векторы (первые 5 значений)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )

    args = parser.parse_args()

    # Создаем клиент
    client = QdrantClient(host=args.host, port=args.port)

    # Проверяем подключение
    try:
        client.get_collection(args.collection)
    except Exception as e:
        print(f"❌ Ошибка подключения к Qdrant: {e}")
        print(f"   Убедитесь, что Qdrant запущен: docker-compose up -d qdrant")
        return

    # Показываем информацию
    if args.sources:
        list_sources(client, args.collection)
    elif args.stats:
        show_stats(client, args.collection)
    else:
        show_stats(client, args.collection)
        show_points(
            client,
            collection_name=args.collection,
            source_filter=args.source,
            limit=args.limit,
            show_vector=args.vector
        )


if __name__ == "__main__":
    main()
```

---

Я создал скрипт `inspect_db.py` для удобного просмотра содержимого базы данных Qdrant.

## Возможности скрипта:

**Показать все данные:**
```bash
python inspect_db.py
```

**Только статистика:**
```bash
python inspect_db.py --stats
```

**Фильтр по источнику:**
```bash
python inspect_db.py --source test-podcast
```

**Список источников:**
```bash
python inspect_db.py --sources
```

**Ограничить количество:**
```bash
python inspect_db.py --limit 5
```

**С векторами:**
```bash
python inspect_db.py --vector
```

---

## 🌐 Альтернативные способы:

### 3. **Qdrant Web UI**
Откройте: http://localhost:6333/dashboard

### 4. **REST API напрямую:**
```bash
# Все точки
curl -X POST http://localhost:6333/collections/audio_rag_chunks/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "with_payload": true}'

# Поиск по фильтру
curl -X POST http://localhost:6333/collections/audio_rag_chunks/points/scroll \
  -H "Content-Type: application/json" \
  -d '{"limit": 10, "with_payload": true, "filter": {"must": [{"key": "source_id", "match": {"value": "test-podcast"}}]}}'
```

---

## 📌 Основные поля в payload:

| Поле | Описание | Пример |
|------|----------|--------|
| `source_id` | ID источника | "test-podcast" |
| `chunk_id` | ID чанка | "test-podcast-f77d4a6e" |
| `text` | Текст чанка | "На сегодняшний день..." |
| `start_offset` | Начало в секундах | 0 |
| `end_offset` | Конец в секундах | 18 |
| `metadata` | Метаданные | {"ingest_mode": "transcript"} |

---

⚠️ **Обратите внимание**: В базе 2 точки с одинаковым текстом - это дубликаты. Возможно, аудио было инжестировано дважды. Если нужно удалить дубликаты или очистить базу, используйте:

```bash
# Очистить базу (осторожно!)
python main.py reset-store
