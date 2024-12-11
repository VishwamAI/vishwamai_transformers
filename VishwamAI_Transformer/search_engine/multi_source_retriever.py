# multi_source_retriever.py

from abc import ABC, abstractmethod
import hashlib
import os
import json
import wikipedia

class Adapter(ABC):
    @abstractmethod
    def retrieve(self, query):
        pass

class WikipediaAdapter(Adapter):
    def retrieve(self, query):
        try:
            page = wikipedia.page(query)
            content = page.content
            return [{'source': 'wikipedia', 'content': content, 'url': page.url}]
        except wikipedia.exceptions.DisambiguationError:
            return []
        except wikipedia.exceptions.PageError:
            return []

class FileSystemAdapter(Adapter):
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def retrieve(self, query):
        results = []
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if query.lower() in content.lower():
                            results.append({
                                'source': 'file',
                                'content': content,
                                'file_path': file_path
                            })
        return results

class JsonDatabaseAdapter(Adapter):
    def __init__(self, db_path):
        with open(db_path, 'r') as f:
            self.db = json.load(f)

    def retrieve(self, query):
        results = []
        for doc in self.db:
            if query.lower() in doc['content'].lower():
                results.append({
                    'source': 'database',
                    'content': doc['content'],
                    'title': doc['title']
                })
        return results

class MultiSourceRetriever:
    def __init__(self, adapters):
        self.adapters = adapters

    def retrieve(self, query):
        results = []
        seen = set()
        for adapter in self.adapters:
            try:
                adapter_results = adapter.retrieve(query)
                for result in adapter_results:
                    content_hash = hashlib.sha256(result['content'].encode()).hexdigest()
                    if content_hash not in seen:
                        seen.add(content_hash)
                        results.append(result)
            except Exception as e:
                # Log the error
                pass
        return results