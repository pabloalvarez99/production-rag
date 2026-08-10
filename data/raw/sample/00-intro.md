# Introduction to Retrieval-Augmented Generation

Retrieval-augmented generation (RAG) is a pattern that grounds a language
model's answers in documents you control. Instead of relying only on what the
model memorized during training, the system first retrieves relevant passages
from a corpus and then asks the model to answer using those passages.

A RAG system has two paths. The offline path, called ingest, turns documents
into searchable chunks: each chunk is embedded into a dense vector that
captures its meaning and stored in a vector database alongside its text and
provenance. The online path, called query, embeds the user's question, finds
the most similar chunks, and passes them to the model as context.

The pattern pays off when answers must be current, citable, or drawn from
private material. Because the retrieved chunks carry their source, the system
can cite where each claim came from, and updating knowledge means
re-ingesting documents rather than retraining a model.
