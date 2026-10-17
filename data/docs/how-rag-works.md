# How rag-assistant works

Retrieval-augmented generation (RAG) answers questions from your own documents
instead of only from what a language model memorised during training.

## Ingest

1. **Load** every .md, .txt and .pdf file in the docs folder.
2. **Chunk** each file into overlapping passages of about 1,000 characters, cutting
   at paragraph or sentence boundaries where possible.
3. **Embed** each passage with a local embedding model (BAAI/bge-small-en-v1.5 by
   default), turning it into a vector that captures its meaning.
4. **Save** the vectors and passages to the `.index/` folder.

## Ask

1. **Embed** the question with the same model.
2. **Retrieve** the passages whose vectors are most similar to the question
   (cosine similarity, top 5 by default).
3. **Generate**: Claude receives the numbered passages and writes an answer that
   cites them, like [1] or [2], and says so when the documents don't contain the answer.

Because the answer is grounded in retrieved passages, you can check every claim
against its source.
