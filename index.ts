import fs from "node:fs/promises";
import {
  Document,
  VectorStoreIndex,
  QdrantVectorStore,
  Ollama,
  serviceContextFromDefaults,
  serviceContextFromServiceContext,
} from "llamaindex";

async function main() {
  const path = "/home/userface/Desktop/debt.txt";
  const essay = await fs.readFile(path, "utf-8");
  const llm = new Ollama({ model: "mistral" });
  const vectorStore = new QdrantVectorStore({
    collectionName: "books",
    url: "http://localhost:6333",
  });

  const document = new Document({ text: essay, id_: path });

  const index = await VectorStoreIndex.fromDocuments([document], {
    vectorStore,
    serviceContext: serviceContextFromDefaults({
      llm,
      embedModel: llm,
      chunkOverlap: 25,
      chunkSize: 50,
    }),
  });

  // const index = await VectorStoreIndex.fromVectorStore(vectorStore, serviceContextFromDefaults({llm, embedModel:llm}))
  const queryEngine = index.asQueryEngine();
  const response = await queryEngine.query({
    query: "Where is the land of a thousand temples?",
  });

  // Output response
  console.log(response.toString());
}

main().catch(console.error);
