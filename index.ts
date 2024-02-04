import fs from "node:fs/promises";
import {
  Document,
  VectorStoreIndex,
  QdrantVectorStore,
  Ollama,
  serviceContextFromDefaults,
  serviceContextFromServiceContext,
  similarity,
} from "llamaindex";

async function main() {
  const path = "slack.txt";
  const essay = await fs.readFile(path, "utf-8")
  ;
  const llm = new Ollama({
    model: "mistral",
  });
  const vectorStore = new QdrantVectorStore({
    collectionName: "slack",
    url: "http://localhost:6333",
  });

  const documents = essay.split("\n").map(text => new Document({ text, id_: path }))

  const index = await VectorStoreIndex.fromDocuments(documents, {
    vectorStore,
    serviceContext: serviceContextFromDefaults({
      llm: llm,
      embedModel: llm,
      // chunkOverlap: 25,
      // chunkSize: 50,
    }),
  });

  // const index = await VectorStoreIndex.fromVectorStore(
  //   vectorStore,
  //   serviceContextFromDefaults({
  //     llm: llm,
  //     embedModel: llm,
  //   })
  // );
  const response = await index
    .asRetriever()
    .retrieve(
      "Napkins"
    );

  // Output response
  console.log(response);
}

main().catch(console.error);
