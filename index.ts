import fs from "node:fs/promises";
import {
  Document,
  VectorStoreIndex,
  serviceContextFromDefaults,
  Ollama,
  QdrantVectorStore,
  storageContextFromDefaults,
  SimilarityPostprocessor,
  NodeWithScore
} from "llamaindex";

async function main() {
  const path = process.argv[2];
  const corpus = await fs.readFile(path, "utf-8");
  const llm = new Ollama({ model: "mistral" });
  const embedModel = new Ollama({ model: "mistral:instruct" });
  const vectorStore = new QdrantVectorStore({
    collectionName: "books",
    url: "http://localhost:6333",
  })
    
 //const index = await VectorStoreIndex.fromDocuments([document], {
 //  vectorStore,
 //  serviceContext: serviceContextFromDefaults({
 //    llm: llm,
 //    embedModel: embedModel,
 //    chunkSize: 50,
 //    chunkOverlap: 25
 //  }),
 //});

 const index = await VectorStoreIndex.fromVectorStore(
   vectorStore,
   serviceContextFromDefaults({
     llm: llm,
     embedModel: embedModel,
   })
 );
  const response = await index.asQueryEngine().query({query:process.argv[3]});
  console.log(response);
}
main().catch(console.error);
