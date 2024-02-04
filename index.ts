import fs from "node:fs/promises";
import {
  Document,
  VectorStoreIndex,
  serviceContextFromDefaults,
  AstraDBVectorStore,
  storageContextFromDefaults,
  SimilarityPostprocessor,
  NodeWithScore
} from "llamaindex";

class MySimilarity extends SimilarityPostprocessor {
  postprocessNodes(nodes: NodeWithScore[]) {
    console.log(nodes)
    return super.postprocessNodes(nodes);
  }
}

async function main() {
  const path = "slack.txt";
  const essay = await fs.readFile(path, "utf-8");
  // const llm = new Ollama({ model: "mistral", });

  const astraCondor = new AstraDBVectorStore({
    params: {
      token: process.env.ASTRA_DB_APPLICATION_TOKEN as string,
      endpoint: process.env.ASTRA_DB_API_ENDPOINT as string
    },
  });

  await astraCondor.connect("condor")
  const documents = essay.split("\n").map(text => new Document({ text, id_: path }))

  // const index = await VectorStoreIndex.fromDocuments(documents, {
  //   storageContext: await storageContextFromDefaults({ vectorStore: astraCondor }),
  //   serviceContext: serviceContextFromDefaults({ }),
  // });

  const index = await VectorStoreIndex.fromVectorStore(
    astraCondor,
    serviceContextFromDefaults()
  );
  const retreiver = index.asRetriever({nodePostprocessors:[
    new MySimilarity({
      similarityCutoff: 0.9,
    })
  ]})
  
  const [{score: a}, {score: b}] = await retreiver.retrieve(
      "guns, germs, and steel"
    );

  // Output response
  console.log(a, b);
}

main().catch(console.error);
