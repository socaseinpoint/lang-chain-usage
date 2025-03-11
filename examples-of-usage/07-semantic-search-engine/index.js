
import dotenv from "dotenv";
dotenv.config();


/**
 * Load the PDF file
 */
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";

const loader = new PDFLoader("./test.pdf");
const docs = await loader.load();

/**
 * Split the documents into chunks
 */
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 10000,
  chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);

console.log({
  docs: docs.length,
  splits: allSplits.length,
});

/**
 * Embed the chunks
 */

import { MistralAIEmbeddings } from "@langchain/mistralai";

const embeddings = new MistralAIEmbeddings({
  model: "mistral-embed",
  apiKey: process.env.MISTRAL_API_KEY,
  temperature: 0
});

const vector1 = await embeddings.embedQuery(allSplits[0].pageContent);
const vector2 = await embeddings.embedQuery(allSplits[1].pageContent);

console.assert(vector1.length === vector2.length);
console.log(`Generated vectors of length ${vector1.length}\n`);
console.log(vector1.slice(0, 10));


/**
 * Save the embeddings to a vector store
 */

import { FaissStore } from "@langchain/community/vectorstores/faiss";

const vectorStore = new FaissStore(embeddings, {});

// Try adding documents with more debugging and a smaller batch
console.log("Starting to add documents to vector store...");
const batchSize = 10;

// await vectorStore.addDocuments(allSplits);

try {
  // Option 1: Process in smaller batches with explicit logging
  for (let i = 0; i < allSplits.length; i += batchSize) {
    const batch = allSplits.slice(i, i + batchSize);
    console.log(`Processing batch ${i/batchSize + 1}/${Math.ceil(allSplits.length/batchSize)}`);
    
    // Process each document individually to see where it hangs
    for (let j = 0; j < batch.length; j++) {
      console.log(`  Adding document ${i+j+1}/${allSplits.length}`);
      await vectorStore.addDocuments([batch[j]]);
    }
  }
  console.log("All documents added successfully");
} catch (error) {
  console.error("Error during document addition:", error);
}

const results1 = await vectorStore.similaritySearch(
  "When was Nike incorporated?"
);

console.log('Results', results1[0]);
