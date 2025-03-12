import dotenv from "dotenv";
import { ChatMistralAI } from "@langchain/mistralai";
import {
  START,
  END,
  StateGraph,
  MemorySaver,
  MessagesAnnotation
} from "@langchain/langgraph";

import { v4 as uuidv4 } from "uuid";

dotenv.config();


// 1) Instantiate the chat model
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  apiKey: process.env.MISTRAL_API_KEY, 
  temperature: 0,
});

// 2) Define a single “node” function that calls the model with all messages so far
//    Note: this function must return an object containing { messages: [...] }
const callModel = async (state) => {
  // state.messages already includes the conversation so far (from memory)
  const response = await llm.invoke(state.messages);
  // Return the new AI response as an array
  return { messages: [response] };
};

// 3) Create a small workflow (state machine) that has a single node and edge
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("model", callModel)
  .addEdge(START, "model")
  .addEdge("model", END);

// 4) Compile it, adding a MemorySaver checkpointer for in-memory persistence
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });

// 5) Each new “thread_id” is like a separate chat session.
const mainThreadConfig = { configurable: { thread_id: uuidv4() } };

// Helper to pretty-print the last AI response
function printLastResponse(output) {
  const { messages } = output;
  const last = messages[messages.length - 1];
  console.log("AI:", last.content);
}

async function runChatExample() {
  // -- Turn 1 on main thread:
  let output = await app.invoke(
    { messages: [{ role: "user", content: "Hi, I'm Bob." }] },
    mainThreadConfig
  );
  printLastResponse(output);

  // -- Turn 2 on main thread:
  output = await app.invoke(
    { messages: [{ role: "user", content: "What's my name?" }] },
    mainThreadConfig
  );
  printLastResponse(output);

  // -- Start a new (fresh) conversation by using a different thread_id
  const newThreadConfig = { configurable: { thread_id: uuidv4() } };
  output = await app.invoke(
    { messages: [{ role: "user", content: "What's my name?" }] },
    newThreadConfig
  );
  printLastResponse(output);

  // -- Return to the original thread and ask again
  output = await app.invoke(
    { messages: [{ role: "user", content: "What's my name?" }] },
    mainThreadConfig
  );
  printLastResponse(output);
}

runChatExample();
