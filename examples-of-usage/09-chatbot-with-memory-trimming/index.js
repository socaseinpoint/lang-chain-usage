import dotenv from "dotenv";
dotenv.config();

import { ChatMistralAI } from "@langchain/mistralai";
import {
  START,
  END,
  StateGraph,
  MemorySaver,
  MessagesAnnotation
} from "@langchain/langgraph";
import { trimMessages } from "@langchain/core/messages";
import { v4 as uuidv4 } from "uuid";

// 1) Instantiate the chat model
const llm = new ChatMistralAI({
  model: "mistral-large-latest",
  apiKey: process.env.MISTRAL_API_KEY, 
  temperature: 0,
});

// 2) Create a trimmer that prunes older messages. Adjust to your needs:
const trimmer = trimMessages({
  // Maximum tokens/characters to keep in the final messages array
  maxTokens: 1000,
  // For short code examples, we just measure token usage as text length.
  // For actual usage, plug in a real token counter that matches your model’s tokenizer.
  tokenCounter: (messages) =>
    messages.reduce((acc, msg) => acc + msg.content.length, 0),

  // Where to begin discarding old messages if we exceed maxTokens
  strategy: "last",         // keep the newest messages, discard oldest
  includeSystem: true,      // always keep system messages
  allowPartial: false,      // don’t partially trim a single message
  startOn: "human",         // start trimming right after a human message
});

// 3) Node function that calls the model
const callModel = async (state) => {
  // Trim older messages before passing them to the LLM
  const trimmedMessages = await trimmer.invoke(state.messages);
  const response = await llm.invoke(trimmedMessages);
  return { messages: [response] };
};

// 4) Create the workflow and compile with an in-memory saver
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("model", callModel)
  .addEdge(START, "model")
  .addEdge("model", END);

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

  // -- Start a new conversation by using a different thread_id
  const newThreadConfig = { configurable: { thread_id: uuidv4() } };
  output = await app.invoke(
    { messages: [{ role: "user", content: "What's my name?" }] },
    newThreadConfig
  );
  printLastResponse(output);

  // -- Return to the original thread
  output = await app.invoke(
    { messages: [{ role: "user", content: "What's my name?" }] },
    mainThreadConfig
  );
  printLastResponse(output);
}

runChatExample();
