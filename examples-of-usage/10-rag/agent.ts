import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { ChatMistralAI } from "@langchain/mistralai";
import { MemorySaver } from "@langchain/langgraph";
import { BaseMessage, HumanMessage, AIMessage } from "@langchain/core/messages";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import dotenv from "dotenv";

dotenv.config();

/**
 * Convert complex messages to plain text that Mistral can process
 * This specifically handles the reference and complex content types
 * that caused the original error
 */
function convertMessagesToPlainText(messages: BaseMessage[]): string {
  return messages
    .map((msg) => {
      const role = msg._getType().toLowerCase();
      let contentText = "";
      
      // Handle the complex content structure from agent outputs
      if (typeof msg.content === "object" && msg.content !== null) {
        // If it's an array (common with tool outputs and references)
        if (Array.isArray(msg.content)) {
          contentText = msg.content
            .map((item) => {
              if (typeof item === "object") {
                // Convert any objects to text representation
                return JSON.stringify(item);
              }
              return String(item);
            })
            .join("\n");
        } else {
          // Handle structured content objects
          contentText = JSON.stringify(msg.content);
        }
      } else {
        // Handle simple text content
        contentText = String(msg.content || "");
      }
      
      return `${role}: ${contentText}`;
    })
    .join("\n\n");
}

// Main execution function
async function main() {
  try {
    // Validate environment variables
    if (!process.env.TAVILY_API_KEY) {
      throw new Error("Missing TAVILY_API_KEY in environment variables");
    }
    if (!process.env.MISTRAL_API_KEY) {
      throw new Error("Missing MISTRAL_API_KEY in environment variables");
    }

    // 1) Define the tool(s)
    const agentTools = [
      new TavilySearchResults({
        maxResults: 3,
        apiKey: process.env.TAVILY_API_KEY,
      }),
    ];

    // 2) Initialize the Mistral LLM
    const agentModel = new ChatMistralAI({
      model: "mistral-large-latest",
      apiKey: process.env.MISTRAL_API_KEY,
      temperature: 0,
    });

    // 3) Initialize memory
    const agentCheckpointer = new MemorySaver();

    // 4) Create the "react" agent
    const agent = createReactAgent({
      llm: agentModel,
      tools: agentTools,
      checkpointSaver: agentCheckpointer,
    });

    console.log("Querying about SF weather...");
    // --- FIRST INVOCATION ---
    const agentFinalState = await agent.invoke(
      { messages: [new HumanMessage("what is the current weather in sf")] },
      { configurable: { thread_id: "42" } },
    );

    // Convert entire response to a single text string
    const flattenedFirstResponse = convertMessagesToPlainText(agentFinalState.messages);
    console.log("=== Agent's answer to SF weather ===");
    console.log(flattenedFirstResponse);

    // --- SECOND INVOCATION ---
    // We manually recreate the context by including the flattened text
    // THIS IS THE KEY FIX - we don't pass the complex message objects
    console.log("\nQuerying about NY weather...");
    const agentNextState = await agent.invoke(
      {
        messages: [
          new HumanMessage(
            `I previously asked about SF weather and you told me: ${flattenedFirstResponse}\n\nWhat about NY?`
          ),
        ],
      },
      { configurable: { thread_id: "43" } }, // Use a new thread ID to avoid message history issues
    );

    // Flatten again to ensure no references or arrays
    const flattenedNextResponse = convertMessagesToPlainText(agentNextState.messages);
    console.log("=== Agent's answer to NY ===");
    console.log(flattenedNextResponse);
  } catch (error) {
    console.error("Error running agent:", error);
    process.exit(1);
  }
}

// Execute the main function
main();
