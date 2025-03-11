import { ChatMistralAI } from "@langchain/mistralai";
import { StructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";

dotenv.config();

class GenerateRandomInts extends StructuredTool {
  name = "generateRandomInts";
  description = "Generate a list of random integers";
  schema = z.object({
    count: z.number().describe("The number of integers to generate"),
    max: z.number().describe("The maximum value for the random integers").default(25)
  });

  async _call({ count, max }) {
    const numbers = [];
    for (let i = 0; i < count; i++) {
      numbers.push(Math.floor(Math.random() * max) + 1);
    }
    return JSON.stringify(numbers);
  }
}

async function main() {
  const generateRandomInts = new GenerateRandomInts();
  
  const model = new ChatMistralAI({
    model: "mistral-large-latest",
    apiKey: process.env.MISTRAL_API_KEY,
    temperature: 0
  });
  
  const llmWithTools = model.bindTools([generateRandomInts]);
  
  // Extract tool calls and prepare args for the tool
  const extractToolCallArgs = (aiMessage) => {
    if (!aiMessage.tool_calls || aiMessage.tool_calls.length === 0) {
      throw new Error("No tool calls found in the response");
    }
    return aiMessage.tool_calls[0].args;
  };
  
  // Create the chain with proper method references
  const chain = llmWithTools
    .pipe(extractToolCallArgs)
    .pipe(args => generateRandomInts._call(args));
  
  const result = await chain.invoke("generate 6 positive ints less than 25");
  
  console.log("Random numbers:", JSON.parse(result));
}

main().catch(console.error);
