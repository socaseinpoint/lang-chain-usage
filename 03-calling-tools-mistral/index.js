import { ChatMistralAI } from "@langchain/mistralai";
import { StructuredTool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";

dotenv.config();

// Use StructuredTool instead of Tool
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

const generateRandomInts = new GenerateRandomInts();

const model = new ChatMistralAI({
  model: "mistral-large-latest",
  apiKey: process.env.MISTRAL_API_KEY,
  temperature: 0
});

const llmWithTools = model.bindTools([generateRandomInts]);

const result = await llmWithTools.invoke(
  "generate 6 positive ints less than 25"
);

console.log(result);
