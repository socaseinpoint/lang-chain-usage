import { ChatMistralAI } from "@langchain/mistralai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import dotenv from "dotenv";
import { z } from "zod";

const ResponseFormatter = z.object({
  answer: z.string().describe("The answer to the user's question"),
  followup_question: z
    .string()
    .describe("A followup question the user could ask"),
});

const schema = ResponseFormatter;

dotenv.config();

const model = new ChatMistralAI({
  model: "mistral-large-latest",
  apiKey: process.env.MISTRAL_API_KEY,
  temperature: 0
});

// Bind schema to model
const modelWithStructure = model.withStructuredOutput(schema);

const messages = [
  new SystemMessage("You are a helpful assistant that can answer questions and provide followup questions."),
  new HumanMessage("How is the weather in Tokyo?"),
];

const result = await modelWithStructure.invoke(messages);

console.log(result);
