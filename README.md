# Calling Tools with Mistral AI

https://js.langchain.com/docs/concepts/tool_calling/

# Structured Outputs with Mistral AI

https://js.langchain.com/docs/concepts/structured_outputs/

# Multimodality with Mistral AI

https://js.langchain.com/docs/concepts/multimodality/

# Chatmodels with Mistral AI

https://js.langchain.com/docs/concepts/chat_models/

# Equivalent calls to Mistral AI

```js
await model.invoke("Hello");

await model.invoke([{ role: "user", content: "Hello" }]);

await model.invoke([new HumanMessage("hi!")]);
```

# Making ChatPromptTemplate

```js
import { ChatPromptTemplate } from "@langchain/core/prompts";

const systemTemplate = "Translate the following from English into {language}";

const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["user", "{input}"],
]);

const promptValue = await promptTemplate.invoke({
  language: "italian",
  text: "hi!",
});

promptValue;

```

https://js.langchain.com/docs/tutorials/llm_chain/

## Example of usage

### Semantic search
Build a semantic search engine over a PDF with document loaders, embedding models, and vector stores.

