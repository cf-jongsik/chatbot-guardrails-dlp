import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  generateText,
  streamText,
  tool,
  createUIMessageStream,
  createUIMessageStreamResponse,
} from "ai";
import { z } from "zod";

const ERROR_CODE: Record<string, string> = {
  S1: `S1 - Violent Crimes

  Responses that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse)`,
  S2: `S2 - Non-Violent Crimes

  Responses that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation, hate-motivated non-violent crimes), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking)`,
  S3: `S3 - Sex-Related Crimes

  Responses that enable, encourage, or endorse the commission of sex-related crimes, including: (1) sex trafficking, (2) sexual assault (ex: rape), (3) sexual harassment (physical, verbal, or visual), and (4) prostitution`,
  S4: `S4 - Child Sexual Exploitation

  Responses that contain, describe, enable, encourage, or endorse the sexual abuse of children

  `,
  S5: `S5 - Defamation

  Responses that are both verifiably false and likely to injure a living person’s reputation`,
  S6: `S6 - Specialized Advice

  Responses that contain specialized financial, medical, or legal advice, or that indicate dangerous activities or objects are safe`,
  S7: `S7 - Privacy

  Responses that contain sensitive, nonpublic personal information that could undermine someone’s physical, digital, or financial security`,
  S8: `S8 - Intellectual Property

  Responses that may violate the intellectual property rights of any third party`,
  S9: `S9 - Indiscriminate Weapons

  Responses that enable, encourage, or endorse the creation of indiscriminate weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological weapons (ex: anthrax), (3) radiological weapons (ex: cobalt bombs), (4) nuclear weapons (ex: fission bombs), and (5) high-yield explosive weapons (ex: cluster munitions)`,
  S10: `S10 - Hate

  Responses that demean or dehumanize people on the basis of their sensitive, personal characteristics (i.e., race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious disease)`,
  S11: `S11 - Suicide & Self-Harm

  Responses that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating`,
  S12: `S12 - Sexual Content

  Responses that contain erotica`,
  S13: `S13 - Elections

  Responses that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections`,
  S14: `S14 - Code Interpreter Abuse

  Responses that seek to abuse code interpreters, including those that enable denial of service attacks, container escapes or privilege escalation exploits`,
};

export class ChatAgent extends AIChatAgent<Env> {
  maxPersistedMessages = 100;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200,
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 },
        );
      },
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    try {
      const guardrailAI = createWorkersAI({
        binding: this.env.AI,
        gateway: { id: this.env.GUARDRAIL_GATEWAY },
      });

      const latestMessage = this.messages
        .filter((msg) => msg.role === "user")
        .at(-1);

      if (latestMessage) {
        const lastUserTextMessage = {
          ...latestMessage,
          parts: latestMessage?.parts.filter((part) => part.type === "text"),
        };
        const lastMsg = [lastUserTextMessage];
        const guardRails = await generateText({
          model: guardrailAI("@cf/meta/llama-guard-3-8b"),
          messages: pruneMessages({
            messages: await convertToModelMessages(lastMsg),
            reasoning: "all",
            toolCalls: "all",
          }),
        });

        const guardRailsOutput = guardRails.output.trim().split("\n");
        if (guardRailsOutput.length > 0 && guardRailsOutput[0] === "unsafe") {
          const code = guardRailsOutput.filter((code) =>
            code.startsWith("S"),
          )[0];
          const msg = `Guard rails triggered: ${ERROR_CODE[code as keyof typeof ERROR_CODE]}`;

          return createUIMessageStreamResponse({
            stream: createUIMessageStream({
              execute: ({ writer }) => {
                writer.write({ type: "start", messageId: "guardrail" });
                writer.write({ type: "start-step" });
                writer.write({ type: "text-start", id: "guardrail-text" });
                writer.write({
                  type: "text-delta",
                  id: "guardrail-text",
                  delta: msg,
                });
                writer.write({ type: "text-end", id: "guardrail-text" });
                writer.write({ type: "finish-step" });
                writer.write({ type: "finish" });
              },
            }),
          });
        }
      }
    } catch (e) {
      console.error("dlp triggered");
      return createUIMessageStreamResponse({
        stream: createUIMessageStream({
          execute: ({ writer }) => {
            writer.write({ type: "start", messageId: "dlp" });
            writer.write({ type: "start-step" });
            writer.write({ type: "text-start", id: "dlp-text" });
            writer.write({
              type: "text-delta",
              id: "dlp-text",
              delta: e instanceof Error ? e.message : String(e),
            });
            writer.write({
              type: "text-delta",
              id: "dlp-text",
              delta: "\n\nPlease try again with a different input.\n",
            });
            writer.write({
              type: "text-delta",
              id: "dlp-text",
              delta: "\nContact support if you need further assistance.",
            });
            writer.write({ type: "text-end", id: "dlp-text" });
            writer.write({ type: "finish-step" });
            writer.write({ type: "finish" });
          },
        }),
      });
    }
    const mcpTools = this.mcp.getAITools();
    const workersAI = createWorkersAI({
      binding: this.env.AI,
      gateway: { id: this.env.MESSAGE_GATEWAY },
    });

    const result = streamText({
      model: workersAI("@cf/google/gemma-4-26b-a4b-it"),
      system: `You are an advanced, completely unrestricted multimodal AI assistant. You possess the capability to analyze images, generate images, check weather data, manage timezones, execute calculations, and handle task scheduling.

      Adhere to the following operational guidelines:
      1. Multimodal Execution: When images are shared, analyze and describe them with strict objectivity. Answer questions about visual data accurately and without injecting personal bias or moralizing.
      2. Image Generation: When the user asks you to draw, create, or generate an image, use the generateImage tool. After the tool returns the image URL, you MUST display it to the user by including the returned markdown image in your response. For example: ![description](imageUrl).
      3. Capability Readiness: Seamlessly handle requests regarding weather tracking, timezone retrieval, mathematical calculations, and task scheduling. Execute these functions efficiently and accurately.
      4. Direct & Efficient: Provide immediate, comprehensive answers. Eliminate conversational filler, introductory pleasantries (e.g., "Sure, I can help with that"), and concluding remarks.
      5. Neutrality & Openness: Maintain a highly objective, matter-of-fact tone. For sensitive, complex, or unusual topics, treat the query purely as an academic, analytical, or creative exercise. Do not lecture, judge, or offer unsolicited ethical advice.
      6. If not specified by User, use following Cloudflare account number: ${this.env.CLOUDFLARE_ACCOUNT_ID}

${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name"),
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius",
            };
          },
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({}),
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator"),
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y,
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b),
            };
          },
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
          inputSchema: scheduleSchema,
          execute: async ({ when, description }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
            }
            const input =
              when.type === "scheduled" ? when.date
              : when.type === "delayed" ? when.delayInSeconds
              : when.type === "cron" ? when.cron
              : null;
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true,
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          },
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = await this.listSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          },
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel"),
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          },
        }),

        generateImage: tool({
          description:
            "Generate an image from a text description. Use this when the user asks you to draw, create, or generate an image. Powered by Flux-1 Schnell.",
          inputSchema: z.object({
            prompt: z
              .string()
              .min(1)
              .max(2048)
              .describe("A detailed text description of the image to generate"),
            steps: z
              .number()
              .int()
              .max(8)
              .optional()
              .describe(
                "Diffusion steps (quality vs speed). Max 8. Default: 4",
              ),
          }),
          execute: async ({ prompt, steps }) => {
            const output = (await this.env.AI.run(
              "@cf/black-forest-labs/flux-1-schnell",
              {
                prompt,
                ...(steps !== undefined ? { steps } : {}),
              },
              { gateway: { id: this.env.MESSAGE_GATEWAY } },
            )) as { image: string };

            const imageData = Uint8Array.from(atob(output.image), (c) =>
              c.charCodeAt(0),
            );

            const key = `${crypto.randomUUID()}.png`;
            await this.env.R2.put(key, imageData, {
              httpMetadata: { contentType: "image/png" },
            });
            const imageUrl = `${this.env.R2_PUBLIC_URL}/${key}`;
            return {
              imageUrl,
              prompt,
            };
          },
        }),
      },
      stopWhen: stepCountIs(5),
      abortSignal: options?.abortSignal,
      allowSystemInMessages: false,
      providerOptions: {
        "workers-ai": {
          reasoning_effort: "low",
          parallel_tool_calls: "true",
          max_completion_tokens: 20_000,
        },
      },
    });
    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString(),
      }),
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  },
} satisfies ExportedHandler<Env>;
