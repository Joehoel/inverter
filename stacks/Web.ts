import { RemixSite, StackContext, use } from "sst/constructs";
import { API } from "./API";
import { Storage } from "./Storage";

export function Web({ stack }: StackContext) {
  const { api, fn } = use(API);
  const { bucket } = use(Storage);

  const web = new RemixSite(stack, "web", {
    bind: [bucket, fn, api],
    buildCommand: "pnpm run build",
    path: "apps/web",
  });

  stack.addOutputs({
    url: web.url || "http://localhost:3000",
  });
}
