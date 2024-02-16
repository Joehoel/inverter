import { Function, Job, StackContext, use } from "sst/constructs";

import { Storage } from "./Storage";

export function API({ stack }: StackContext) {
  const { bucket } = use(Storage);

  const fn = new Function(stack, "inverter", {
    handler: "packages/functions/src/inverting",
    description: "Lambda function for inverting PowerPoints",
    environment: {
      BUCKET: bucket.bucketName,
    },
    runtime: "container",
    bind: [bucket],
  });

  return { fn };
}
