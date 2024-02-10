import { Bucket, Cron, StackContext } from "sst/constructs";

export function Storage({ stack }: StackContext) {
  const bucket = new Bucket(stack, "Uploads");

  new Cron(stack, "cron", {
    schedule: "rate(1 day)",
    job: {
      function: {
        bind: [bucket],
        handler: "packages/functions/src/delete.handler",
      },
    },
  });

  stack.addOutputs({
    name: bucket.bucketName,
  });

  return { bucket };
}
