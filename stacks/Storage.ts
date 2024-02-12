import { Bucket, Cron, Queue, StackContext, use } from "sst/constructs";
import { API } from "./API";

export function Storage({ stack }: StackContext) {
  const queue = new Queue(stack, "queue");

  const bucket = new Bucket(stack, "Uploads", {
    notifications: {
      process: {
        type: "queue",
        queue,
      },
    },
  });

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
