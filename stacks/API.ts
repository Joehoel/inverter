import { Api, Function, StackContext, use } from "sst/constructs";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";

import { Storage } from "./Storage";

export function API({ stack }: StackContext) {
  const { bucket } = use(Storage);

  const api = new Api(stack, "api", {
    routes: {
      "GET /": "packages/functions/src/lambda.handler",
    },
  });

  const fn = new Function(stack, "inverter", {
    handler: "packages/functions/src/inverting",
    description: "Lambda function for inverting PowerPoints",
    environment: {
      BUCKET: bucket.bucketName,
    },
    runtime: "container",
    initialPolicy: [
      new PolicyStatement({
        actions: ["s3:GetObject", "s3:PutObject"],
        resources: [bucket.bucketArn + "/*"],
        effect: Effect.ALLOW,
      }),
    ],
    url: true,
    bind: [bucket],
  });

  stack.addOutputs({
    endpoint: api.url,
    fn: fn.url,
  });

  return { api, fn };
}
