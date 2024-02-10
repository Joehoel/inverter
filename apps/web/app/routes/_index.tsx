import { z } from "zod";
import { InvokeCommand, LambdaClient } from "@aws-sdk/client-lambda";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import { Group, Text, rem } from "@mantine/core";
import { Dropzone, MIME_TYPES } from "@mantine/dropzone";
import {
  ActionFunctionArgs,
  json,
  unstable_parseMultipartFormData,
  type MetaFunction,
} from "@remix-run/node";
import { Form, useActionData } from "@remix-run/react";
import { IconPhoto, IconUpload, IconX } from "@tabler/icons-react";
import { Bucket } from "sst/node/bucket";
import { Function } from "sst/node/function";
import { Button } from "~/components/ui/button";
import { s3 } from "~/lib/s3";
import { s3UploaderHandler } from "~/upload-handler.server";

export const meta: MetaFunction = () => {
  return [
    { title: "Inverter" },
    { name: "description", content: "PowerPoint Inverter!" },
  ];
};

const lambda = new LambdaClient({});

async function invert(fileKey: string) {
  const cmd = new InvokeCommand({
    FunctionName: Function.inverter.functionName,
    Payload: JSON.stringify({
      file_key: fileKey,
    }),
  });

  const response = await lambda.send(cmd);
  const payload = JSON.parse(new TextDecoder().decode(response.Payload));
  const { key } = z
    .object({
      key: z.string(),
    })
    .parse(payload);

  const url = await getSignedUrl(
    s3,
    new GetObjectCommand({
      Bucket: Bucket.Uploads.bucketName,
      Key: key,
    }),
    { expiresIn: 15 * 60 }
  );

  return url;
}

export async function action({ request }: ActionFunctionArgs) {
  const formData = await unstable_parseMultipartFormData(
    request,
    s3UploaderHandler
  );

  const fileName = String(formData.get("upload"));
  const key = fileName.split(".")[0];
  const url = await invert(key);

  return json({
    url,
  });
}

export default function Index() {
  const data = useActionData<typeof action>();

  return (
    <Form method="POST" encType="multipart/form-data">
      <Dropzone
        onDrop={(files) => {
          console.log({ files });
        }}
        onReject={(files) => console.log("rejected files", files)}
        maxSize={5 * 1024 ** 2}
        accept={[MIME_TYPES.pptx]}
        name="upload"
      >
        <Group
          justify="center"
          gap="xl"
          mih={220}
          style={{ pointerEvents: "none" }}
        >
          <Dropzone.Accept>
            <IconUpload
              style={{
                width: rem(52),
                height: rem(52),
                color: "var(--mantine-color-blue-6)",
              }}
              stroke={1.5}
            />
          </Dropzone.Accept>
          <Dropzone.Reject>
            <IconX
              style={{
                width: rem(52),
                height: rem(52),
                color: "var(--mantine-color-red-6)",
              }}
              stroke={1.5}
            />
          </Dropzone.Reject>
          <Dropzone.Idle>
            <IconPhoto
              style={{
                width: rem(52),
                height: rem(52),
                color: "var(--mantine-color-dimmed)",
              }}
              stroke={1.5}
            />
          </Dropzone.Idle>

          <div>
            <Text size="xl" inline>
              Drag images here or click to select files
            </Text>
            <Text size="sm" c="dimmed" inline mt={7}>
              Attach as many files as you like, each file should not exceed 5mb
            </Text>
          </div>
        </Group>
      </Dropzone>
      <Button type="submit">Submit</Button>
      {data && data.url && (
        // Download link
        <a href={data.url} download>
          Download
        </a>
      )}
    </Form>
  );
}
