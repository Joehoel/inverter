import { InvokeCommand, LambdaClient } from "@aws-sdk/client-lambda";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import {
  Button,
  Code,
  Container,
  Flex,
  Group,
  List,
  Text,
  ThemeIcon,
  Title,
  rem,
} from "@mantine/core";
import { Dropzone, FileWithPath, MIME_TYPES } from "@mantine/dropzone";
import {
  ActionFunctionArgs,
  json,
  unstable_parseMultipartFormData,
  type MetaFunction,
} from "@remix-run/node";
import { useFetcher } from "@remix-run/react";
import {
  IconDownload,
  IconFile,
  IconPhoto,
  IconUpload,
  IconX,
} from "@tabler/icons-react";
import { motion } from "framer-motion";
import { useState } from "react";
import { Bucket } from "sst/node/bucket";
import { Function } from "sst/node/function";
import { z } from "zod";
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
  const response = await lambda.send(
    new InvokeCommand({
      FunctionName: Function.inverter.functionName,
      Payload: Buffer.from(
        JSON.stringify({
          file_key: fileKey,
        }),
      ),
    }),
  );

  const payload = JSON.parse(new TextDecoder().decode(response.Payload));

  const result = z
    .object({
      key: z.string(),
    })
    .parse(payload);

  const url = await getSignedUrl(
    s3,
    new GetObjectCommand({
      Bucket: Bucket.Uploads.bucketName,
      Key: result.key,
    }),
    { expiresIn: 15 * 60 },
  );

  return url;
}

export async function action({ request }: ActionFunctionArgs) {
  const formData = await unstable_parseMultipartFormData(
    request,
    s3UploaderHandler,
  );

  const fileName = String(formData.get("upload"));

  console.log("FILE_NAME", fileName);

  // const key = fileName.split(".")[0];
  const url = await invert(fileName).catch(() => {
    console.error("Failed to invert file", fileName);
    return null;
  });

  return json({
    url,
  });
}

export default function Index() {
  const fetcher = useFetcher<typeof action>();
  const isLoading = fetcher.state !== "idle";
  const url = fetcher.data?.url;
  const [files, setFiles] = useState<FileWithPath[]>([]);

  return (
    <div>
      <Title>Inverted!</Title>

      <Container>
        <fetcher.Form method="POST" encType="multipart/form-data">
          <Dropzone
            disabled={isLoading}
            onDrop={setFiles}
            maxSize={5 * 1024 ** 2}
            accept={[MIME_TYPES.pptx]}
            name="upload"
            multiple={false}
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
                  Attach as many files as you like, each file should not exceed
                  5mb
                </Text>
              </div>
            </Group>
          </Dropzone>
          {/* <Progress value={50} styles={{ root: { marginBlock: 20 } }} /> */}

          <FileList files={files} />

          <Button disabled={isLoading} type="submit">
            Invert
          </Button>
          {url && (
            // Download link
            <Button
              component="a"
              disabled={isLoading || !url}
              href={url}
              rightSection={<IconDownload size={14} />}
            >
              Download
            </Button>
          )}
        </fetcher.Form>
      </Container>
    </div>
  );
}

function FileList({ files }: { files: FileWithPath[] }) {
  if (files.length === 0) {
    return null;
  }

  return (
    <MotionList
      initial="hidden"
      animate="visible"
      variants={{
        visible: {
          opacity: 1,
          transition: {
            when: "beforeChildren", // Animate the parent first
            staggerChildren: 0.1,
          },
        },
        hidden: {
          opacity: 0,
        },
      }}
      spacing={"xs"}
      size="md"
    >
      {files.map((file) => (
        <MotionListItem
          key={file.path}
          variants={{
            visible: {
              opacity: 1,
              x: 0,
            },
            hidden: { opacity: 0, x: -50 },
          }}
          icon={
            <ThemeIcon size={24} radius="xl">
              <IconFile style={{ width: rem(16), height: rem(16) }} />
            </ThemeIcon>
          }
          // You can also add custom transition per item if needed
          transition={{ duration: 0.15 }}
        >
          <Flex direction={"row"} align={"center"} content="space-between">
            <Text>{file.path}</Text>
            <Text>
              <Code>{file.size}</Code>
            </Text>
          </Flex>
        </MotionListItem>
      ))}
    </MotionList>
  );
}

const MotionList = motion(List);
const MotionListItem = motion(List.Item);
