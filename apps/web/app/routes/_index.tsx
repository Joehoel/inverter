import { InvokeCommand, LambdaClient } from "@aws-sdk/client-lambda";
import { GetObjectCommand } from "@aws-sdk/client-s3";
import {
  Box,
  Button,
  Code,
  Container,
  Flex,
  Group,
  List,
  Progress,
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
  IconSlideshow,
  IconUpload,
  IconX,
} from "@tabler/icons-react";
import { motion } from "framer-motion";
import { useState } from "react";
import { Bucket } from "sst/node/bucket";
import { Function } from "sst/node/function";
import { s3 } from "~/lib/s3";
import { s3UploaderHandler } from "~/upload-handler.server";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";

export const meta: MetaFunction = () => {
  return [
    { title: "Inverter" },
    { name: "description", content: "PowerPoint Inverter!" },
  ];
};

const lambda = new LambdaClient({});

async function invert(files: string[]) {
  const response = await lambda.send(
    new InvokeCommand({
      FunctionName: Function.inverter.functionName,
      Payload: JSON.stringify({
        file_keys: files,
      }),
    })
  );

  const payload = JSON.parse(new TextDecoder().decode(response.Payload));
  console.log({ payload });
  const fileKey = payload.key;
  const url = await getSignedUrl(
    s3,
    new GetObjectCommand({
      Bucket: Bucket.Uploads.bucketName,
      Key: fileKey,
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

  const files = formData.getAll("upload").map((file) => file.toString());
  const url = await invert(files);

  return json({
    url: url,
  });
}

export default function Index() {
  const fetcher = useFetcher<typeof action>();
  const isLoading = fetcher.state !== "idle";
  const [files, setFiles] = useState<FileWithPath[]>([]);

  return (
    <div>
      <Title>Inverted!</Title>

      <Container>
        <fetcher.Form method="POST" encType="multipart/form-data">
          <Dropzone
            onDrop={setFiles}
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
                <IconSlideshow
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
                  Drag PowerPoint files here to invert
                </Text>
                <Text size="sm" c="dimmed" inline mt={7}>
                  Attach as many files as you like, each file should not exceed
                  5mb
                </Text>
              </div>
            </Group>
          </Dropzone>
          <Box my="sm">
            <FileList files={files} />
          </Box>
          {fetcher.data?.url ? (
            // Download link
            <Button
              component="a"
              disabled={isLoading || !fetcher.data?.url}
              href={fetcher.data?.url}
              rightSection={<IconDownload size={14} />}
            >
              Download
            </Button>
          ) : (
            <Button disabled={isLoading} type="submit">
              Invert
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
