import type { PutObjectCommandInput } from "@aws-sdk/client-s3";
import { PutObjectCommand } from "@aws-sdk/client-s3";
import type { UploadHandler } from "@remix-run/node";
import { Bucket } from "sst/node/bucket";
import { s3 } from "./lib/s3";

async function uploadStreamToS3(
  data: AsyncIterable<Uint8Array>,
  key: string,
  contentType: string
) {
  const params: PutObjectCommandInput = {
    Bucket: Bucket.Uploads.bucketName,
    Key: key,
    Body: await convertToBuffer(data),
    ContentType: contentType,
  };

  await s3.send(new PutObjectCommand(params));

  console.log(
    `Uploaded ${key} to ${Bucket.Uploads.bucketName} with content type ${contentType}`
  );

  return key;
}

// The UploadHandler gives us an AsyncIterable<Uint8Array>, so we need to convert that to something the aws-sdk can use.
// Here, we are going to convert that to a buffer to be consumed by the aws-sdk.
async function convertToBuffer(a: AsyncIterable<Uint8Array>) {
  const result = [];
  for await (const chunk of a) {
    result.push(chunk);
  }
  return Buffer.concat(result);
}

export const s3UploaderHandler: UploadHandler = async ({
  filename,
  data,
  contentType,
}) => {
  return await uploadStreamToS3(data, filename!, contentType);
};
