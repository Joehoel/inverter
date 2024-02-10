import json
import os
from typing import List, Optional
from pptx import Presentation
from io import BytesIO
import boto3
from aws_lambda_powertools.utilities.parser import BaseModel, event_parser
from aws_lambda_powertools.utilities.typing import LambdaContext

BUCKET = os.environ["BUCKET"]
s3 = boto3.client("s3")


class InvertRequest(BaseModel):
    file_key: str


@event_parser(model=InvertRequest)
def handler(event: InvertRequest, context: LambdaContext):
    try:
        response = s3.get_object(Bucket=BUCKET, Key=event.file_key)
        body = response["Body"].read()

        presentation = Presentation(BytesIO(body))
        slide_layout = presentation.slide_layouts[0]
        slide = presentation.slides.add_slide(slide_layout)
        title = slide.shapes.title
        title.text = "Added by lamda function"

        output = BytesIO()
        presentation.save(output)

        new_key = "modified_" + event.file_key

        s3.put_object(Bucket=BUCKET, Key=new_key, Body=output.getvalue())

        return {"status": "success", "key": new_key}
    except Exception as e:
        print(e)
        return {"error": str(e)}
