"""Shared config + Bedrock client. Imported by all use-case modules."""
import os

from dotenv import load_dotenv

load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")
EMBED_MODEL_ID = os.getenv("EMBED_MODEL_ID", "")


def bedrock_runtime():
    """Return a boto3 bedrock-runtime client for the configured region."""
    import boto3

    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def check_bedrock_access() -> int:
    """Quick connectivity check — returns number of foundation models visible."""
    import boto3

    client = boto3.client("bedrock", region_name=AWS_REGION)
    return len(client.list_foundation_models()["modelSummaries"])


if __name__ == "__main__":
    print(f"Region: {AWS_REGION}")
    print(f"LLM:    {BEDROCK_MODEL_ID or '(unset)'}")
    print(f"Embed:  {EMBED_MODEL_ID or '(unset)'}")
    try:
        print(f"Bedrock models visible: {check_bedrock_access()}")
    except Exception as e:  # noqa: BLE001
        print(f"Bedrock check failed: {e}")
