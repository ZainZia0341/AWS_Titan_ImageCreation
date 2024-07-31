import boto3
from botocore.exceptions import ClientError

# Initialize the Bedrock client
bedrock = boto3.client(service_name='bedrock')

# Job details
job_name = 'fine-tuning-titan-image-generation-job'
model_name = 'fine_tuned_titan_image_model'
role_arn = 'arn:aws:iam::236515146491:role/TitanImageGenerationFineTunning'  # Replace with your actual role ARN

# Base model ID for the Titan Image Generation model
base_model_id = "amazon.titan-image-generator-v1:0"  # Ensure this is the correct model ID

# Training Data S3 URI
training_data_s3_uri = "s3://finetunningforimagegeneration/fine_tuning_images/updated_metadata.jsonl"

# Output Data S3 URI
output_data_s3_uri = "s3://finetunningforimagegeneration/output/"

# Number of steps based on 23 images
num_steps = 4000  # Setting the number of steps based on AWS recommendations for 23 images

# Hyperparameters based on AWS recommendations
hyperparameters = {
    "steps": "4000",  # Number of steps
    "batchSize": "8",         # Batch size
    "learningRate": "0.00001" # Learning rate
}

# Create model customization job
try:
    response = bedrock.create_model_customization_job(
        customizationType="FINE_TUNING",
        jobName=job_name,
        customModelName=model_name,
        roleArn=role_arn,
        baseModelIdentifier=base_model_id,
        hyperParameters=hyperparameters,
        trainingDataConfig={"s3Uri": training_data_s3_uri},
        outputDataConfig={"s3Uri": output_data_s3_uri}  # Output S3 path
    )
    print("Model customization job created:", response)
except ClientError as error:
    print("Error creating model customization job:", error)
