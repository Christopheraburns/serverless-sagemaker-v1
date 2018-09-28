# 8/15/18 - burnsca@amazon.com
# A lambda function to respond to  a Box (box.com) skill event and kick off a SageMaker training process

from __future__ import print_function
import requests
import boto3
import logging
import json
import io
import time
import os
import zipfile
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri


origin = "linearlearner"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
job_id = str(time.time()).replace(".", "-")
bucket = origin + "-" + job_id
prefix1 = 'train'
lambdaRoleARN = 'arn:aws:iam::056149205531:role/lambda_sagemaker'
dataset = ""
output_location = 's3://{}'.format(bucket)


def sagemakerTrain():
    try:
        # get the ARN of the executing role (to pass to Sagemaker for training)
        role = 'arn:aws:iam::056149205531:role/service-role/AmazonSageMaker-ExecutionRole-20180112T102983'
        s3_train_data = 's3://{}/train/{}'.format(bucket, dataset)
        container = get_image_uri(boto3.Session().region_name, 'linear-learner')

        session = sagemaker.Session()

        # set up the training params
        linear = sagemaker.estimator.Estimator(container,
                                               role,
                                               train_instance_count=1,
                                               train_instance_type='ml.c4.xlarge',
                                               output_path=output_location,
                                               sagemaker_session=session)

        # set up the hyperparameters
        linear.set_hyperparameters(feature_dim=13,
                                   predictor_type='regressor',
                                   epochs=10,
                                   loss='absolute_loss',
                                   optimizer='adam',
                                   mini_batch_size=200)

        linear.fit({'train': s3_train_data}, wait=False)


    except Exception as err:
        logger.error("Error while launching SageMaker training: {}".format(err))


def createLambdaTrigger():
    config = "<NotificationConfiguration><CloudFunctionConfiguration>"
    config += "<Filter><S3Key><FilterRule>"
    config += "<Name>prefix</Name><Value>" + "" + "/</Value>"
    config += "</FilterRule></S3Key></Filter>"
    config += "<Id>ObjectCreatedEvents</Id>"
    config += "<CloudFunction>" + "" + "</CloudFunction>"
    config += "<Event>s3:ObjectCreated:*</Event>"
    config += "</CloudFunctionConfiguration></NotificationConfiguration>"



def lambdaFunctionGenerator(origin):
    try:
        # Import statements
        code = "import os\n"
        code += "import io\n"
        code += "import boto3\n"
        code += "import sagemaker\n"
        code += "from sagemaker import get_execution_role\n"
        code += "\n"

        # S3 setup
        code += "bucket = '" + bucket + "'\n"
        code += "prefix = '" + prefix1 + "'\n"
        code += "output_location = 's3://{}/{}/output'.format(bucket, prefix)\n"
        code += "job_id = 'job-epoch'\n"
        code += "\n"
        code += "role = get_execution_role()\n"
        code += "\n"

        # Handler function
        code += "def lambda_handler(event, context):\n"
        code += "\tsm = boto3.client('sagemaker')\n"
        code += "\n"
        # prepare a model context for hosting
        code += "\thosting_container = {\n"
        code += "\t\t'Image': '" + origin + "',\n"
        code += "\t\t'ModelDataUrl': 's3://" + bucket + "/output/" + job_id + "/output/model.tar.gz'\n"
        code += "\t}\n"
        code += "\tcreate_model_response = sm.create_model(\n"
        code += "\t\tModelName='" + job_id + "',\n"
        code += "\t\tExecutionRoleArn=role,\n"
        code += "\t\tPrimaryContainer=hosting_container)\n"
        code += "\n"

        # Create an endpoint configuration
        code += "\tendpoint_config='" + job_id + "-endpoint-config'\n"
        code += "\tcreate_endpoint_config_response = sm.create_endpoint_config(\n"
        code += "\t\tEndpointConfigName=endpoint_config,\n"
        code += "\t\tProductionVariants=[{\n"
        code += "\t\t\t'InstanceType': 'ml.m4.xlarge',\n"  # TODO find a way to make these values configurable
        code += "\t\t\t'InitialIntanceCount':1,\n"
        code += "\t\t\t'ModelName':'" + job_id + "',\n"
        code += "\t\t\t'VariantName':'AllTraffic'}]\n"
        code += "\t)"
        code += "\n"

        # Deploy the endpoint - but don't wait around for it
        code += "\tendpoint = '" + job_id + "-endpoint'\n"
        code += "\n"
        code += "\tcreate_endpoint_response=sm.create_endpoint(\n"
        code += "\t\tEndpointName=endpoint,\n"
        code += "\t\tEndpointConfigName=endpoint_config\n"
        code += "\t)"
    except Exception as err:
        logger.error("Unable to write Lambda_function.py. Exiting. err: {}".format(err))

    return code

def createLambdaFunction():
    try:
        s3 = boto3.resource('s3')
        logger.info("Downloading the Lambda function template...")

        s3.Bucket("serverless-sagemaker").download_file('SagemakerReadyLambdaTemplate.zip',
                                                        '/tmp/SagemakerReadyLambdaTemplate.zip')

        if (os.path.exists('/tmp/SagemakerReadyLambdaTemplate.zip')):
            logger.info("...SagemakerReadyLambdaTemplate.zip download successfully")
        else:
            raise ValueError("Unable to download SagemakerReadyLambdaTemplate.zip!")

        # write lambda_function.py
        # TODO - figure out why we have to switch from .resource('s3') to .client('s3')
        s3 = boto3.client('s3')
        logger.info("writing lambda_function.py...")

        theCode = lambdaFunctionGenerator(origin)

        with open("/tmp/lambda_function.py", "w") as f:
            f.write(theCode)

        logger.info('adding custom lambda_function.py to upload...')
        zipper = zipfile.ZipFile('/tmp/SagemakerReadyLambdaTemplate.zip', 'a')

        zipper.write('/tmp/lambda_function.py', '/tmp/SagemakerReadyLambdaTemplate.zip')
        zipper.close()

        logger.info('uploading new compressed file to S3')
        # Send the zip file to our newly created S3 bucket
        with open('/tmp/SagemakerReadyLambdaTemplate.zip', 'rb') as data:
            s3.upload_fileobj(data, bucket, 'lambdafunction.zip')

        _lambda = boto3.client('lambda')

        logger.info("creating the custom Lambda Function")
        createLambdaResponse = _lambda.create_function(
            FunctionName='lambda-deploy-' + job_id,
            Runtime='python3.6',
            Role=lambdaRoleARN,
            Handler='lambda_function.lambda_handler',
            Code={
                'S3Bucket': bucket,
                'S3Key': 'lambdafunction.zip'
            },
            Description='Lambda deploy function for job-id ' + job_id,
            Timeout=299,
            MemorySize=2048,
            Publish=True
        )

        function_arn = createLambdaResponse['FunctionArn']

        logger.info(createLambdaResponse)
    except Exception as err:
        logger.error("unable to create a lambda function: Exiting. err: {}".format(err))



def transferfile(file_id, file_name, read_token):
    success = True
    msg = ""
    # Get the file info so we can see if this Lambda function has sufficient time to download it
    url = 'https://api.box.com/2.0/files/' + file_id + '?access_token=' + read_token
    r = requests.get(url)
    json_ = json.loads(r.content)
    file_Size = json_['size']
    logger.info('Filesize = ' + str(file_Size) + ' bytes')

    # TODO
    # Write some code to determine if the file can be streamed to S3 within 3+ minutes - if not, move to chunky upload
    # We need an additional 30-40 seconds to create the LambdaFunction that will be triggered by the model.tar.gz
    # creation event

    start = time.time()
    # Stream the newly found file on Box to S3
    logger.info("Streaming target file(" + file_name + ") from api.box.com")
    url = 'https://api.box.com/2.0/files/' + file_id + '/content?access_token=' + read_token
    r = requests.get(url)

    buffer = io.BytesIO()
    buffer.write(r.content)
    # reset memory stream back to the beginning of the file
    buffer.seek(0)

    # upload the file from box to S3
    logger.info("Writing file stream to S3...")
    s3 = boto3.resource('s3')

    try:
        s3.Object('linearlearner-' + job_id, "train/" + file_name).upload_fileobj(buffer)
        buffer.close()

    except Exception as err:
        logger.error("Unable to stream from Box to S3! " + err)
        msg = err
        success = False

    stop = time.time()
    # log a few details from the transfer

    xferTime = stop - start  # in seconds
    mb = file_Size / 1000000  # convert file size to MB
    xferRate = mb / float(xferTime)

    logger.info("FileStreamTime: {}".format(xferTime))
    logger.info("TransferRate: {} (mb/sec)".format(xferRate))

    response = {"Status": str(success), "xferTime": str(xferTime), "xferRate": str(xferRate), "msg": msg}

    return response


def createS3Bucket():
    success = True
    global output_location
    global bucket
    s3 = boto3.client('s3')
    output_location = 's3://{}/output'.format(bucket)
    try:
        logger.info("Creating S3 bucket {} and sub directories for Training data and Model output.".format(bucket))
        s3CreateResponse = s3.create_bucket(
            ACL='private',
            Bucket=bucket
        )
        if str(s3CreateResponse).__contains__(bucket):  # Bucket created successfully
            logger.info("...s3 bucket created successfully. Creating bucket sub-folders...")
    except Exception as err:
        logger.error("Unable to create S3 bucket.  Exiting.  Err: {}".format(err))
        success = False

    # Create S3 bucket "train" sub-folder
    # TODO this feels "hacky" is there a better way to create an empty S3 sub "directory"?
    try:
        prefixOneCreateResponse = s3.put_object(
            Bucket=bucket,
            Body='',
            Key='train' + '/'
        )
        if str(prefixOneCreateResponse).__contains__("'HTTPStatusCode': 200"):
            logger.info("...'train' sub-folder created successfully")
        else:
            raise ValueError("unable to create s3 'train' sub-directory: {}".format(prefixOneCreateResponse))
    except Exception as err:
        logger.error("s3 'train' sub-directory not created.  Exiting. Err: {}".format(err))
        sucess = False

    return success

# This lambda_handler is written specifically for the Box Skill - it can easily be adapted for other sources
def lambda_handler(event, context):
    global origin
    global dataset

    logger.info("Extracting BoxSkill payload-parameters from the event object")
    logger.info(event)

    skill = event['skill']['id']

    if skill is None:
        logger.error("Skill-Id not found in the parameters")
        skill = "not_found"  # Not a fatal error

    write_token = event['token']['write']['access_token']
    if write_token is None:
        logger.error("Write access token not found in the parameters")
        write_token = "not_found"  # Not a fatal error

    read_token = event['token']['read']['access_token']
    if read_token is None:
        logger.error("Read access token not found in the parameters - Fatal Error")
        read_token = "not_found"  # Fatal error - no way to stream the box file to s3 without this.
        # TODO notify and exit gracefully

    file_id = event['source']['id']
    if file_id is None:
        logger.error("File ID not found in the parameters - Fatal Error")
        file_id = "not_found"  # Fatal error - no way to stream the box file to s3 without this.
        # TODO - notify and exit gracefully

    file_name = event['source']['name']
    if file_name is None:
        logger.error("File Name not found in the parameters - Fatal Error")
        file_name = "not_found"  # Fatal error - no way to stream the box file to s2 without this.
        # TODO - notify and exit gracefully
    else:
        dataset = file_name


    # Create S3 directory based off the folder the dataset is in
    # S3 bucket naming convention:
    #   [algo]-[time.time()]/train
    #   [alog]-[time.time()]/time.time()
    # time.time() will be the unique job-id of this serverless-sagemaker "session"
    if(createS3Bucket()):

        # Transfer the dataset from the original directory to s3 for sagemaker training
        transferResult = transferfile(file_id, file_name, read_token)

        if transferResult['Status'] == 'True':
            parse_result = '{"skill-id": ' + skill + ', "write_token": ' + write_token + ', \
                    "read_token": ' + read_token + ', "file_id": ' + file_id + ', "file_name": ' + file_name + ', \
                    "StreamTransferTime": ' + transferResult['xferTime'] + ', "StreamTransferRate": ' + \
                   transferResult['xferRate'] + '}'
        else:
            parse_result = '{"error": ' + transferResult['msg'] + ']'

        # Create the Lambda function that will be triggered when the Sagemaker training process is complete
        createLambdaFunction()

        # TODO Create Trigger for the Lambda Function we just created
        createLambdaTrigger()

        # Kick off the SageMaker training
        sagemakerTrain()

        # NOTE!!!!  This writes Write and Read tokens to CloudWatch - don't do this!!!  This is only for development
        # Read the above ^^ note ^^ that you probably skipped over!!
        logger.info("Event parsed. Results = " + str(transferResult))

    # Return a response suitable for the API Gateway
    return parse_result