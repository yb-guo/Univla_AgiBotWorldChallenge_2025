cd /work
mkdir -p ~/.blobfuse2
echo $AZURE_ACCESS_TOKEN
echo "test"
/work/amlt_tools/aml/mount_storage.sh
# pip install --no-cache-dir ".[test, aloha, xarm, pusht, dynamixel, smolvla]"
pip install -U amlt --index-url https://msrpypi.azurewebsites.net/stable/leloojoo
/bin/bash