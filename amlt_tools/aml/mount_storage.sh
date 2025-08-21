# https://lightaml3149727845.blob.core.windows.net/sharedcontainer
# pushi dataset
# account_name=rushaml2996082614
# container_name=sharedcontainer # sharedcontainer

# srobot dataset
account_name=msrasrobotazur5594418711
container_name=msrasrobotvlnoutputs # msrasrobotvlnoutputs

mount_path=/work/shared_outputs/
tmp_path=/mnt/outputs_tmp_data/
 
# Write the config file
cat > ~/.blobfuse2/config.yaml << EOL
file_cache:
    path: $tmp_path
azstorage:
    type: block
    account-name: $account_name
    endpoint: https://$account_name.blob.core.windows.net
    container: $container_name
    mode: azcli
EOL
# Mount the directory
myuser=$(whoami)
mkdir $mount_path -p
mkdir $tmp_path -p
blobfuse2 mount $mount_path --config-file ~/.blobfuse2/config.yaml -o allow_other


account_name=msrasrobotazur5594418711
container_name=msrasrobotvlndata # msrasrobotvlnoutputs

mount_path=/work/data/
tmp_path=/mnt/data_tmp_data/
 
# Write the config file
cat > ~/.blobfuse2/config.yaml << EOL
file_cache:
    path: $tmp_path
azstorage:
    type: block
    account-name: $account_name
    endpoint: https://$account_name.blob.core.windows.net
    container: $container_name
    mode: azcli
EOL
# Mount the directory
myuser=$(whoami)
mkdir $mount_path -p
mkdir $tmp_path -p
blobfuse2 mount $mount_path --config-file ~/.blobfuse2/config.yaml -o allow_other