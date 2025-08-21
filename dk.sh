#!/bin/bash
DOCKERFILE="Dockerfile"
INSTALL_SOURCE="default"
COMMAND=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dockerfile=*)
            DOCKERFILE="${1#*=}"
            ;;
        --install_source=*)
            INSTALL_SOURCE="${1#*=}"
            ;;
        start|stop|restart|help|--help)
            COMMAND=$1
            ;;
        *)
            echo "Invalid option: $1"
            help
            exit 1
            ;;
    esac
    shift
done

# Ensure a command is provided
if [[ -z "$COMMAND" ]]; then
    echo "Error: No command provided."
    help
    exit 1
fi

# Set container and image names based on the Dockerfile
if [[ "$DOCKERFILE" == "DockerfileM300" ]]; then
    CONTAINER_NAME="m300_aigbotworld_challenge_training"
    IMAGE_NAME="m300_aigbotworld_challenge_training:latest"
else
    CONTAINER_NAME="unilva_agibotworld_trainings"
    IMAGE_NAME="unilva_agibotworld_trainings:latest"
fi
echo "Container name: $CONTAINER_NAME"
echo "Image name: $IMAGE_NAME"
echo "Dockerfile: $DOCKERFILE"

# Get Azure token
TOKEN=$(az account get-access-token --resource https://management.azure.com --query accessToken -o tsv)

# Define functions
start() {
    if [ ! "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
            echo "Container exists but is stopped. Restarting the container..."
            docker start $CONTAINER_NAME
            docker exec -it $CONTAINER_NAME /bin/bash
        else
            echo "Building and starting the container with source: $INSTALL_SOURCE..."
            docker build --file $DOCKERFILE --build-arg INSTALL_SOURCE=$INSTALL_SOURCE -t $IMAGE_NAME .
            docker run -it \
                --gpus all \
                -e NVIDIA_DRIVER_CAPABILITIES=all \
                --name $CONTAINER_NAME \
                -v ./:/work \
                -v /tmp/.X11-unix:/tmp/.X11-unix \
                -v ~/.azure:/root/.azure \
                -v ~/.cache/huggingface/hub:/root/.cache/huggingface/hub \
                -e DISPLAY=unix$DISPLAY \
                -e AZURE_ACCESS_TOKEN=$TOKEN \
                --shm-size=10g \
                --net=host \
                --privileged \
                $IMAGE_NAME /work/setup.sh
        fi
    else
        echo "Container already running. Attaching to the container..."
        docker exec -it $CONTAINER_NAME /bin/bash
    fi
}

stop() {
    if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
        echo "Stopping and removing the container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    elif [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
        echo "Container is already stopped. Removing the container..."
        docker rm $CONTAINER_NAME
    else
        echo "Warning: Container not found."
    fi
}

restart() {
    stop
    start
}

help() {
    echo "Usage: $0 {start|stop|restart|--help} [--dockerfile=<Dockerfile>] [--install_source=<source>]"
    echo "start   - Build and start the container if it doesn't exist, restart if it is stopped, or attach to it if it is running."
    echo "stop    - Stop and remove the container if it exists."
    echo "restart - Stop and then start the container."
    echo "--help  - Display this help message."
}

# Execute the command
case "$COMMAND" in
    start)
        start
        ;;
    stop)
        stop
        ;;
    restart)
        restart
        ;;
    help|--help)
        help
        ;;
    *)
        echo "Invalid command: $COMMAND"
        help
        exit 1
        ;;
esac