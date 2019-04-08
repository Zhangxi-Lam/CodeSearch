## Command to build docker image

### `docker build -t docker-hello .`

## Run in attach mode, bind port and mount file system

### `docker run -it -p 3000:80 -v ~/Workspace/CodeSearch/Docker/.:/usr/src/app --rm --name my-running-app docker-hello`