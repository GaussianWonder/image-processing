# Image Processing Template

## Specs

This is a cross platform ImageProcessing template. It should work on any device with conan and cmake installed.

## Build

Generally, the order is

1. Install dependencies
2. Build
3. Run

**Several ways to do that will be provided**, both *using* the helper script, and *without using* it.

At the time of writing this, the conan version used is `Conan version 1.44.0` with experimental features enabled. Future versions of conan (>2.0+) should not be affected.

The following aliases are used in this document:

```bash
alias run="./run.sh"
alias log="./log_run.sh"
```

Both **run** and **log** accept the same arguments, that of the command to **run**. Both commands will log runtime information, however `log.sh` will also log the whole building process.

<div class="page" />

### Help menu

```bash
run help
```

```text
Example:
  run clean build opengl_template
  run opengl_template
  run build
  run clean build
  run clean
  run conan

  :help   displays this message
  :clean  clean build folder
  :build  build project
  :exec   execute the executable
  :cb     clean build shorthand
  :conan  same as build
  :dependencies  conan install dependencies
```

The run script will take its arguments and execute them sequentially.

Custom commands can be provided as stringified paths to the run command.

Since this is a simple script, i recommend to read it for further customisation and understanding of the build process.

### Get up and running

```bash
run clean dependencies build execute
```

### Fast rebuild and run

```bash
run build execute
```

<div class="page" />

### Build without the helper script

```bash
# Create build folder
mkdir build 
cd build
# Copy dependencies that will be compiled alongside this project
conan source .. --source-folder dependencies
# Build and Link other dependencies that do not require dependency management
conan install .. --build missing
# Build the project
conan build ..
```

## Recommendations

If using VSCode, I recommend the following settings for **command-runner**.

```json
"command-runner.terminal.autoClear": true,
"command-runner.terminal.autoFocus": true,
"command-runner.commands": {
  "install": "./run.sh dependencies",
  "build": "./run.sh conan",
  "run": "./run.sh conan execute",
  "clean": "./run.sh clean",
  "build run": "./run.sh dependencies conan execute",
  "log build": "./log_run.sh dependencies conan",
  "log run": "./log_run.sh conan execute",
  "log build run": "./log_run.sh dependencies conan execute"
}
```

Now useful commands are accesible via the `CTRL + SHIFT + R` shortcut.
