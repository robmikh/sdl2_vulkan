#!/bin/bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd)"

REPO_ROOT_PATH="$(dirname $(dirname "$CURRENT_DIR"))"
BUILDPATH="$REPO_ROOT_PATH/xcode"

cmake . "-B$BUILDPATH" -GXcode
open "$REPO_ROOT_PATH/xcode/sdl2_vulkan.xcodeproj"
