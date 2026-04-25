#!/bin/bash

# Sync to remote with rclone
#
# Check environment
[ -z "$PREFIX_LOCAL_RCLONE_PROJECT" ] && { echo "Prefijo de projecto local no establecido " 1>&2 ; exit 1 ;}
[ -z "$PREFIX_REMOTE_RCLONE_PROJECT" ] && { echo "Prefijo de projecto remoto no establecido " 1>&2 ; exit 1 ;}


local_folder="$(basename $PWD)"
rclone sync gdrive:"${PREFIX_REMOTE_RCLONE_PROJECT}/$local_folder" "${PREFIX_LOCAL_RCLONE_PROJECT}/$local_folder"  --progress
