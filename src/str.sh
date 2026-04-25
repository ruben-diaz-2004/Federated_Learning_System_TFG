#!/bin/bash

# Sync to remote with rclone
#
# Check environment
[ -z "$PREFIX_LOCAL_RCLONE_PROJECT" ] && { echo "Prefijo de projecto local no establecido " 1>&2 ; exit 1 ;}
[ -z "$PREFIX_REMOTE_RCLONE_PROJECT" ] && { echo "Prefijo de projecto remoto no establecido " 1>&2 ; exit 1 ;}

local_folder="$(basename $PWD)"

local_path=$(find "$PREFIX_LOCAL_RCLONE_PROJECT" -name $local_folder -type d)
[ -z "$local_path" ] && { echo "Ruta fuera de proyecto local" 1>&2 ; exit 1 ; }

removed_prefix="${local_path#"$PREFIX_LOCAL_RCLONE_PROJECT"}"

rclone sync "${PREFIX_LOCAL_RCLONE_PROJECT}/$removed_prefix" gdrive:"${PREFIX_REMOTE_RCLONE_PROJECT}/$removed_prefix" --progress
