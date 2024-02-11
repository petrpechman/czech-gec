#!/bin/bash

last_dir=$(pwd | sed 's|.*/||')
sleep_pod="sleep-pod-5bf578fd5b-pxw2v"
use_cp_again=false

while getopts ":c" opt; do
  case ${opt} in
    c )
      use_cp_again=true
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      ;;
  esac
done
shift $((OPTIND -1))


echo ${last_dir}

kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/results-dev ./tmp/checkpoint/results-dev
kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/results-test ./tmp/checkpoint/results-test

if [ "$use_cp_again" = true ]; then
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/dev-retag.all ./tmp/checkpoint/dev-retag.all
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/test-retag.all ./tmp/checkpoint/test-retag.all
fi


create_order
