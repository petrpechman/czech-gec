#!/bin/bash

last_dir=$(pwd | sed 's|.*/||')
sleep_pod="sleep-pod-5bf578fd5b-787tl"
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
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/dev.all ./tmp/checkpoint/dev.all
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/test.all ./tmp/checkpoint/test.all

  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/dev-retag.all ./tmp/checkpoint/dev-retag.all
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/dev_total_geccc ./tmp/checkpoint/dev_total_geccc
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-formal-dev-retag ./tmp/checkpoint/natives-formal-dev-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-formal-dev ./tmp/checkpoint/natives-formal-dev
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-formal-test-retag ./tmp/checkpoint/natives-formal-test-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-formal-test ./tmp/checkpoint/natives-formal-test
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-web-informal-dev-retag ./tmp/checkpoint/natives-web-informal-dev-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-web-informal-dev ./tmp/checkpoint/natives-web-informal-dev
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-web-informal-test-retag ./tmp/checkpoint/natives-web-informal-test-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives-web-informal-test ./tmp/checkpoint/natives-web-informal-test
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/results-dev ./tmp/checkpoint/results-dev
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/results-test ./tmp/checkpoint/results-test
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/retag_dev_total_geccc ./tmp/checkpoint/retag_dev_total_geccc
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/retag_test_total_geccc ./tmp/checkpoint/retag_test_total_geccc
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/romani-dev-retag ./tmp/checkpoint/romani-dev-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/romani-dev ./tmp/checkpoint/romani-dev
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/romani-test-retag ./tmp/checkpoint/romani-test-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/romani-test ./tmp/checkpoint/romani-test
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/second-learners-dev-retag ./tmp/checkpoint/second-learners-dev-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/second-learners-dev ./tmp/checkpoint/second-learners-dev
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/second-learners-test-retag ./tmp/checkpoint/second-learners-test-retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/second-learners-test ./tmp/checkpoint/second-learners-test
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/test-retag.all ./tmp/checkpoint/test-retag.all
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/test_total_geccc ./tmp/checkpoint/test_total_geccc

  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives_formal_dev_new100_retag ./tmp/checkpoint/natives_formal_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives_web_informal_dev_new100_retag ./tmp/checkpoint/natives_web_informal_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/romani_dev_new100_retag ./tmp/checkpoint/romani_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/second_learners_dev_new100_retag ./tmp/checkpoint/second_learners_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives_formal_dev_new100_retag ./tmp/checkpoint/natives_formal_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/natives_web_informal_dev_new100_retag ./tmp/checkpoint/natives_web_informal_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/romani_dev_new100_retag ./tmp/checkpoint/romani_dev_new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/second_learners_dev_new100_retag ./tmp/checkpoint/second_learners_dev_new100_retag

  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/dev.new100 ./tmp/checkpoint/dev.new100
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/test.new100 ./tmp/checkpoint/test.new100
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/dev.new100_retag ./tmp/checkpoint/dev.new100_retag
  kubectl cp ${sleep_pod}:/pechmanp/czech-gec/code/src/${last_dir}/tmp/checkpoint/test.new100_retag ./tmp/checkpoint/test.new100_retag
fi


create_order
