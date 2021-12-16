#!/bin/bash

context=`kubectl config current-context`
echo "Do you want to deploy to $context? (Y/n):"
read ans; echo

if [[ $ans =~ ^(n|N)$ ]]; then
    echo "canceled."
    exit 0
fi

if [ -z $PROJECT_ID ]; then
    read -p "input PROJECT_ID:" PROJECT_ID; echo
fi

if [ -z $TAG ]; then
    read -p "input TAG:" TAG; echo
fi

if [ -z $APP_NAME ]; then
    read -p "input APP_NAME:" APP_NAME; echo
fi

if [ -z $NAMESPACE ]; then
    read -p "input NAMESPACE:" NAMESPACE; echo
fi

ns_check=`kubectl get ns | grep $NAMESPACE`
if [ -z $ns_check ]; then
    echo "create namespace $NAMESPACE"
    kubectl create ns $NAMESPACE
fi

kubectl config set-context --current --namespace=$NAMESPACE

echo "docker build and push to gcr"
docker build -t ${APP_NAME}:${TAG} .
docker tag ${APP_NAME}:${TAG} gcr.io/${PROJECT_ID}/${APP_NAME}:${TAG}
docker push gcr.io/${PROJECT_ID}/${APP_NAME}:${TAG}

echo "envsubst manifest file"
cat kustomization.yaml.template | envsubst > kustomization.yaml
kubectl apply -k ./

echo "done"
