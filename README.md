# What this?
This repository is used for [Start Python Club #73](https://startpython.connpass.com/event/222478/).
The following demo applications are stored in this repository:
- Streamlit sample application to demonstrate ui widget, visualization, cache and session state
- Deployment configuration to Google App Engine
- Terraform code to deploy the infrastructure to run streamlit for GKE

## Preliminary
- Install gcloud command
- Install docker

## Local Build
```
docker-compose up
```

You can access demo-app in localhost:8080

## App Engine Build

```
gcloud app deploy
```

## Terraform
### Preliminary
- Creating tf-state bucket called `terraform-state-prototype` on GCS.

```
gsutil mb -b on -l asia-northeast1 gs://terraform-state-prototype
```

- Setting `terraform.tfvars`

### Deployment

```
terraform init
terraform plan -var-file=terraform.tfvars
terraform apply -var-file=terraform.tfvars
```

## Kubernetes
### Preliminary
- Install `gettext`

```
apt install -qq -y gettext
```

- Setup Kubernetes

```
gcloud components install kubectl
```

- Change context

```
gcloud container clousters get-credentials --zone asia-northeast1 prototype-cluster
kubectl ctx ${context}
```

### Deploy

Run deployment script

```
./deploy_kubernetes.sh 
```

Run deployment script with .env file

```
export $(cat .env | grep -v ^# | xargs); bash ./deploy_kubernetes.sh
```
