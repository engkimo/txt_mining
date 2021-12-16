variable "network" {
  type = map(string)

  default = {
    vpc_name = "prototype-vpc"
    subnet_name = "prototype-vpc-subnet"
    vpc_access_name = "prototype-vpc-serverless"
  }
}

variable "gke" {
  type  = map(string)

  default = {
    cluster_name = "prototype-cluster"
    location = "asia-northeast1"
    master_ipv4_cidr_block = "172.23.0.0/28"
  }
}

variable "nat" {
  type  = map(string)

  default = {
    router_name = "prototype-router"
    nat_name = "prototype-nat"

  }
}

# specified variable

variable "project_id" {
  type = string
  description = "gcp project id"
}

variable "master_authorized_ip" {
  type = string
  description = "gke master authorized ip"
}

