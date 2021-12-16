terraform {
  required_version = ">= 0.13"
  backend "gcs" {
    bucket = "terraform-state-prototype"
    prefix = "terraform/state"
  }
}