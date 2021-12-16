resource "google_compute_network" "prototype_vpc" {
  project = var.project_id
  name = var.network.vpc_name
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "prototype_subnet" {
  name          = var.network.subnet_name
  ip_cidr_range = "10.0.0.0/24"
  region        = "asia-northeast1"
  network       = google_compute_network.prototype_vpc.id
}

# resource "google_compute_global_address" "private_ip_address" {
#   name          = "private-ip-address"
#   purpose       = "VPC_PEERING"
#   address_type  = "INTERNAL"
#   prefix_length = 16
#   network       = google_compute_network.prototype_vpc.id
# }

# resource "google_service_networking_connection" "private_vpc_connection" {
#   network                 = google_compute_network.prototype_vpc.id
#   service                 = "servicenetworking.googleapis.com"
#   reserved_peering_ranges = [google_compute_global_address.private_ip_address.name]
# }

# resource "google_vpc_access_connector" "connector" {
#   name          = var.network.vpc_access_name
#   ip_cidr_range = "10.8.0.0/28"
#   network       = google_compute_network.prototype_vpc.name
# }