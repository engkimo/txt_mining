resource "google_service_account" "default" {
  account_id   = "service-account-id"
  display_name = "Service Account"
}

resource "google_container_cluster" "prototype_cluster" {
  name     = var.gke.cluster_name
  location = var.gke.location
  network = google_compute_network.prototype_vpc.id
  subnetwork = google_compute_subnetwork.prototype_subnet.id

  // remove_default_node_pool = true
  enable_autopilot         = true
  private_cluster_config {
      enable_private_endpoint = false
      enable_private_nodes = true
      master_ipv4_cidr_block = var.gke.master_ipv4_cidr_block
  }

  master_authorized_networks_config {
      cidr_blocks {
          cidr_block = var.master_authorized_ip
      }
  }

  ip_allocation_policy {
  }
}